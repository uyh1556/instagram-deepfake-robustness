"""Inference-only adapters to reviewed, pinned upstream architecture definitions.

Execute selected original classes, not the large benchmark's detector registry.
All inference weights are loaded strictly; no training or random-weight fallback.
"""
import ast
import json
import logging
import os
import tempfile
import warnings
from pathlib import Path

import cv2
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir())/'celebdf_public_hf_cache'))
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

# Upstream ArcFace uses the old API even with fp16=False; suppress only this repeated warning.
warnings.filterwarnings('ignore', message=r'.*torch\.cuda\.amp\.autocast.*deprecated.*', category=FutureWarning)

PINS = {
    'recce': 'f188b1c105465e2e5377eb536a95022ae0e4522d',
    'ucf': 'f188b1c105465e2e5377eb536a95022ae0e4522d',
    'selfi': 'c20f256a901d1894e4d96e79534a5b17b23e2795',
}


def definitions(path, names, environment):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f'Expected upstream definitions not present: {path}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), environment)


class Registry:
    def register_module(self, module_name):
        return lambda cls: cls


class InputTransform:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32).reshape(1,1,3)
        self.std = np.array(std, dtype=np.float32).reshape(1,1,3)

    def __call__(self, image):
        # Upstream benchmark load_rgb uses cv2.INTER_CUBIC; ToTensor + Normalize.
        array = cv2.resize(np.asarray(image.convert('RGB')), (self.size,self.size), interpolation=cv2.INTER_CUBIC)
        array = (array.astype(np.float32)/255. - self.mean)/self.std
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1)))


def read_state(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or not all(torch.is_tensor(v) for v in value.values()):
        raise ValueError('Expected verified public raw state_dict, not a different checkpoint format')
    return {k.removeprefix('module.'): v for k,v in value.items()}


def recce(repo, checkpoint):
    env = {'torch':torch, 'nn':nn, 'F':F}
    definitions(repo/'training/networks/xception.py', ['SeparableConv2d','Block'], env)
    env['encoder_params'] = {'xception': {'features':2048,
        'init_op': lambda: timm.create_model('legacy_xception', pretrained=False)}}
    definitions(repo/'training/detectors/recce_detector.py', ['Recce','GraphReasoning','GuidedAttention'], env)
    model = env['Recce'](num_classes=2)
    state = read_state(checkpoint)
    if not all(k.startswith(('model.','backbone.')) for k in state):
        raise ValueError('Unexpected RECCE checkpoint keys')
    # The benchmark RecceDetector explicitly never uses self.backbone; only self.model.
    active = {k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')}
    model.load_state_dict(active, strict=True)
    return model, InputTransform(256, [.5]*3, [.5]*3)


def selfi(repo, checkpoint):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    source = repo/'training/networks/iresnet_network/iresnet.py'
    namespace = {}
    exec(compile(source.read_text(), str(source), 'exec'), namespace)
    class Identity(nn.Module):
        def __init__(self, unused_path):
            super().__init__()
            self.iresnet = namespace['iresnet100']()
        def features(self, x):
            return self.iresnet(x)
    env = {'torch':torch, 'nn':nn, 'F':F, 'logger':logging.getLogger('selfi'),
           'AbstractDetector':nn.Module, 'DETECTOR':Registry(),
           'BACKBONE':{'selfi_iresnet100':Identity}, 'LOSSFUNC':{'cross_entropy':nn.CrossEntropyLoss},
           'get_clip_visual': lambda model_name: (None, CLIPVisionModel(CLIPVisionConfig(patch_size=16)).vision_model)}
    definitions(repo/'training/detectors/selfi_detector.py', ['SELFIDetector'], env)
    config = {'backbone_name':'clip','face_model_name':'selfi_iresnet100',
              'face_model_pretrained':'included_in_full_checkpoint',
              'backbone_config': {'IAFM':True, 'fusion_method':'weighted_sum', 'relevance_loss':False,
                  'cosine_diversity':False,'relevance_predictor':'concat','use_identity_real_fake_head':True},
              'loss_func':{'cls_loss':'cross_entropy'},'loss_weight':{'cls_weight':1}}
    model = env['SELFIDetector'](config)
    model.load_state_dict(read_state(checkpoint), strict=True)
    # Public SELFI_clip.yaml uses ImageNet mean/std, NOT the plain CLIP transform.
    return model, InputTransform(224, [.485,.456,.406], [.229,.224,.225])


def ucf(repo, checkpoint):
    env = {'torch':torch,'nn':nn,'F':F,'np':np,'logger':logging.getLogger('ucf'),
           'AbstractDetector':nn.Module,'DETECTOR':Registry(),'BACKBONE':Registry()}
    definitions(repo/'training/networks/xception.py', ['SeparableConv2d','Block','Xception'], env)
    definitions(repo/'training/detectors/ucf_detector.py',
                ['UCFDetector','sn_double_conv','r_double_conv','AdaIN','Conditional_UNet','MLP','Conv2d1x1','Head'],env)
    class InferenceUCF(env['UCFDetector']):
        def build_backbone(self, config):
            return env['Xception'](config['backbone_config'])
        def build_loss(self, config):
            return {}  # Training losses not executed; they contain no inference parameters.
    config={'backbone_config':{'num_classes':2,'mode':'adjust_channel','inc':3,'dropout':False},
            'encoder_feat_dim':512,'train_dataset':['FF-F2F','FF-DF','FF-FS','FF-NT']}
    model=InferenceUCF(config)
    model.load_state_dict(read_state(checkpoint),strict=True)
    return model, InputTransform(256,[.5]*3,[.5]*3)


def load(name, repo, checkpoint):
    return {'recce':recce,'ucf':ucf,'selfi':selfi}[name](Path(repo),Path(checkpoint))


def probabilities(name, model, images):
    if name == 'recce':
        return model(images).float().softmax(1)[:,1]
    if name == 'ucf':
        # Upstream requires labels only to update bookkeeping, not for its prediction.
        dummy=torch.zeros(len(images),dtype=torch.long,device=images.device)
        return model({'image':images,'label':dummy},inference=True)['cls'].float().softmax(1)[:,1]
    return model({'image':images}, inference=True)['prob'].float()
