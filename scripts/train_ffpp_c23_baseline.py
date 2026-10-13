"""Native ROI16 FF++ c23 baselines; no target data, JPEG, smoothing, or UDA."""
import os
os.environ.setdefault('USE_TF','0')
import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,roc_auc_score
from tqdm import tqdm

import train_celebdf_baseline as common

ROOT=Path(__file__).resolve().parents[1]
METHODS=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
DINO_REVISION='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
PREPROCESS={
 'xception':{'size':299,'resize':'bicubic direct square','mean':[.5]*3,'std':[.5]*3},
 'clip_b16':{'size':224,'resize':'OpenAI CLIP bicubic short-side + center crop','mean':[.48145466,.4578275,.40821073],'std':[.26862954,.26130258,.27577711]},
 'dinov2_b14':{'size':224,'resize':'bicubic direct square, no extra center crop','mean':[.485,.456,.406],'std':[.229,.224,.225]}}


class DinoClassifier(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder;self.head=nn.Linear(encoder.config.hidden_size,2)
    def forward(self,x):
        return self.head(self.encoder(pixel_values=x).last_hidden_state[:,0])


def build_model(name,pretrained=True):
    if name!='dinov2_b14':return common.model_and_transform(name,pretrained)
    from transformers import Dinov2Config,Dinov2Model
    if pretrained:
        encoder=Dinov2Model.from_pretrained('facebook/dinov2-base',revision=DINO_REVISION,use_safetensors=True)
    else:
        encoder=Dinov2Model(Dinov2Config(image_size=518,patch_size=14,layer_norm_eps=1e-6))
    t=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
                          transforms.ToTensor(),transforms.Normalize(PREPROCESS[name]['mean'],PREPROCESS[name]['std'])])
    return DinoClassifier(encoder),t


def head(model,name):return model.get_classifier() if name=='xception' else model.head


def selected_modules(model,name,blocks):
    if name=='clip_b16':return [*list(model.visual.transformer.resblocks)[-blocks:],model.visual.ln_post]
    if name=='dinov2_b14':return [*list(model.encoder.encoder.layer)[-blocks:],model.encoder.layernorm]
    return []


def stage(model,name,blocks,warmup):
    for p in model.parameters():p.requires_grad=False
    for p in head(model,name).parameters():p.requires_grad=True
    model.eval();head(model,name).train()
    if warmup:return
    if name=='xception':
        for p in model.parameters():p.requires_grad=True
        model.train()
    else:
        for module in selected_modules(model,name,blocks):
            module.train()
            for p in module.parameters():p.requires_grad=True
        if name=='clip_b16':model.visual.proj.requires_grad=True


def video_balanced(frame,n,seed):
    frame=frame.sample(frac=1,random_state=seed).copy()
    videos=list(frame.video_id.unique());random.Random(seed).shuffle(videos)
    order={v:i for i,v in enumerate(videos)}
    frame['_round']=frame.groupby('video_id',sort=False).cumcount()
    frame['_video_order']=frame.video_id.map(order)
    if len(frame)<n:raise ValueError('Insufficient method images')
    return frame.sort_values(['_round','_video_order'],kind='stable').head(n).drop(columns=['_round','_video_order'])


def epoch_sample(pool,seed):
    real=pool[pool.method.eq('original')]
    n=len(real)//4*4
    if n<4:raise ValueError('No usable Real training pool')
    parts=[video_balanced(real,n,seed)]
    for i,m in enumerate(METHODS):parts.append(video_balanced(pool[pool.method.eq(m)],n//4,seed+i+1))
    return pd.concat(parts,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)


def load_data(args):
    frames=[]
    for part in ['trainval','test']:
        path=args.data_root/'manifests'/f'ffpp_c23_4methods_roi16_{part}_v1.csv'
        frames.append(pd.read_csv(path,dtype=str,keep_default_na=False))
    frame=pd.concat(frames,ignore_index=True)
    protocol_path=args.data_root/'manifests/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    if protocol['name']!='ffpp_c23_4methods_roi16_v1':raise ValueError('Wrong dataset protocol')
    if frame.crop_path.duplicated().any() or frame.groupby('video_id').split.nunique().gt(1).any():raise ValueError('Duplicate image/video leakage')
    membership={}
    for r in frame[['source_ids','split']].drop_duplicates().itertuples(index=False):
        for identity in r.source_ids.split('|'):
            if identity in membership and membership[identity]!=r.split:raise ValueError('Source/pair split leakage')
            membership[identity]=r.split
    for split in ['train','val','test']:
        subset=frame[frame.split.eq(split)]
        if set(subset.method)!=set(['original',*METHODS]):raise ValueError('Split lacks a method')
        for m in ['original',*METHODS]:
            if len(subset[subset.method.eq(m)])!=protocol['counts'][split+'/'+m]:raise ValueError('Manifest count differs from TAR protocol')
    if not frame.class_index.astype(int).eq(frame.method.ne('original').astype(int)).all():raise ValueError('Labels/provenance mismatch')
    for path in frame.crop_path:
        relative=Path(path)
        if relative.is_absolute() or '..' in relative.parts or not (args.data_root/relative).is_file():raise FileNotFoundError(path)
    hashes={'trainval':common.digest(args.data_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv'),
            'test':common.digest(args.data_root/'manifests/ffpp_c23_4methods_roi16_test_v1.csv'),
            'protocol':common.digest(protocol_path)}
    return frame,hashes


def binary_metrics(y,p):
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'auc':float(roc_auc_score(y,p)),'accuracy':float(accuracy_score(y,pred)),
            'f1':float(f1_score(y,pred,zero_division=0)),'real_fpr':float(fp/(tn+fp)),
            'fake_fnr':float(fn/(tp+fn)),'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}


def frame_metrics(frame,p):
    y=frame.class_index.astype(int).to_numpy();overall=binary_metrics(y,p)
    per={}
    for method in METHODS:
        mask=frame.method.isin(['original',method]).to_numpy()
        per[method]=binary_metrics(y[mask],p[mask])
    return {'images':len(frame),'threshold':.5,'frame':overall,'per_method':per,
            'macro_method_auc':sum(v['auc'] for v in per.values())/4}


def amp_context(args):
    # No FP16: use BF16 on supported GPUs, otherwise full FP32.
    return torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=args.device=='cuda' and torch.cuda.is_bf16_supported())


def evaluate(model,frame,transform,args,title):
    model.eval();prob=np.empty(len(frame));total=0
    loader=common.make_loader(common.Crops(frame,args.data_root,transform),args)
    with torch.inference_mode():
        for x,y,ix in tqdm(loader,desc=title,mininterval=5,dynamic_ncols=True):
            x,y=x.to(args.device),y.to(args.device)
            with amp_context(args):logits=model(x)
            if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite evaluation logits')
            logits=logits.float();loss=nn.functional.cross_entropy(logits,y)
            prob[ix.numpy()]=logits.softmax(1)[:,1].cpu().numpy();total+=float(loss)*len(y)
    values=frame_metrics(frame,prob);values['loss']=total/len(frame)
    predictions=frame[['video_id','method','class_index','crop_path','roi_status']].copy()
    predictions['probability_fake']=prob
    return values,predictions


def configuration(args,hashes):
    cfg={'protocol':'ffpp_c23_4methods_baseline_v1','model':args.model,'seed':args.seed,'data_hashes':hashes,
         'epochs':args.epochs,'batch_size':args.batch_size,'patience':args.patience,'min_delta':1e-4,
         'warmup_epochs':args.warmup_epochs,'train_blocks':args.train_blocks if args.model!='xception' else 'all',
         'encoder_learning_rate':args.encoder_lr,'head_learning_rate':args.head_lr,'weight_decay':args.weight_decay,
         'preprocessing':PREPROCESS[args.model],'augmentation':'horizontal flip0.5 only; no JPEG/noise/smoothing',
         'sampling':'same deterministic epoch list across backbones; Real:Fake1:1; four fake methods equal; video-balanced without replacement',
         'checkpoint_selection':'clean FF++ validation macro4 frame ROC-AUC; no target/test model selection',
         'precision':'BF16 if supported, otherwise FP32; finite loss + gradient checks',
         'source_code_sha256':common.digest(Path(__file__)),'shared_helper_sha256':common.digest(Path(common.__file__)),
         'versions':{x:importlib.metadata.version(x) for x in ['torch','torchvision','timm','Pillow','numpy','pandas','scikit-learn']}}
    if args.model=='dinov2_b14':cfg.update(pretrained='facebook/dinov2-base',pretrained_revision=DINO_REVISION,transformers=importlib.metadata.version('transformers'))
    elif args.model=='clip_b16':cfg.update(pretrained='OpenAI CLIP ViT-B/16',clip_version=importlib.metadata.version('clip'))
    else:cfg['pretrained']='timm legacy_xception ImageNet'
    return cfg,hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def run_dir(args):return args.output_root/f'{args.model}_ffpp_c23_roi16_baseline_v1_seed{args.seed}'


def train(args):
    frame,hashes=load_data(args);cfg,identity=configuration(args,hashes);run=run_dir(args)
    run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=cfg:raise RuntimeError('Run config differs; preserve existing run')
    if (run/'completed.json').exists():
        if json.loads((run/'completed.json').read_text())['identity']!=identity or not (run/'best.pt').is_file():raise RuntimeError('Completed run mismatch')
        print('Skip completed:',run,flush=True);return
    common.json_write(run/'config.json',cfg);common.seed_all(args.seed)
    resumed=(run/'last.pt').is_file()
    model,transform=build_model(args.model,pretrained=not resumed);model.to(args.device)
    stage(model,args.model,args.train_blocks,False)
    head_ids={id(p) for p in head(model,args.model).parameters()}
    encoder_params=[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer=torch.optim.AdamW([{'params':encoder_params,'lr':args.encoder_lr},
                                {'params':list(head(model,args.model).parameters()),'lr':args.head_lr}],weight_decay=args.weight_decay)
    start,best,anchor,stale,history=0,-1.,-1.,0,[]
    if resumed:
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if state['identity']!=identity:raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model_state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start,best,anchor,stale,history=(state[k] for k in ['epoch','best','anchor','stale','history']);del state
        print('Resume after completed epoch:',start,flush=True)
    pool=frame[frame.split.eq('train')].reset_index(drop=True);val=frame[frame.split.eq('val')].reset_index(drop=True)
    sampled=epoch_sample(pool,args.seed)
    print('Epoch image counts:',sampled.method.value_counts().to_dict(),'Validation:',len(val),flush=True)
    print('Model:',args.model,'precision:',cfg['precision'],'run:',run,flush=True)
    for epoch in range(start,args.epochs):
        if stale>=args.patience:break
        epoch_seed=args.seed+epoch*100003;common.seed_all(epoch_seed)
        warmup=epoch<args.warmup_epochs;stage(model,args.model,args.train_blocks,warmup)
        sampled=epoch_sample(pool,epoch_seed);loader=common.make_loader(common.Crops(sampled,args.data_root,transform,True,epoch_seed),args,True,epoch_seed)
        total,count=0.,0
        print(f'Epoch {epoch+1}/{args.epochs}: '+('classifier warmup' if warmup else 'fine-tuning'),flush=True)
        for x,y,_ in tqdm(loader,desc='Train',mininterval=5,dynamic_ncols=True):
            x,y=x.to(args.device),y.to(args.device);optimizer.zero_grad(set_to_none=True)
            with amp_context(args):logits=model(x)
            loss=nn.functional.cross_entropy(logits.float(),y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; last completed epoch preserved')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(y);count+=len(y)
        values,_=evaluate(model,val,transform,args,'Validation');auc=values['macro_method_auc']
        improved=auc>best;best=max(best,auc)
        if auc>anchor+1e-4:anchor,stale=auc,0
        else:stale+=1
        history.append({'epoch':epoch+1,'stage':'warmup' if warmup else 'fine_tune','train_loss':total/count,'val':values,
                        'epoch_sample_sha256':hashlib.sha256('\n'.join(sampled.crop_path).encode()).hexdigest()})
        state={'identity':identity,'config':cfg,'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
               'epoch':epoch+1,'best':best,'anchor':anchor,'stale':stale,'history':history}
        if improved:common.save(run/'best.pt',{k:state[k] for k in ['identity','config','model_state_dict','epoch','best']})
        common.save(run/'last.pt',state);del state
        common.json_write(run/'history.json',history)
        pd.DataFrame([{'epoch':h['epoch'],'stage':h['stage'],'train_loss':h['train_loss'],'val_macro_auc':h['val']['macro_method_auc'],
                      'val_real_fpr':h['val']['frame']['real_fpr'],'val_fake_fnr':h['val']['frame']['fake_fnr']} for h in history]).to_csv(run/'history.csv',index=False)
        print(f'Val macro AUC={auc:.4f} | Real FPR={values["frame"]["real_fpr"]:.2%} | best={best:.4f} | patience={stale}/{args.patience}',flush=True)
    if not (run/'best.pt').is_file():raise RuntimeError('No completed epoch')
    common.json_write(run/'completed.json',{'identity':identity,'best_val_macro_auc':best,'epochs_completed':len(history)})
    print('Training completed:',run,flush=True)


def test(args):
    frame,hashes=load_data(args);run=run_dir(args);checkpoint=run/'best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=state['config']
    if cfg['model']!=args.model or cfg['seed']!=args.seed or cfg['data_hashes']!=hashes:raise ValueError('Test model/seed/data mismatch')
    if (run/'test_metrics.json').exists() and (run/'test_predictions.csv').exists():
        saved=json.loads((run/'test_metrics.json').read_text())
        if saved.get('checkpoint_sha256')==common.digest(checkpoint):print('Reuse completed FF++ test:',run,flush=True);return
    model,transform=build_model(args.model,pretrained=False)
    model.load_state_dict(state['model_state_dict'],strict=True);del state;model.to(args.device)
    test_frame=frame[frame.split.eq('test')].reset_index(drop=True)
    values,pred=evaluate(model,test_frame,transform,args,'FF++ Test')
    values.update(model=args.model,checkpoint_sha256=common.digest(checkpoint),data_hashes=hashes,
                  protocol='FF++ c23 custom source-aware test; NOT official split',checkpoint_selection=cfg['checkpoint_selection'])
    pred.to_csv(run/'test_predictions.csv',index=False);common.json_write(run/'test_metrics.json',values)
    print('Test macro4 AUC:',round(values['macro_method_auc'],4),'Real FPR:',f'{values["frame"]["real_fpr"]:.2%}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','test'])
    p.add_argument('--model',choices=list(PREPROCESS),required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--patience',type=int,default=4);p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--train-blocks',type=int,default=2);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--encoder-lr',type=float)
    p.add_argument('--head-lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    args.encoder_lr=args.encoder_lr or (1e-4 if args.model=='xception' else 1e-5)
    if not 1<=args.train_blocks<=12 or args.epochs<=args.warmup_epochs or args.warmup_epochs<0 or args.patience<1 or args.batch_size<1:raise ValueError('Invalid training settings')
    common.seed_all(args.seed);(train if args.action=='train' else test)(args)


if __name__=='__main__':main()
