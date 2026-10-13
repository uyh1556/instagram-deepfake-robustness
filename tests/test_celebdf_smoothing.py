"""Toy flow checks, not reported experiment performance."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_celebdf_smoothing as evaluation
import train_celebdf_baseline as baseline


class EvaluationTests(unittest.TestCase):
    def test_prepare_two_models_four_conditions_reuse(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/'data';data.mkdir()
            rows=[]
            rng=np.random.default_rng(42)
            for label in [0,1]:
                for n,status in enumerate(['detected','fallback_letterbox']):
                    path=f'{label}_{n}.png'
                    Image.fromarray(rng.integers(0,256,(40,40,3),dtype=np.uint8)).save(data/path)
                    rows.append(dict(video_id=f'class{label}/video{n}',class_index=str(label),
                                     roi_status=status,crop_path=path,label='real' if label==0 else 'fake'))
            frame=pd.DataFrame(rows)
            split={'manifest_sha256':'fixture','protocol':'fixture'}
            cfg=json.loads((evaluation.ROOT/'configs/celebdf_smoothing_evaluation_v1.json').read_text())
            provenance={'fixture':True}
            args=argparse.Namespace(data_root=data,manifest_root=root/'manifests',cache_root=root/'cache',
                output_root=root/'results',runs_root=root/'runs',models=['xception','clip_b16'],
                seed=42,workers=0,batch_size=2,bootstrap_repeats=8,device='cpu')
            def toy(name,pretrained=False):
                return nn.Sequential(nn.Flatten(),nn.Linear(3*8*8,2)),transforms.Compose([
                    transforms.Resize((8,8)),transforms.ToTensor()])
            for name in args.models:
                run=args.runs_root/f'{name}_celebdf_identity_v1_seed42';run.mkdir(parents=True)
                model,_=toy(name)
                torch.save({'config':{'split':split,'model':name,'seed':42},'model_state_dict':model.state_dict()},run/'best.pt')
            fake_full=frame.assign(split='test')
            with patch.object(evaluation,'inputs',return_value=(frame,split,cfg,provenance)),\
                 patch.object(baseline,'load_manifest',return_value=(fake_full,split)),\
                 patch.object(baseline,'model_and_transform',side_effect=toy):
                evaluation.prepare(args)
                self.assertFalse((args.cache_root/'smooth_weak'/'0_1.png').exists())
                self.assertTrue((args.cache_root/'smooth_weak'/'0_0.png').exists())
                evaluation.evaluate(args)
                evaluation.evaluate(args)
            table=pd.read_csv(args.output_root/'condition_summary.csv')
            delta=pd.read_csv(args.output_root/'paired_deltas.csv')
            self.assertEqual(len(table),8)
            self.assertEqual(len(delta),6)
            self.assertTrue(table.unchanged_fallback_images.eq(2).all())


if __name__=='__main__':
    unittest.main()
