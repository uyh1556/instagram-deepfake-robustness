"""Small CPU flow tests; toy metrics are NOT experiment results."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import create_celebdf_identity_split as split
import train_celebdf_baseline as training


class ProtocolTests(unittest.TestCase):
    def test_cross_identity_excluded_and_deterministic(self):
        rows = []
        for i in range(20):
            for method, label, vid in [('Celeb-real', 0, f'id{i}_0000'), ('Celeb-synthesis', 1, f'id{i}_id{i}_0000')]:
                rows.append(dict(video_id=f'{method}/{vid}', crop_path=f'{method}/{vid}.png',
                                 method=method, class_index=str(label), split='non_test_pool'))
        base, groups = split.partition(pd.DataFrame(rows))
        a, b = groups['train'][0], groups['test'][0]
        extra = pd.DataFrame([dict(video_id=f'Celeb-synthesis/{a}_{b}_0001', crop_path='cross.png',
                                  method='Celeb-synthesis', class_index='1', split='test')])
        combined, _ = split.partition(pd.concat([pd.DataFrame(rows), extra], ignore_index=True))
        self.assertEqual(combined.iloc[-1]['split'], 'excluded_cross_identity')
        self.assertEqual(split.partition(pd.DataFrame(rows))[0].to_csv(), base.to_csv())

    def test_shared_epoch_sampling_and_training_resume_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifests, runs = [root/x for x in ['data', 'manifests', 'runs']]
            data.mkdir(); manifests.mkdir()
            rows = []
            for i, role in enumerate(['train', 'val', 'test']):
                for label in [0, 1]:
                    method = 'Celeb-real' if label == 0 else 'Celeb-synthesis'
                    video = f'{method}/id{i}_0000' if label == 0 else f'{method}/id{i}_id{i}_0000'
                    for frame in range(2):
                        path = f'{role}_{label}_{frame}.png'
                        Image.new('RGB', (12, 12), (40+label*180,)*3).save(data/path)
                        rows.append(dict(video_id=video, crop_path=path, class_index=str(label),
                                         roi_status='detected', split=role))
            frame = pd.DataFrame(rows)
            frame.to_csv(manifests/'manifest.csv', index=False)
            cfg = {'manifest_sha256': training.digest(manifests/'manifest.csv'), 'protocol': split.PROTOCOL}
            (manifests/'protocol.json').write_text(json.dumps(cfg))
            a = training.balanced_epoch(frame[frame.split.eq('train')], 42)
            self.assertEqual(a.class_index.value_counts().to_dict(), {'0': 2, '1': 2})
            transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
            def toy(name, pretrained=True):
                return nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 2)), transform
            args = argparse.Namespace(model='xception', seed=42, data_root=data, manifest_root=manifests,
                                      output_root=runs, epochs=1, patience=4, batch_size=2, workers=0,
                                      learning_rate=None, head_learning_rate=1e-4, weight_decay=1e-4,
                                      device='cpu', log_interval_seconds=60)
            with patch.object(training, 'model_and_transform', toy):
                training.train(args)
                training.train(args)  # completed skip
                run = runs/'xception_celebdf_identity_v1_seed42'
                (run/'completed.json').unlink()  # only temporary test fixture
                training.train(args)  # last.pt resume
                training.test(args)
            values = json.loads((run/'test_metrics.json').read_text())
            self.assertEqual(values['images'], 4)
            self.assertEqual(values['videos'], 2)


if __name__ == '__main__':
    unittest.main()
