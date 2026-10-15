"""CPU protocol checks for C, not experimental detector performance."""
import ast
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

import test_smoothing_preservation as fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_consistency_c as c


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreservationTests()
        self.fixture.setUp()
        self.args, self.root = a, root = self.fixture.args, self.fixture.root
        self.b_hashes = {str(p): c.base.common.digest(p) for p in a.baseline_runs_root.rglob('*.pt')}
        a.b_evaluation_root = None
        a.runs_root, a.manifest_root, a.output_root = root/'c_runs', root/'c_manifests', root/'c_results'
        a.learning_rate, a.target_images, a.target_weight, a.extra_strength = 1e-3, 8, .1, .25
        a.uda_manifest_dir = root/'old_uda'; a.uda_manifest_dir.mkdir()
        for kind in ['real', 'fake']:
            rows = []
            for i in range(8):
                path = f'{kind}_train_{i}.png'
                rng = np.random.default_rng(i+20*(kind == 'fake'))
                Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(a.target_root/path)
                rows.append(dict(crop_path=path, video_id=f'{kind}_target{i//2}', split='non_test_pool',
                    roi_status='detected' if i else 'fallback_letterbox', class_index=str(int(kind == 'fake'))))
            pd.DataFrame(rows).to_csv(a.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv', index=False)
        pool, val, target, cfg, data, _ = c.b.adaptation.dataset_inputs(a)
        (a.uda_manifest_dir/'protocol.json').write_text(json.dumps(dict(data=data, seed=42,
            source_images_per_epoch=a.source_images, val_images=a.val_images, target_images=a.target_images)))
        target.to_csv(a.uda_manifest_dir/'target_unlabelled.csv', index=False)
        val.to_csv(a.uda_manifest_dir/'source_validation.csv', index=False)

    def tearDown(self):
        self.fixture.tearDown()

    def test_initializer_pair_composition_balance_and_loss(self):
        a = self.args
        dataset_order = torch.utils.data.TensorDataset(torch.arange(16))
        expected = torch.cat([v for (v,) in c.base.common.make_loader(dataset_order, a, True, 42)])
        self.assertTrue(torch.equal(c.source_order(a, 16, 42), expected))
        pool, val, target, cfg, _ = c.mixed.sda.inputs(a)
        sampled = c.b.adaptation.balanced_source(pool, a.source_images, 42)
        _, transform = c.base.build_model('xception', pretrained=False)
        dataset = c.TargetPairs(sampled, target, a, transform, 0, cfg)
        for start in range(0, len(dataset), a.batch_size):
            labels = [r['sampler_class_index'] for r in dataset.metadata()[start:start+a.batch_size]]
            self.assertEqual(labels.count(0), labels.count(1))
        row = target[target.roi_status.eq('detected')].to_dict('records')[0]
        dataset.sampler.plan[0] = (row, 'smooth_strong', False)
        first, second = dataset[0]
        with Image.open(a.target_root/row['crop_path']) as image:
            observed = c.b.adaptation.smooth(image.convert('RGB'), .75, cfg['bilateral'])
            extra = c.b.adaptation.smooth(observed, .25, cfg['bilateral'])
            self.assertTrue(torch.equal(first, transform(observed)))
            self.assertTrue(torch.equal(second, transform(extra)))
        self.assertEqual(len(dataset[0]), 2)  # neither label nor strength is returned
        s = c.b.Correction(4, 8)
        sf, tf = torch.randn(4, 2, 4), torch.randn(4, 2, 4)
        sm, tm, y = torch.randn(4, 2), torch.randn(4, 2), torch.tensor([0, 1, 0, 1])
        with torch.no_grad():
            s.net[-1].bias.fill_(.02)
        losses = c.loss_for(s, sf, sm, y, tf, tm, a)
        source = c.b.loss_for(s, sf, sm, y, 1., 1.)[0]
        observed, extra = s(tf[:, 0], tm[:, 0]), s(tf[:, 1], tm[:, 1])
        consistency = nn.functional.smooth_l1_loss(observed/s.scale, extra/s.scale)
        self.assertTrue(torch.allclose(losses[0], source+.1*consistency))
        losses[0].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in s.parameters()))

    def test_three_model_train_skip_resume_eval_and_notebook(self):
        a = self.args
        c.train(a); c.train(a)
        for name in a.models:
            run = a.runs_root/c.run_name(name, 42)
            state = torch.load(run/'last.pt', weights_only=True)
            self.assertEqual(state['epoch'], 2)
            for h in state['history']:
                self.assertEqual(h['target_real_exposures'], h['target_fake_exposures'])
        self.assertNotIn('class_index', pd.read_csv(a.manifest_root/'target_unlabelled.csv').columns)
        full_root = a.runs_root
        a.models, a.runs_root = ['clip_b16'], self.root/'resume_c'
        original, calls = c.b.validate, [0]
        def interrupt(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError('simulated interruption')
            return original(*args, **kwargs)
        with patch.object(c.b, 'validate', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                c.train(a)
        c.train(a)
        expected = torch.load(full_root/c.run_name('clip_b16', 42)/'last.pt', weights_only=True)
        resumed = torch.load(a.runs_root/c.run_name('clip_b16', 42)/'last.pt', weights_only=True)
        for k, value in expected['correction_state_dict'].items():
            self.assertTrue(torch.equal(value, resumed['correction_state_dict'][k]), k)
        self.assertEqual(expected['history'], resumed['history'])
        a.runs_root, a.models = full_root, list(c.base.PREPROCESS)
        ea = copy.copy(a)
        ea.data_root, ea.manifest, ea.runs_root, ea.output_root = a.target_root, self.fixture.test, a.baseline_runs_root, a.baseline_evaluation_root
        c.evaluation.prepare(ea); c.evaluation.evaluate(ea)
        c.evaluate(a)
        stamps = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*preservation_c*.csv')}
        c.evaluate(a)
        self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*preservation_c*.csv')})
        table = pd.read_csv(a.output_root/'comparison.csv')
        self.assertEqual(len(table[table.strategy.isin(['baseline', 'preservation_c'])]), 24)
        self.assertEqual(self.b_hashes, {str(p): c.base.common.digest(p) for p in a.baseline_runs_root.rglob('*.pt')})
        from IPython.core.inputtransformer2 import TransformerManager
        nb = json.loads((c.b.ROOT/'notebooks/train_test_ffpp_c23_consistency_c_seed42.ipynb').read_text())
        cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(cells), 7)
        for cell in cells:
            code = ''.join(cell['source'])
            self.assertFalse(code.lstrip().startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(code))


if __name__ == '__main__':
    unittest.main()
