"""Small CPU protocol tests; no claims about detector performance."""
import ast
import copy
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_preservation as b


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(3*12*12, 4))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.encoder(x))


def build(*args, **kwargs):
    return Tiny(), transforms.Compose([transforms.Resize((12, 12)), transforms.ToTensor()])


def outputs(name, model, x):
    f = model.encoder(x)
    return model.fc(f), f


class PreservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='preservation-test-')
        self.root = root = Path(self.temp.name)
        self.args = a = Namespace(source_root=root/'ff', target_root=root/'cdf', baseline_runs_root=root/'baseline',
            runs_root=root/'runs', manifest_root=root/'manifests', feature_cache_root=root/'features',
            output_root=root/'results', cache_root=root/'cache', baseline_evaluation_root=root/'baseline_eval',
            reference_evaluations=root/'old_aug', source_validation=None, protocol=root/'protocol.json',
            models=list(b.base.PREPROCESS), seed=42, source_images=16, val_images=16, batch_size=4,
            workers=0, device='cpu', epochs=2, patience=4, hidden=8, bound=2., learning_rate=.001,
            weight_decay=.0001, pair_weight=1., clean_weight=1., clean_ber_tolerance=.01,
            clean_error_tolerance=.02, auc_tolerance=.005)
        for folder in [a.source_root, a.target_root]:
            (folder/'manifests').mkdir(parents=True)
        rng = np.random.default_rng(42)
        rows = []
        for split in ['train', 'val']:
            for k, method in enumerate(['original', *b.base.METHODS]):
                for i in range(8):
                    path = f'{split}_{k}_{i}.png'
                    Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(a.source_root/path)
                    rows.append(dict(crop_path=path, video_id=f'{split}_{k}_{i//2}', source_ids=f'{split}_{i//2}',
                        split=split, class_index=str(int(k > 0)), method=method,
                        roi_status='detected' if i else 'fallback_letterbox'))
        source = a.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv'
        pd.DataFrame(rows).to_csv(source, index=False)
        protocol = a.source_root/'manifests/protocol.json'; protocol.write_text('{}')
        self.initial_hashes = {}
        for name in a.models:
            run = a.baseline_runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed42'; run.mkdir(parents=True)
            cfg = dict(protocol='ffpp_c23_4methods_baseline_v1', model=name, seed=42,
                preprocessing=b.base.PREPROCESS[name], train_blocks='all' if name == 'xception' else 2,
                data_hashes={'trainval': b.base.common.digest(source), 'protocol': b.base.common.digest(protocol)})
            torch.manual_seed(42)
            torch.save(dict(config=cfg, identity=b.evaluation.key(cfg), model_state_dict=Tiny().state_dict()), run/'best.pt')
            (run/'config.json').write_text(json.dumps(cfg))
            (run/'completed.json').write_text(json.dumps({'identity': b.evaluation.key(cfg)}))
            self.initial_hashes[name] = b.base.common.digest(run/'best.pt')
        rows = []
        for y in [0, 1]:
            for i in range(4):
                path = f'test_{y}_{i}.png'
                Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(a.target_root/path)
                rows.append(dict(crop_path=path, video_id=f'test_{y}_{i//2}', class_index=str(y), split='test',
                                 roi_status='detected' if i else 'fallback_letterbox'))
        self.test = a.target_root/'manifests/celebdf_v2_roi16_test_v1.csv'
        pd.DataFrame(rows).to_csv(self.test, index=False)
        self.cfg = json.loads((b.ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json').read_text())
        self.cfg.update(test_images=8, test_videos=4, real_images=4, fake_images=4)
        a.protocol.write_text(json.dumps(self.cfg))
        self.patches = [patch.object(b.base, 'build_model', side_effect=build),
                        patch.object(b.adaptation, 'outputs', side_effect=outputs),
                        patch.dict(b.adaptation.FEATURES, dict.fromkeys(a.models, 4))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_loss_zero_init_bound_and_backbone_freeze(self):
        c = b.Correction(4, hidden=8, scale=2., bound=2.)
        f, margins = torch.randn(6, 2, 4), torch.randn(6, 2)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        total, ce, restore, anchor = b.loss_for(c, f, margins, y, 1., 1.)
        expected = .5*(nn.functional.binary_cross_entropy_with_logits(margins[:, 0], y.float())
                      + nn.functional.binary_cross_entropy_with_logits(margins[:, 1], y.float()))
        self.assertTrue(torch.equal(c(f[:, 0], margins[:, 0]), margins[:, 0]))
        self.assertTrue(torch.allclose(ce, expected))
        self.assertTrue(torch.allclose(restore, nn.functional.smooth_l1_loss(margins[:, 1]/2, margins[:, 0]/2)))
        self.assertEqual(anchor.item(), 0.)
        total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters()))
        with torch.no_grad():
            c.net[-1].bias.fill_(1e4)
        self.assertTrue(((c(f[:, 0], margins[:, 0])-margins[:, 0]).abs() <= 4.00001).all())
        a = self.args
        state, _ = b.evaluation.checkpoint(Namespace(runs_root=a.baseline_runs_root, seed=42), 'xception')
        model, _ = b.frozen_model(a, 'xception', state)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        _, val, _ = b.source_inputs(a)
        a.source_validation = self.root/'previous_val.csv'; val.to_csv(a.source_validation, index=False)
        b.source_inputs(a)
        bad = val.copy(); bad.loc[0, 'class_index'] = '9'; bad.to_csv(a.source_validation, index=False)
        with self.assertRaises(ValueError):
            b.source_inputs(a)

    def test_three_models_train_skip_evaluate_reuse_and_selection(self):
        a = self.args
        # There are no CDF non-test manifests: training must never request them.
        b.train(a)
        b.train(a)
        for name in a.models:
            run = a.runs_root/b.run_name(name, 42)
            best = torch.load(run/'best.pt', weights_only=True)
            self.assertNotIn('model_state_dict', best)
            self.assertLessEqual(best['best'], best['reference']['mean_ber'])
            self.assertTrue(b.eligible(best['validation'], best['reference'], a))
            original = a.baseline_runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed42/best.pt'
            self.assertEqual(b.base.common.digest(original), self.initial_hashes[name])
        ea = Namespace(data_root=a.target_root, manifest=self.test, protocol=a.protocol, cache_root=a.cache_root,
            output_root=a.baseline_evaluation_root, runs_root=a.baseline_runs_root, models=a.models,
            seed=42, batch_size=4, workers=0, device='cpu')
        b.evaluation.prepare(ea); b.evaluation.evaluate(ea)
        a.reference_evaluations.mkdir()
        (a.reference_evaluations/'evaluation_protocol.json').write_bytes((a.baseline_evaluation_root/'evaluation_protocol.json').read_bytes())
        for name in a.models:
            for condition in self.cfg['conditions']:
                old = json.loads((a.baseline_evaluation_root/f'{name}_{condition}.json').read_text())
                old['strategy'] = 'augmentation'
                (a.reference_evaluations/f'{name}_augmentation_{condition}.json').write_text(json.dumps(old))
        b.evaluate(a)
        before = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*preservation_b*.csv')}
        b.evaluate(a)
        after = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*preservation_b*.csv')}
        self.assertEqual(before, after)
        table = pd.read_csv(a.output_root/'comparison.csv')
        self.assertEqual(len(table), 36)
        self.assertTrue(np.allclose(table.balanced_error_rate, .5*(table.real_fpr+table.fake_fnr)))
        for name in a.models:
            clean = pd.read_csv(a.output_root/f'{name}_preservation_b_normal.csv').set_index('crop_path')
            strong = pd.read_csv(a.output_root/f'{name}_preservation_b_smooth_strong.csv').set_index('crop_path')
            old = pd.read_csv(a.baseline_evaluation_root/f'{name}_normal.csv').set_index('crop_path')
            self.assertTrue(np.array_equal(clean.probability_fake, old.probability_fake))
            mask = clean.roi_status.ne('detected')
            self.assertTrue(np.array_equal(clean.loc[mask, 'probability_fake'], strong.loc[mask, 'probability_fake']))
            worst = copy.deepcopy(torch.load(a.runs_root/b.run_name(name, 42)/'best.pt', weights_only=True)['reference'])
            worst['normal']['frame']['fake_fnr'] = 1.
            self.assertFalse(b.eligible(worst, torch.load(a.runs_root/b.run_name(name, 42)/'best.pt', weights_only=True)['reference'], a))
        # Exercise evaluation of a genuinely nonzero saved correction as well.
        for name in a.models:
            path = a.runs_root/b.run_name(name, 42)/'best.pt'
            state = torch.load(path, weights_only=True)
            state['correction_state_dict']['net.2.bias'].fill_(.05)
            state['epoch'] = 1
            torch.save(state, path)
        a.output_root = self.root/'nonzero_results'
        b.evaluate(a)
        for name in a.models:
            values = pd.read_csv(a.output_root/f'{name}_preservation_b_normal.csv')
            self.assertTrue(values.correction.gt(0).all())

    def test_exact_resume_and_notebook(self):
        a = self.args; a.models = ['xception']
        b.train(a)
        uninterrupted = torch.load(a.runs_root/b.run_name('xception', 42)/'last.pt', weights_only=True)
        a.runs_root = self.root/'resume'
        original, calls = b.validate, [0]
        def interrupt(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError('simulated interruption')
            return original(*args, **kwargs)
        with patch.object(b, 'validate', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                b.train(a)
        b.train(a)
        resumed = torch.load(a.runs_root/b.run_name('xception', 42)/'last.pt', weights_only=True)
        for k, v in uninterrupted['correction_state_dict'].items():
            self.assertTrue(torch.equal(v, resumed['correction_state_dict'][k]), k)
        self.assertEqual(uninterrupted['history'], resumed['history'])
        from IPython.core.inputtransformer2 import TransformerManager
        nb = json.loads((b.ROOT/'notebooks/train_test_ffpp_c23_preservation_b_seed42.ipynb').read_text())
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 8)
        for cell in cells:
            code = ''.join(cell['source'])
            self.assertFalse(code.lstrip().startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(code))


if __name__ == '__main__':
    unittest.main()
