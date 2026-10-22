"""Small CPU checks for fixed-schedule teacher-protection UDA on pool v2."""
import ast
import copy
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_uda_teacher_pool_v2 as method
import test_smoothing_rank_sda_pool_v2 as fixture
import test_smoothing_preservation as tiny


def state(path, field='model_state_dict'):
    return torch.load(path, weights_only=True)[field]


def same(a, b):
    return all(torch.equal(v, b[k]) for k, v in a.items())


class TeacherPoolV2Tests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def configure(self, runs, strategies=None, epochs=2):
        a = copy.copy(self.args)
        a.strategies = strategies or list(method.STRATEGIES)
        a.runs_root, a.manifest_root, a.output_root = self.root/runs, self.root/f'{runs}_manifests', self.root/f'{runs}_results'
        a.epochs, a.pseudo_threshold, a.pool_reference, a.reference_roots = epochs, .5, None, []
        return a

    def test_snd_headview_and_decoupled_gradients(self):
        p = np.r_[np.full(50, .01), np.full(50, .99)]
        # Two tight prediction clusters: each point's neighbours spread over its 49 identical peers.
        self.assertAlmostEqual(method.snd(p), np.log(49), places=4)
        self.assertEqual(method.snd(p, seed=1), method.snd(p, seed=1))
        model, _ = tiny.build()
        aux = copy.deepcopy(model.fc)
        x = torch.rand(3, 3, 12, 12)
        with patch.object(method.adaptation, 'outputs', side_effect=tiny.outputs):
            self.assertTrue(torch.equal(method.HeadView('clip_b16', model, model.fc)(x), model(x)))
            _, features = method.adaptation.outputs('clip_b16', model, x)
            loss, _, _ = method.uda.confidence_loss(aux(features), torch.tensor([[3., -3.]]*3), .5)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.fc.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))

    def test_controls_reproduce_previous_runs(self):
        a = self.configure('teacher', ['ce_rank_fixed', 'ts_ema', 'ts_frozen_teacher'])
        method.prepare(a)
        method.train(a)
        ce = copy.copy(self.args); ce.strategies = ['ce_rank_pooled']
        ce.runs_root, ce.manifest_root = self.root/'ce', self.root/'ce_manifests'
        method.pool.train(ce)
        self.assertTrue(same(state(ce.runs_root/method.pool.run_name('dinov2_b14', 'ce_rank_pooled', 42)/'last.pt'),
                             state(a.runs_root/method.run_name('dinov2_b14', 'ce_rank_fixed', 42)/'final.pt')))
        ts = copy.copy(a); ts.strategies = ['ts_confidence']
        ts.runs_root, ts.manifest_root = self.root/'ts', self.root/'ts_manifests'
        ts.quantile = .25
        method.uda.train(ts)
        self.assertTrue(same(state(ts.runs_root/method.uda.run_name('dinov2_b14', 'ts_confidence', 42)/'last.pt'),
                             state(a.runs_root/method.run_name('dinov2_b14', 'ts_ema', 42)/'final.pt')))
        frozen = a.runs_root/method.run_name('dinov2_b14', 'ts_frozen_teacher', 42)
        baseline = state(a.baseline_runs_root/'dinov2_b14_ffpp_c23_roi16_baseline_v1_seed42/best.pt')
        self.assertTrue(same(baseline, state(frozen/'last.pt', 'teacher_state_dict')))
        history = json.loads((frozen/'history.json').read_text())
        self.assertEqual(len(history), 2)
        self.assertTrue({'mean_entropy', 'snd', 'teacher_student_agreement'} <= set(history[-1]['validation']['label_free']))
        ce_history = json.loads((a.runs_root/method.run_name('dinov2_b14', 'ce_rank_fixed', 42)/'history.json').read_text())
        self.assertTrue(all(h['pseudo_accuracy'] is None for h in ce_history))

    def test_decoupled_resume_and_evaluate(self):
        ref = copy.copy(self.args); ref.strategies = ['ce_rank_pooled', 'sda_new_balanced']
        ref.runs_root, ref.manifest_root, ref.output_root = self.root/'ref', self.root/'ref_manifests', self.root/'ref_results'
        method.pool.train(ref)
        method.pool.evaluate(ref)
        a = self.configure('teacher', ['ce_rank_fixed', 'ts_ema', 'ts_decoupled_head'], epochs=3)
        a.reference_roots, a.pool_reference = [ref.output_root], ref.manifest_root/'pool_v2/summary.json'
        method.train(a)
        run = a.runs_root/method.run_name('dinov2_b14', 'ts_decoupled_head', 42)
        last = torch.load(run/'last.pt', weights_only=True)
        initial_head = {k: v for k, v in state(a.baseline_runs_root/'dinov2_b14_ffpp_c23_roi16_baseline_v1_seed42/best.pt').items() if k.startswith('fc.')}
        self.assertFalse(all(torch.equal(v, last['aux_state_dict'][k[3:]]) for k, v in initial_head.items()))
        self.assertIsNotNone(json.loads((run/'history.json').read_text())[-1]['validation'].get('adapt_val_aux_head'))
        resumed = self.configure('resumed', ['ts_decoupled_head'], epochs=3)
        resumed.pool_reference = a.pool_reference
        save = method.base.common.save
        def interrupt(path, value):
            save(path, value)
            if path.name == 'last.pt' and value['epoch'] == 2:
                raise RuntimeError('intentional interruption')
        with patch.object(method.base.common, 'save', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'intentional'):
            method.train(resumed)
        method.train(resumed)
        restored = torch.load(resumed.runs_root/method.run_name('dinov2_b14', 'ts_decoupled_head', 42)/'last.pt', weights_only=True)
        self.assertEqual(last['history'], restored['history'])
        for field in ['model_state_dict', 'teacher_state_dict', 'aux_state_dict']:
            self.assertTrue(same(last[field], restored[field]))
        method.evaluate(a)
        mix = pd.read_csv(a.output_root/'mixture_summary.csv')
        expected = {'baseline', 'ce_rank_pooled(ref)', 'sda_new_balanced(ref)'} | {s+t for s in a.strategies for t in ['', '@ffpp']}
        self.assertEqual(set(mix.strategy), expected)
        self.assertIn('gap_closure_vs_sda', mix)
        boot = pd.read_csv(a.output_root/'bootstrap.csv')
        self.assertEqual(set(zip(boot.strategy, boot.reference)), {('ts_ema', 'ce_rank_fixed'),
            ('ts_decoupled_head', 'ce_rank_fixed'), ('ts_decoupled_head', 'ts_ema')})
        selections = pd.read_csv(a.output_root/'selections.csv')
        self.assertTrue((selections[~selections.strategy.str.contains('@')].epoch == 3).all())
        times = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        method.evaluate(a)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})

    def test_colab(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_uda_teacher_pool_v2_seed42'
        text = (method.ROOT/'docs/colab'/f'{stem}.md').read_text()
        notebook = json.loads((method.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        cells = re.findall(r'```python\n(.*?)```', text, re.S)
        self.assertEqual(cells, [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'])
        self.assertEqual(len(cells), 7)
        for cell in cells:
            ast.parse(TransformerManager().transform_cell(cell))
        self.assertIn("f'{EXPERIMENT}_{GROUP}'", cells[1])
        self.assertIn('drive_folder_guard.check', cells[1])


if __name__ == '__main__':
    unittest.main()
