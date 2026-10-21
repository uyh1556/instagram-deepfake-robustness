"""Small CPU checks for pool-v2 UDA: pseudo rules, label isolation, resume and evaluation."""
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
import ffpp_c23_smoothing_rank_uda_pool_v2 as method
import test_smoothing_rank_sda_pool_v2 as fixture


def weights(path):
    return torch.load(path, weights_only=True)['model_state_dict']


class UdaPoolV2Tests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def configure(self, runs, strategies=None):
        a = copy.copy(self.args)
        a.strategies = strategies or list(method.STRATEGIES)
        a.runs_root, a.manifest_root, a.output_root = self.root/runs, self.root/f'{runs}_manifests', self.root/f'{runs}_results'
        a.quantile, a.pseudo_threshold, a.pool_reference, a.reference_roots = .25, .5, None, []
        return a

    def test_pseudo_rules(self):
        student = torch.zeros(4, 2, requires_grad=True)
        teacher = torch.tensor([[3., -3.], [0., .1], [-3., 3.], [1., 1.]], requires_grad=True)
        loss, labels, selected = method.confidence_loss(student, teacher, .9)
        legacy, _, _ = method.engine.pseudo_loss(student, teacher, .9)
        self.assertTrue(torch.equal(loss, legacy))
        loss, labels, selected = method.rank_quantile_loss(student, teacher, -1., 1.)
        self.assertEqual(selected.tolist(), [True, False, True, False])
        self.assertEqual(labels[selected].tolist(), [0, 1])
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(student.grad[1], torch.zeros(2)) and torch.equal(student.grad[3], torch.zeros(2)))
        low, high = method.quantile_thresholds(np.arange(100.), .2)
        self.assertAlmostEqual(low, 19.8) ; self.assertAlmostEqual(high, 79.2)
        with self.assertRaises(ValueError):
            method.quantile_thresholds(np.ones(10), .2)
        validation = {'ffpp': {'conditions': {c: {'real_fpr': .1, 'fake_fnr': .1} for c in method.CONDITIONS}}}
        self.assertAlmostEqual(method.bbse_prior(.4, validation)['pi_hat'], .375)

    def test_labels_never_change_training_and_zero_weight_matches_ce_rank(self):
        a = self.configure('uda')
        method.prepare(a)
        method.train(a)
        for strategy in a.strategies:
            history = json.loads((a.runs_root/method.run_name('dinov2_b14', strategy, 42)/'history.json').read_text())
            self.assertTrue(all(h['target_exposures'] == 16 and h['bbse']['pi_true_exposures'] is not None for h in history))
            if strategy == 'ts_rank_quantile':
                self.assertTrue(all(h['rank_thresholds'][0] < h['rank_thresholds'][1] for h in history))
        frozen = pd.read_csv(a.manifest_root/'target_exposures/epoch01.csv')
        self.assertFalse({'class_index', 'kind', 'partition'} & set(frozen))
        # Inverting the logging labels changes logs but not one weight.
        flipped = self.configure('flipped')
        original = method.truth_lookup
        with patch.object(method, 'truth_lookup', side_effect=lambda m: {k: 1-v for k, v in original(m).items()}):
            method.train(flipped)
        for strategy in a.strategies:
            ours, theirs = [weights(r/method.run_name('dinov2_b14', strategy, 42)/'last.pt') for r in [a.runs_root, flipped.runs_root]]
            self.assertTrue(all(torch.equal(v, theirs[k]) for k, v in ours.items()))
        h1 = json.loads((a.runs_root/method.run_name('dinov2_b14', 'ts_confidence', 42)/'history.json').read_text())
        h2 = json.loads((flipped.runs_root/method.run_name('dinov2_b14', 'ts_confidence', 42)/'history.json').read_text())
        self.assertNotEqual([h['bbse']['pi_true_exposures'] for h in h1], [h['bbse']['pi_true_exposures'] for h in h2])
        # target_weight 0: same source objective/batches as pool-v2 CE+Rank (pooled selection).
        zero = self.configure('zero', ['ts_confidence']); zero.target_weight = 0.
        method.train(zero)
        ce = copy.copy(self.args); ce.strategies = ['ce_rank_pooled']
        ce.runs_root, ce.manifest_root = self.root/'ce', self.root/'ce_manifests'
        method.pool.train(ce)
        ours = weights(zero.runs_root/method.run_name('dinov2_b14', 'ts_confidence', 42)/'last.pt')
        golden = weights(ce.runs_root/method.pool.run_name('dinov2_b14', 'ce_rank_pooled', 42)/'last.pt')
        self.assertTrue(all(torch.equal(v, ours[k]) for k, v in golden.items()))

    def test_resume_and_evaluate_with_references(self):
        ref = copy.copy(self.args); ref.strategies = ['ce_rank_pooled', 'sda_new_balanced']
        ref.runs_root, ref.manifest_root, ref.output_root = self.root/'ref', self.root/'ref_manifests', self.root/'ref_results'
        method.pool.train(ref)
        method.pool.evaluate(ref)
        a = self.configure('uda')
        a.reference_roots, a.pool_reference = [ref.output_root], ref.manifest_root/'pool_v2/summary.json'
        method.train(a)
        resumed = self.configure('resumed'); resumed.strategies = ['ts_rank_quantile']
        resumed.pool_reference, resumed.reference_roots = a.pool_reference, a.reference_roots
        save = method.base.common.save
        def interrupt(path, state):
            save(path, state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(method.base.common, 'save', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'intentional'):
            method.train(resumed)
        method.train(resumed)
        golden = torch.load(a.runs_root/method.run_name('dinov2_b14', 'ts_rank_quantile', 42)/'last.pt', weights_only=True)
        restored = torch.load(resumed.runs_root/method.run_name('dinov2_b14', 'ts_rank_quantile', 42)/'last.pt', weights_only=True)
        self.assertEqual(golden['history'], restored['history'])
        for field in ['model_state_dict', 'teacher_state_dict']:
            self.assertTrue(all(torch.equal(v, restored[field][k]) for k, v in golden[field].items()))
        bad = copy.copy(a); bad.family_cap = 2; bad.manifest_root = self.root/'bad_manifests'
        with self.assertRaisesRegex(ValueError, 'differs'):
            method.inputs(bad)
        for strategy in a.strategies:  # force trained checkpoints so evaluation runs inference
            run = a.runs_root/method.run_name('dinov2_b14', strategy, 42)
            last = torch.load(run/'last.pt', weights_only=True)
            method.base.common.save(run/'best.pt', {k: last[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']}
                | dict(validation=last['history'][-1]['validation']))
        method.evaluate(a)
        mix = pd.read_csv(a.output_root/'mixture_summary.csv')
        self.assertEqual(set(mix.strategy), {'baseline', 'ts_confidence', 'ts_rank_quantile', 'ce_rank_pooled', 'sda_new_balanced'})
        reference = pd.read_csv(ref.output_root/'mixture_summary.csv').set_index('strategy')
        mine = mix.set_index('strategy')
        self.assertAlmostEqual(mine.loc['ce_rank_pooled', 'pooled_auc'], reference.loc['ce_rank_pooled', 'pooled_auc'], places=10)
        self.assertIn('gap_closure_vs_sda', mix)
        self.assertTrue({'seen_identity_pooled_auc'}.issubset(mix))
        boot = pd.read_csv(a.output_root/'bootstrap.csv')
        self.assertEqual(set(zip(boot.strategy, boot.reference)), {('ts_confidence', 'ce_rank_pooled'),
            ('ts_rank_quantile', 'ce_rank_pooled'), ('ts_rank_quantile', 'ts_confidence')})
        pseudo = pd.read_csv(a.output_root/'pseudo_labels.csv')
        self.assertEqual(set(pseudo.strategy), set(a.strategies))
        report = json.loads((a.output_root/'dinov2_b14_ts_rank_quantile_normal.json').read_text())
        self.assertGreater(report['selected_epoch'], 0)
        times = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        method.evaluate(a)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})

    def test_colab(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_rank_uda_pool_v2_seed42'
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
