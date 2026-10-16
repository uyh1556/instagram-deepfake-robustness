"""Gradient/protocol/resume tests with tiny CPU models; no GPU result claims."""
import ast
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_ranking as ranking
import test_smoothing_preservation as fixture


class RankingTests(unittest.TestCase):
    def setUp(self):
        fixture.PreservationTests.setUp(self)
        a = self.args
        a.strategies = list(ranking.STRATEGIES)
        a.feature_batch_size = 8
        a.ranking_weight, a.ranking_gap = .5, 1.
        a.train_cache = self.root/'native_cache'
        a.evaluation_pixel_cache = self.root/'cdf_native_cache'
        a.epochs = 2

    def tearDown(self):
        fixture.PreservationTests.tearDown(self)

    def test_worst_view_gradient_and_matched_head(self):
        labels = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([[-1., 1.], [-2., .5], [.3, -.2], [.7, .1]], requires_grad=True)
        _, _, rank = ranking.losses(scores, labels, 2., .5, 1.)
        rank.backward()
        self.assertTrue(torch.all(scores.grad[:2, 1] > 0))  # descent lowers worst real
        self.assertTrue(torch.all(scores.grad[2:, 1] < 0))  # descent raises worst fake
        self.assertTrue(torch.equal(scores.grad[:, 0], torch.zeros(4)))
        total, ce, original = ranking.losses(scores, labels, 2., 0., 1.)
        self.assertTrue(torch.equal(total, ce))
        _, _, shifted = ranking.losses(scores+5, labels, 2., 0., 1.)
        self.assertTrue(torch.allclose(original, shifted))
        with self.assertRaises(ValueError):
            ranking.losses(scores, torch.zeros(4), 2., .5, 1.)
        head = ranking.CurrentClassifier(4, 8)
        old = ranking.response.ResponseClassifier(4, 8, False)
        with torch.no_grad():
            head.net[-1].weight.fill_(.04)
        old.load_state_dict(head.state_dict(), strict=True)
        f, m = torch.randn(7, 4), torch.randn(7)
        self.assertTrue(torch.equal(head(f, m), old(torch.stack([f, f+999], 1), torch.stack([m, m-999], 1))))
        batches = ranking.balanced_batches(labels, 4, 42)
        self.assertEqual(sorted(sum(batches, [])), list(range(4)))
        self.assertTrue(all(labels[b].sum() == len(b)//2 for b in batches))

    def test_cache_pixels_fallback_and_raw_auc(self):
        pool, _, data = ranking.shared.source_inputs(self.args)
        ranking.pixel_helpers.prepare_pixel_cache(self.args, self.cfg, data)
        plan = ranking.response.training_plan(pool.head(6), 42, self.cfg)
        fast = ranking.ObservedViews(plan, self.args.source_root, fixture.build()[1], self.cfg, self.args._pixel_cache)
        slow = ranking.ObservedViews(plan, self.args.source_root, fixture.build()[1], self.cfg)
        for i in range(len(plan)):
            self.assertTrue(torch.equal(fast[i][0], slow[i][0]))
        with patch.object(ranking.pixel_helpers.pixel_cache, 'bilateral', side_effect=AssertionError('cache missed')):
            for i in range(len(plan)):
                fast[i]
        # Probabilities collapse to 1 in FP32, while raw-score ranking remains perfect.
        frame = pd.DataFrame({'class_index': ['0', '0', '1', '1']})
        result = ranking.metrics(frame, np.array([100., 101., 102., 103.]))
        self.assertEqual(result['auc'], 1.)
        self.assertEqual(result['auc_probability'], .5)
        self.assertEqual(result['real_fpr'], 1.)
        self.assertEqual(result['fake_fnr'], 0.)

    def test_train_all_models_evaluate_reuse_and_preserve(self):
        a = self.args
        target = a.target_root
        a.target_root = self.root/'NO_TARGET_DURING_TRAINING'
        ranking.train(a)
        a.target_root = target
        ranking.train(a)
        for name in a.models:
            plans, batches = [], []
            for strategy in a.strategies:
                run = a.runs_root/ranking.run_name(name, strategy, a.seed)
                last = torch.load(run/'last.pt', weights_only=True)
                selected = torch.load(run/'best.pt', weights_only=True)
                epoch0 = json.loads((run/'epoch0_validation.json').read_text())
                self.assertLessEqual(selected['best'], epoch0['mean_ber'])
                self.assertNotIn('model_state_dict', last)
                self.assertFalse(last['config']['target_training'])
                self.assertTrue(all(np.isfinite(r['loss']) for r in last['history']))
                plans.append([r['source_plan_sha256'] for r in last['history']])
                batches.append([r['batch_order_sha256'] for r in last['history']])
            self.assertEqual(plans[0], plans[1])
            self.assertEqual(batches[0], batches[1])
            self.assertEqual(ranking.base.common.digest(a.baseline_runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed42/best.pt'), self.initial_hashes[name])
        ranking.evaluate(a)
        before = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*.json')}
        with patch.object(ranking, 'features_for', side_effect=AssertionError('Inference on completed evaluation')):
            ranking.evaluate(a)
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*.json')})
        table = pd.read_csv(a.output_root/'comparison.csv')
        self.assertTrue(np.allclose(table.balanced_error_rate, .5*(table.real_fpr+table.fake_fnr)))
        self.assertTrue(set(['baseline', *a.strategies]).issubset(table.strategy))
        cross = pd.read_csv(a.output_root/'cross_condition_auc.csv')
        mixture = pd.read_csv(a.output_root/'mixture_summary.csv')
        for row in mixture.itertuples():
            part = cross[(cross.model == row.model) & (cross.strategy == row.strategy)]
            self.assertEqual(len(part), 16)
            self.assertAlmostEqual(part.auc.mean(), row.pooled_auc)
        # Old results/checkpoints and data are never rewritten.
        with self.assertRaises(ValueError):
            ranking.train(type(a)(**{**vars(a), 'ranking_gap': 3.}))

    def test_exact_resume_between_strategies(self):
        a = self.args
        a.models = ['clip_b16']
        ranking.train(a)
        reference = {s: torch.load(a.runs_root/ranking.run_name('clip_b16', s, 42)/'last.pt', weights_only=True) for s in a.strategies}
        a.runs_root = self.root/'resumed_runs'
        original_save = ranking.base.common.save
        interrupted = False

        def stop(path, value):
            nonlocal interrupted
            original_save(path, value)
            if Path(path).name == 'last.pt' and not interrupted:
                interrupted = True
                raise RuntimeError('simulated interruption between strategy updates')

        with patch.object(ranking.base.common, 'save', side_effect=stop):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                ranking.train(a)
        ranking.train(a)
        for strategy in a.strategies:
            actual = torch.load(a.runs_root/ranking.run_name('clip_b16', strategy, 42)/'last.pt', weights_only=True)
            self.assertEqual(actual['history'], reference[strategy]['history'])
            for key in actual['head_state_dict']:
                self.assertTrue(torch.equal(actual['head_state_dict'][key], reference[strategy]['head_state_dict'][key]), key)

    def test_notebook_cells(self):
        from IPython.core.interactiveshell import InteractiveShell
        path = ranking.ROOT/'notebooks/train_test_ffpp_c23_smoothing_ranking_seed42.ipynb'
        notebook = json.loads(path.read_text())
        code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 7)
        shell = InteractiveShell.instance()
        for cell in code:
            source = ''.join(cell['source'])
            self.assertNotEqual(source.lstrip().splitlines()[0], 'python')
            ast.parse(shell.input_transformer_manager.transform_cell(source))
        self.assertIn("MODELS=['clip_b16']", ''.join(code[1]['source']))


if __name__ == '__main__':
    unittest.main()
