"""CPU checks for the v2 diagonal trainer: feature consistency, strategy losses, training/skip/seeds and evaluation."""
import argparse
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_v2 as v2
import build_beauty_smoothing_v2_cache as builder
import evaluate_beauty_smoothing_v2 as evaluation
import diagnose_v2_compensation as compensation
import ffpp_c23_smoothing_v2_diagonal as diagonal
import test_beauty_smoothing_v2 as v2_fixture
import test_smoothing_rank_sda_pool_v2 as fixture


class ConsistencyTests(unittest.TestCase):
    def test_zero_when_identical_and_gradient_only_on_smoothed_half(self):
        f = torch.randn(6, 5, requires_grad=True)
        both = torch.cat([f, f])
        self.assertAlmostEqual(diagonal.feature_consistency(both, 6, torch.ones(6, dtype=torch.bool)).item(), 0., places=6)
        clean = torch.randn(4, 5, requires_grad=True)
        smooth = torch.randn(4, 5, requires_grad=True)
        loss = diagonal.feature_consistency(torch.cat([clean, smooth]), 4, torch.tensor([True, True, False, True]))
        loss.backward()
        self.assertTrue(clean.grad is None or torch.equal(clean.grad, torch.zeros_like(clean)))   # original = stop-gradient anchor
        self.assertTrue(torch.equal(smooth.grad[2], torch.zeros(5)))     # fallback frame excluded
        self.assertGreater(smooth.grad.abs().sum().item(), 0)


class DiagonalTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def train_args(self, **extra):
        a = self.args
        values = dict(source_root=a.source_root, baseline_runs_root=a.baseline_runs_root, runs_root=self.root/'diag_runs',
                      manifest_root=self.root/'diag_manifests', archive_dir=self.root/'drive', source_validation=None,
                      cache_root=self.root/'cache_train', train_filter='bilateral', models=['dinov2_b14'],
                      strategies=list(diagonal.STRATEGIES), seed=42, baseline_seed=42, epochs=2, patience=3, source_images=16,
                      val_images=16, batch_size=4, workers=0, encoder_lr=1e-4, head_lr=1e-4, weight_decay=1e-4,
                      ranking_weight=.25, fc_weights=[1.], device='cpu')
        return argparse.Namespace(**{**values, **extra})

    def build_caches(self):
        a = self.args
        v2_fixture.write_calibration(self.root/'calibration.json')
        for dataset, root in [('ffpp_trainval', a.source_root), ('cdf_test', a.target_root)]:
            builder.build(argparse.Namespace(dataset=dataset, data_root=root, calibration=self.root/'calibration.json',
                archive_dir=self.root/'drive', cache_root=self.root/'cache_build', parts=v2.SmoothingCache.PARTS,
                workers=1, parse_batch=4, device='cpu', membership=None))

    def test_train_skip_seed_and_evaluate(self):
        with patch.object(v2, 'SkinParser', v2_fixture.FakeParser), patch.object(v2, 'full', side_effect=v2_fixture.fake_full):
            self.build_caches()
            t = self.train_args()
            diagonal.train(t)
            labels = [label for label, _, _ in diagonal.variants(diagonal.STRATEGIES, [1.])]
            self.assertEqual(labels, ['aug_single', 'aug_pair', 'aug_pair_rank', 'aug_pair_rank_fc1'])
            runs = {s: t.runs_root/diagonal.run_name('dinov2_b14', 'bilateral', s, 42) for s in labels}
            for strategy, run in runs.items():
                self.assertTrue((run/'completed.json').is_file() and (run/'best.pt').is_file())
                history = json.loads((run/'history.json').read_text())
                fc = [h['feature_consistency'] for h in history]
                self.assertEqual(any(x > 0 for x in fc), strategy == 'aug_pair_rank_fc1')
            # Only the train-filter + mask tars were unpacked for training.
            self.assertEqual(sorted(p.name for p in (self.root/'cache_train/ffpp_trainval').iterdir()), ['bilateral', 'mask'])
            # Completed runs are skipped, not retrained.
            with patch.object(diagonal, 'PairedV2', side_effect=AssertionError('retrained')):
                diagonal.train(t)
            # fc_weight=0 reproduces aug_pair_rank exactly: the consistency term is the only difference.
            z = self.train_args(runs_root=self.root/'zero_runs', fc_weights=[0., 10.], strategies=['aug_pair_rank', 'aug_pair_rank_fc'])
            diagonal.train(z)
            w = [torch.load(z.runs_root/diagonal.run_name('dinov2_b14', 'bilateral', s, 42)/'last.pt', weights_only=True)['model_state_dict']
                 for s in ['aug_pair_rank', 'aug_pair_rank_fc0', 'aug_pair_rank_fc10']]
            self.assertFalse(all(torch.equal(w[0][k], w[2][k]) for k in w[0]))    # lambda 10 is a separate, different run
            self.assertTrue(all(torch.equal(w[0][k], w[1][k]) for k in w[0]))
            # Seed 43 fine-tunes the same seed-42 baseline.
            s43 = self.train_args(seed=43, strategies=['aug_pair'])
            diagonal.train(s43)
            cfg = json.loads((s43.runs_root/diagonal.run_name('dinov2_b14', 'bilateral', 'aug_pair', 43)/'config.json').read_text())
            base_cfg = json.loads((runs['aug_pair']/'config.json').read_text())
            self.assertEqual(cfg['initial_checkpoint_sha256'], base_cfg['initial_checkpoint_sha256'])
            self.assertEqual((cfg['seed'], cfg['baseline_seed']), (43, 42))

        e = argparse.Namespace(dataset='cdf_test', data_root=self.args.target_root, baseline_runs_root=self.args.baseline_runs_root,
            archive_dir=self.root/'drive', output_root=self.root/'eval', cache_root=self.root/'cache_eval', protocol=self.args.protocol,
            models=['dinov2_b14'], seed=42, baseline_seed=42, eval_batch_size=4, workers=0, device='cpu',
            runs_root=t.runs_root, train_filters=['bilateral'], strategies=list(diagonal.STRATEGIES), fc_weights=[1.], membership=None)
        table = evaluation.evaluate(e)
        self.assertEqual(len(table), (1+len(diagonal.STRATEGIES))*len(v2.FILTERS))
        trained = table[table.method.ne('baseline')]
        self.assertEqual(set(trained.strategy), {f'{s}@bilateral' for s in labels})
        self.assertEqual(set(trained[trained['filter'].eq('bilateral')].held_out), {False})
        self.assertEqual(set(trained[trained['filter'].ne('bilateral')].held_out), {True})
        self.assertFalse(table[table.method.eq('baseline')].held_out.any())

        # Strength curve from the saved predictions only (no model is built).
        with patch.object(evaluation.base, 'build_model', side_effect=AssertionError('inference')):
            c = compensation.curve(argparse.Namespace(eval_root=e.output_root))
        self.assertEqual(len(c), len(table))
        self.assertEqual({k for k in c if k.startswith(('real_', 'fake_'))},
                         {f'{s}_{l}' for s in ['real', 'fake'] for l in ['weak', 'medium', 'strong']})
        strong = table.set_index(['strategy', 'filter']).real_shift
        self.assertTrue(np.allclose(c.set_index(['strategy', 'filter']).real_strong.reindex(strong.index), strong))
        # Trace probe on FF++ val with the train-filter cache.
        with patch.object(v2, 'full', side_effect=v2_fixture.fake_full):
            pr = compensation.probe(argparse.Namespace(source_root=self.args.source_root, baseline_runs_root=self.args.baseline_runs_root,
                runs_root=t.runs_root, archive_dir=self.root/'drive', cache_root=self.root/'cache_probe', output_root=self.root/'probe',
                train_filter='bilateral', models=['dinov2_b14'], strategies=['aug_pair_rank', 'aug_pair_rank_fc'], fc_weights=[1.],
                seed=42, baseline_seed=42, images=16, folds=2, batch_size=4, workers=0, device='cpu'))
        self.assertEqual(list(pr.strategy), ['baseline', 'aug_pair_rank@bilateral', 'aug_pair_rank_fc1@bilateral'])
        self.assertTrue(pr.probe_auc.between(0, 1).all() and (pr.feature_shift >= 0).all())


if __name__ == '__main__':
    unittest.main()
