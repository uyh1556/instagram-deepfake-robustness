"""CPU checks for the source-only ablation: aug_pair_rank == pool v2 CE+Rank, other losses, resume, evaluation."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_source_ablation as ablation
import test_smoothing_rank_sda_pool_v2 as fixture


def weights(path):
    return torch.load(path, weights_only=True)['model_state_dict']


def same(a, b):
    return all(torch.equal(v, b[k]) for k, v in a.items())


class SourceAblationTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def configure(self, runs, strategies):
        a = copy.copy(self.args)
        a.strategies, a.epochs, a.bootstrap = strategies, 2, 20
        a.runs_root, a.manifest_root, a.output_root = self.root/runs, self.root/f'{runs}_manifests', self.root/f'{runs}_results'
        return a

    def test_single_view_mask_and_pair_loss(self):
        model = fixture.fixture.tiny.Tiny() if hasattr(fixture.fixture, 'tiny') else None
        clean, smooth = torch.zeros(400, 3, 12, 12), torch.ones(400, 3, 12, 12)
        captured = {}

        def outputs(name, model, x):
            captured['x'] = x
            return torch.zeros(len(x), 2, requires_grad=True), None
        with patch.object(ablation.adaptation, 'outputs', side_effect=outputs):
            labels = torch.tensor([0, 1]*200)
            ablation.source_loss('aug_single', 'x', model, clean, smooth, labels, .25, 7)
            kept = (captured['x'][:, 0, 0, 0] == 0).float().mean().item()
            self.assertTrue(.18 < kept < .32)
            first = captured['x'].clone()
            ablation.source_loss('aug_single', 'x', model, clean, smooth, labels, .25, 7)
            self.assertTrue(torch.equal(first, captured['x']))
            ablation.source_loss('clean_only', 'x', model, clean, smooth, labels, .25, 7)
            self.assertTrue(torch.equal(captured['x'], clean))
            _, _, rank = ablation.source_loss('aug_pair', 'x', model, clean, smooth, labels, .25, 7)
            loss, ce, _ = ablation.source_loss('aug_pair', 'x', model, clean, smooth, labels, .25, 7)
            self.assertEqual(captured['x'].shape[0], 800)
            self.assertTrue(torch.equal(loss, ce))
            loss, ce, rank = ablation.source_loss('aug_pair_rank', 'x', model, clean, smooth, labels, .25, 7)
            self.assertTrue(torch.allclose(loss, ce+.25*rank))

    def test_ce_rank_matches_pool_v2_and_others_differ(self):
        ref = self.configure('pool', ['ce_rank_pooled'])
        ref.strategies = ['ce_rank_pooled']
        ablation.pool.prepare(ref)
        ablation.pool.train(ref)
        a = self.configure('ablation', list(ablation.STRATEGIES))
        ablation.train(a)
        golden = weights(ref.runs_root/ablation.pool.run_name('dinov2_b14', 'ce_rank_pooled', 42)/'last.pt')
        runs = {s: a.runs_root/ablation.run_name('dinov2_b14', s, 42) for s in ablation.STRATEGIES}
        self.assertTrue(same(golden, weights(runs['aug_pair_rank']/'last.pt')))
        for strategy in ['clean_only', 'aug_single', 'aug_pair']:
            self.assertFalse(same(golden, weights(runs[strategy]/'last.pt')), strategy)
        batches = {s: [h['source_batches_sha256'] for h in json.loads((r/'history.json').read_text())] for s, r in runs.items()}
        self.assertEqual(len({tuple(v) for v in batches.values()}), 1)   # identical samples/batches across conditions

    def test_resume_and_evaluate(self):
        a = self.configure('ablation', ['aug_single', 'aug_pair_rank'])
        ablation.train(a)
        b = self.configure('resumed', ['aug_single'])
        save = ablation.base.common.save

        def interrupt(path, value):
            save(path, value)
            if path.name == 'last.pt' and value['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(ablation.base.common, 'save', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'intentional'):
            ablation.train(b)
        ablation.train(b)
        self.assertTrue(same(weights(a.runs_root/ablation.run_name('dinov2_b14', 'aug_single', 42)/'last.pt'),
                             weights(b.runs_root/ablation.run_name('dinov2_b14', 'aug_single', 42)/'last.pt')))
        ablation.evaluate(a)
        table = pd.read_csv(a.output_root/'summary.csv')
        self.assertEqual(list(table.strategy), ['baseline', 'aug_single', 'aug_pair_rank'])
        self.assertTrue((table.fp_gap == table.normal_auc-table.fp_cross_auc).all())
        boot = pd.read_csv(a.output_root/'bootstrap_gaps_vs_aug_pair_rank.csv')
        self.assertEqual(set(boot.strategy), {'baseline', 'aug_single'})
        self.assertEqual(set(boot.metric), {'fp_gap', 'fn_gap'})
        self.assertTrue((table.fn_gap == table.normal_auc-table.fn_cross_auc).all())
        times = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        # An evaluation-only edit to this file changes its digest; completed runs must still be skipped.
        own = ablation.Path(ablation.__file__).name
        with patch.object(ablation, 'code_hashes', side_effect=lambda: {**ablation.pool.code_hashes(), own: 'edited'}):
            ablation.train(a)
        ablation.evaluate(a)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})


if __name__ == '__main__':
    unittest.main()
