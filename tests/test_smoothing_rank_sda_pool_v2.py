"""Small CPU checks for pool v2, balanced SDA, pooled-AUC selection and evaluation."""
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
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_rank_sda_pool_v2 as method
import test_smoothing_mean_teacher as fixture

TEST_VIDEOS = {'Celeb-real/id0_0000': '0', 'YouTube-real/00001': '0',
               'Celeb-synthesis/id1_id0_0003': '1', 'Celeb-synthesis/id2_id3_0004': '1'}
# A: one CR + 4 fakes of scene id4_0000 (cap 3); B/C: 1 CR + 1 fake; D: fake only.
KEPT = ['Celeb-real/id4_0000', 'Celeb-real/id5_0001', 'Celeb-real/id6_0002',
        'Celeb-synthesis/id4_id5_0000', 'Celeb-synthesis/id4_id6_0000', 'Celeb-synthesis/id4_id7_0000',
        'Celeb-synthesis/id4_id8_0000', 'Celeb-synthesis/id5_id6_0001', 'Celeb-synthesis/id6_id4_0002',
        'Celeb-synthesis/id7_id9_0005', 'YouTube-real/00010', 'YouTube-real/00011',
        'YouTube-real/00012', 'YouTube-real/00013']
# Excluded by one-hop keys; id8_id2_0004 only through its SECOND id (kept by the old rule).
EXCLUDED = ['Celeb-real/id0_0003', 'Celeb-real/id1_0003', 'Celeb-synthesis/id1_id5_0003',
            'Celeb-synthesis/id9_id0_0003', 'Celeb-synthesis/id8_id2_0004']


def write_videos(root, videos, split, rng):
    rows = []
    for video in videos:
        kind = video.split('/')[0]
        for frame in range(2):
            path = f'frames/{video}/{frame:06d}.png'
            (root/path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(root/path)
            rows.append(dict(crop_path=path, video_id=video, method=kind, split=split,
                             class_index=method.KIND_LABEL[kind], label='fake' if kind == 'Celeb-synthesis' else 'real',
                             roi_status='detected' if frame else 'fallback_letterbox'))
    return pd.DataFrame(rows)


class PoolV2Tests(unittest.TestCase):
    def setUp(self):
        fixture.MeanTeacherTests.setUp(self)
        a = self.args
        rng = np.random.default_rng(7)
        write_videos(a.target_root, TEST_VIDEOS, 'test', rng).to_csv(self.test, index=False)
        catalog = write_videos(a.target_root, KEPT+EXCLUDED, 'non_test_pool', rng)
        for kind in ['real', 'fake']:
            catalog[catalog.label.eq(kind)].to_csv(a.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv', index=False)
        a.models, a.strategies = ['dinov2_b14'], ['ce_rank_pooled', 'sda_new_balanced', 'sda_old_balanced']
        a.hard_manifest_root, a.family_cap, a.adapt_val_fraction = None, 3, .2
        a.bootstrap, a.eval_batch_size, a.ranking_weight, a.delta_cap_ratio = 10, 4, .25, .05

    tearDown = fixture.MeanTeacherTests.tearDown

    def test_pool_rules_and_freeze(self):
        a = self.args
        _, _, old, _, _, _ = method.adaptation.dataset_inputs(a)
        frame, summary = method.freeze_pool(a, old)
        videos = set(frame.video_id)
        self.assertFalse(videos & set(EXCLUDED))
        self.assertIn('Celeb-synthesis/id8_id2_0004', set(old.video_id))  # old transitive rule missed it
        self.assertEqual(sum(v.startswith('Celeb-synthesis/id4_') for v in videos), 3)
        kinds = frame.drop_duplicates('video_id').kind.value_counts().to_dict()
        self.assertEqual(kinds, {'Celeb-synthesis': 6, 'Celeb-real': 3, 'YouTube-real': 3})
        self.assertFalse(frame.groupby('scene_group').partition.nunique().gt(1).any())
        val = frame[frame.partition.eq('adapt_val')]
        self.assertEqual(set(val.class_index), {'0', '1'})
        self.assertTrue(frame.class_index.eq(frame.kind.map(method.KIND_LABEL)).all())
        unlabelled = pd.read_csv(a.manifest_root/'pool_v2/adapt_train_unlabelled.csv')
        self.assertEqual(list(unlabelled), ['crop_path', 'video_id', 'roi_status'])
        self.assertEqual(set(unlabelled.video_id), set(frame[frame.partition.eq('adapt_train')].video_id))
        self.assertEqual(summary['eligible_videos']['YouTube-real'], 4)
        method.freeze_pool(a, old)  # deterministic re-run accepted
        changed = copy.copy(a); changed.family_cap = 2
        with self.assertRaisesRegex(ValueError, 'differs'):
            method.freeze_pool(changed, old)

    def test_balanced_plan_and_pooled_summary(self):
        a = self.args
        _, _, old, _, _, _ = method.adaptation.dataset_inputs(a)
        frame, _ = method.build_pool(a, old)
        train = frame[frame.partition.eq('adapt_train')]
        plan = method.balanced_plan(train, 32, 4, 42)
        self.assertEqual(plan, method.balanced_plan(train, 32, 4, 42))
        self.assertTrue(all('class_index' not in row for row in plan))
        labels = train.set_index('crop_path').class_index.to_dict()
        for start in range(0, 32, 4):
            self.assertEqual(sorted(labels[r['crop_path']] for r in plan[start:start+4]), ['0', '0', '1', '1'])
        with self.assertRaises(ValueError):
            method.balanced_plan(train, 30, 4, 42)
        rng = np.random.default_rng(3)
        y = np.r_[np.zeros(7), np.ones(9)].astype(int)
        banks = [rng.normal(size=16)+k*y for k in range(4)]
        records = {c: {'auc': roc_auc_score(y, b), 'balanced_error_rate': .1} for c, b in zip(method.CONDITIONS, banks)}
        s = method.summarize(y, banks, records)
        self.assertAlmostEqual(s['pooled_auc'], roc_auc_score(np.tile(y, 4), np.concatenate(banks)), places=12)
        self.assertAlmostEqual(s['cross_auc'][3][0], roc_auc_score(np.r_[np.zeros(7), np.ones(9)],
                               np.r_[banks[3][y == 0], banks[0][y == 1]]), places=12)
        self.assertEqual(s['min_cross_auc'], min(map(min, s['cross_auc'])))

    def test_train_matches_ce_rank_resumes_and_evaluates(self):
        a = self.args
        old = copy.copy(a); old.strategies = ['augmentation']
        method.engine.train(old)  # previous hard manifests for the CE+Rank v1 reference
        ce = copy.copy(old)
        ce.hard_manifest_root, ce.strategies = old.manifest_root, ['ce_rank']
        ce.runs_root, ce.manifest_root = self.root/'ce_runs', self.root/'ce_manifests'
        method.rank.train(ce)
        a.hard_manifest_root = old.manifest_root
        a.runs_root, a.manifest_root, a.output_root = self.root/'v2_runs', self.root/'v2_manifests', self.root/'v2_results'
        protected = {str(p): method.base.common.digest(p) for root in [ce.runs_root, a.baseline_runs_root]
                     for p in root.rglob('*') if p.is_file()}
        method.prepare(a)
        method.train(a)
        # Same seeds/batches/loss: only the checkpoint rule differs from CE+Rank v1.
        mine = torch.load(a.runs_root/method.run_name('dinov2_b14', 'ce_rank_pooled', 42)/'last.pt', weights_only=True)
        golden = torch.load(next(ce.runs_root.rglob('last.pt')), weights_only=True)
        self.assertTrue(all(torch.equal(v, mine['model_state_dict'][k]) for k, v in golden['model_state_dict'].items()))
        self.assertEqual([h['source_batches_sha256'] for h in mine['history']], [h['source_batches_sha256'] for h in golden['history']])
        for strategy in a.strategies:
            run = a.runs_root/method.run_name('dinov2_b14', strategy, 42)
            state = torch.load(run/'last.pt', weights_only=True)
            done = json.loads((run/'completed.json').read_text())
            values = [json.loads((run/'epoch0_validation.json').read_text())] + [h['validation'] for h in state['history']]
            self.assertEqual(done['best_ffpp']['pooled_auc'], max(v['ffpp']['pooled_auc'] for v in values))
            self.assertEqual(done['best_adapt_val']['pooled_auc'], max(v['adapt_val']['pooled_auc'] for v in values))
            exposures = [h['target_class_exposures'] for h in state['history']]
            if strategy == 'ce_rank_pooled':
                self.assertTrue(all(e == {'real': 0, 'fake': 0} for e in exposures))
                self.assertIsNone(state['config']['target_label_sha256'])
            else:
                self.assertTrue(all(e == {'real': 8, 'fake': 8} for e in exposures))
        old_cfg = json.loads((a.runs_root/method.run_name('dinov2_b14', 'sda_old_balanced', 42)/'config.json').read_text())
        self.assertIsNotNone(old_cfg['hard_reference_manifest'])
        saved = {str(p): method.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        method.train(a)
        self.assertEqual(saved, {str(p): method.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        # Interrupted run resumes to identical weights/history.
        resumed = copy.copy(a); resumed.runs_root = self.root/'resumed'; resumed.strategies = ['sda_new_balanced']
        save = method.base.common.save
        def interrupt(path, state):
            save(path, state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(method.base.common, 'save', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'intentional'):
            method.train(resumed)
        method.train(resumed)
        restored = torch.load(resumed.runs_root/method.run_name('dinov2_b14', 'sda_new_balanced', 42)/'last.pt', weights_only=True)
        golden = torch.load(a.runs_root/method.run_name('dinov2_b14', 'sda_new_balanced', 42)/'last.pt', weights_only=True)
        self.assertEqual(golden['history'], restored['history'])
        self.assertTrue(all(torch.equal(v, restored['model_state_dict'][k]) for k, v in golden['model_state_dict'].items()))
        # Force trained (non-epoch0) checkpoints so evaluation runs real inference.
        for strategy in a.strategies:
            run = a.runs_root/method.run_name('dinov2_b14', strategy, 42)
            last = torch.load(run/'last.pt', weights_only=True)
            for key, file in method.CHECKPOINTS.items():
                method.base.common.save(run/file, {k: last[k] for k in ['identity', 'config', 'model_state_dict', 'epoch']}
                    | dict(selection=key, best=last['best'], validation=last['history'][-1]['validation']))
        method.evaluate(a)
        mixture = pd.read_csv(a.output_root/'mixture_summary.csv')
        overlap = old_cfg['adapt_val_overlaps_training']
        expected = {'baseline', 'ce_rank_pooled', 'ce_rank_pooled@adapt_val', 'sda_new_balanced',
                    'sda_new_balanced@adapt_val', 'sda_old_balanced'} | (set() if overlap else {'sda_old_balanced@adapt_val'})
        self.assertEqual(set(mixture.strategy), expected)
        self.assertEqual(len(pd.read_csv(a.output_root/'comparison.csv')), 4*len(expected))
        self.assertEqual(len(pd.read_csv(a.output_root/'cross_condition_auc.csv')), 16*len(expected))
        boot = pd.read_csv(a.output_root/'bootstrap_vs_ce_rank_pooled.csv')
        self.assertEqual(len(boot), 2*(len(expected)-1))
        self.assertTrue((boot.ci_low <= boot.ci_high).all())
        groups = pd.read_csv(a.output_root/'subgroups.csv')
        self.assertTrue({'fpr_Celeb-real', 'fpr_YouTube-real', 'auc_fake_vs_Celeb-real'}.issubset(groups))
        report = json.loads((a.output_root/'dinov2_b14_sda_new_balanced_normal.json').read_text())
        self.assertGreater(report['selected_epoch'], 0)
        self.assertNotIn('reused_predictions', report)
        times = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        method.evaluate(a)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        for path, digest in protected.items():
            self.assertEqual(digest, method.base.common.digest(Path(path)))

    def test_colab(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_rank_sda_pool_v2_seed42'
        text = (method.ROOT/'docs/colab'/f'{stem}.md').read_text()
        notebook = json.loads((method.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        cells = re.findall(r'```python\n(.*?)```', text, re.S)
        self.assertEqual(cells, [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'])
        self.assertEqual(len(cells), 7)
        for cell in cells:
            ast.parse(TransformerManager().transform_cell(cell))


if __name__ == '__main__':
    unittest.main()
