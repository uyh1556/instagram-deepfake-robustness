"""Small CPU regression checks for the saved-prediction diagnostic."""
import argparse
import ast
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import analyze_smoothing_error_tradeoff as diagnostic


class TradeoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.y = np.array([0]*4+[1]*4)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, scores):
        scores = np.asarray(scores, dtype=float)
        frame = pd.DataFrame({'class_index':self.y.astype(str), 'raw_margin':scores})
        return {'frame':frame, 'auc':roc_auc_score(self.y,scores),
                'fixed':diagnostic.counts(self.y,scores,0)}

    def write_record(self, folder, strategy, condition, scores, reverse=False):
        folder.mkdir(parents=True, exist_ok=True)
        cfg = {'conditions':dict(zip(diagnostic.CONDITIONS,[0,.25,.5,.75]))}
        protocol = {'manifest_sha256':'fixture-manifest','config':cfg,
                    'smoothing_code_sha256':'fixture-code','opencv':'fixture'}
        (folder/'evaluation_protocol.json').write_text(json.dumps(protocol))
        scores = np.asarray(scores, dtype=float)
        frame = pd.DataFrame({'crop_path':[f'images/v{i}/000.png' for i in range(8)],
                              'video_id':[f'v{i:03}' for i in range(8)],
                              'class_index':self.y,'roi_status':['detected']*8,
                              'raw_margin':scores,'probability_fake':1/(1+np.exp(-scores))})
        prefix = folder/f'clip_b16_{strategy}_{condition}'
        frame.iloc[::-1 if reverse else 1].to_csv(prefix.with_suffix('.csv'),index=False)
        fixed = diagnostic.counts(self.y,scores,0)
        report = {'unit':'frame','precision':'FP32','threshold':.5,'images':8,
                  'manifest_sha256':'fixture-manifest','checkpoint_sha256':f'checkpoint-{strategy}',
                  'real_fpr':fixed['fpr'],'fake_fnr':fixed['fnr'],
                  'balanced_error_rate':fixed['ber'],'auc':roc_auc_score(self.y,scores)}
        prefix.with_suffix('.json').write_text(json.dumps(report))
        return prefix

    def args(self):
        return argparse.Namespace(evaluation_roots=[self.root/'old',self.root/'new',self.root/'absent'],
            output_root=self.root/'output',models=['clip_b16'],strategies=list(diagnostic.STRATEGIES),
            references=['baseline','augmentation'],expected_images=8,plots=False)

    def prepare(self):
        scores = [-1,.6,.2,-.4,.8,-.6,.5,.1]
        for strategy in ['baseline','augmentation','mean_teacher']:
            for condition in diagnostic.CONDITIONS:
                self.write_record(self.root/'old/group',strategy,condition,scores)
        for condition in diagnostic.CONDITIONS:
            self.write_record(self.root/'new/group','baseline',condition,scores,reverse=True)
            self.write_record(self.root/'new/group','clean_to_mixed',condition,
                              [-.8,-.3,-.2,-.1,.9,.8,.7,.6],reverse=True)
        # Partial optional runs must not be silently treated as four-condition averages.
        self.write_record(self.root/'new/group','normal_only','normal',scores)

    def test_ties_offsets_and_all_deterministic_thresholds(self):
        ref = self.record([-1,.6,.2,-.4,.8,-.6,.5,.1])
        for values in [[0,0,1,1,0,0,1,1],[-10]*8,[2,-3,4,-5,6,-7,8,-9]]:
            candidate = self.record(values)
            result, points = diagnostic.compare_condition(candidate,ref)
            all_counts = [diagnostic.counts(self.y,np.asarray(values),t) for t in
                          [np.inf,*sorted(set(values),reverse=True)]]
            self.assertEqual(set(zip(points.fp,points.fn)),set((r['fp'],r['fn']) for r in all_counts))
            feasible = [r for r in all_counts if r['fn'] <= ref['fixed']['fn']]
            self.assertAlmostEqual(result['oracle_at_reference_fnr_fpr'],min(r['fpr'] for r in feasible))
            feasible = [r for r in all_counts if r['fp'] <= ref['fixed']['fp']]
            self.assertAlmostEqual(result['oracle_at_reference_fpr_fnr'],min(r['fnr'] for r in feasible))
            self.assertEqual(result['oracle_strictly_lower_both'],any(
                r['fp'] < ref['fixed']['fp'] and r['fn'] < ref['fixed']['fn'] for r in all_counts))
        shifted = self.record(ref['frame'].raw_margin.to_numpy()-10)
        out,_ = diagnostic.compare_condition(shifted,ref)
        self.assertAlmostEqual(out['delta_auc'],0)
        self.assertAlmostEqual(out['oracle_fpr_reduction_pp_at_reference_fnr'],0)
        self.assertFalse(out['oracle_strictly_lower_both'])

    def test_conditionwise_success_does_not_imply_a_common_threshold(self):
        y = np.array([0]*4+[1]*4)
        ref = self.record([-1,.6,.2,-.4,.8,-.6,.5,.1])
        # Each condition has perfect ranking, but the two score scales cannot share a cutoff.
        groups = [self.record([-4,-3,-2,-1,1,2,3,4]),
                  self.record([96,97,98,99,101,102,103,104])]*2
        self.assertTrue(all(diagnostic.compare_condition(c,ref)[0]['oracle_strictly_lower_both'] for c in groups))
        result = diagnostic.common_threshold(groups,[ref]*4)
        self.assertFalse(result['oracle_one_threshold_nonworse_every_condition'])
        for t in np.r_[np.inf,np.unique(np.concatenate([r['frame'].raw_margin for r in groups]))]:
            rows = [diagnostic.counts(y,r['frame'].raw_margin.to_numpy(),t) for r in groups]
            self.assertFalse(all(r['fp'] <= ref['fixed']['fp'] and r['fn'] <= ref['fixed']['fn'] for r in rows))

    def test_end_to_end_reordered_duplicates_partial_runs_and_preservation(self):
        self.prepare()
        protected = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        args = self.args()
        args.plots = True
        diagnostic.analyze(args)
        self.assertTrue((args.output_root/'clip_b16_error_curves.png').is_file())
        fixed = pd.read_csv(args.output_root/'fixed_threshold.csv')
        points = pd.read_csv(args.output_root/'operating_points.csv')
        common = pd.read_csv(args.output_root/'one_threshold_all_conditions.csv')
        self.assertEqual(len(fixed),16)
        self.assertEqual(len(points),32)
        selected = points[points.strategy.eq('clean_to_mixed')]
        self.assertTrue(selected.fixed_both_strictly_lower.all())
        self.assertTrue(selected.auc_maintained.all())
        self.assertTrue(common[common.strategy.eq('clean_to_mixed')].oracle_one_threshold_strictly_lower_both_every_condition.all())
        audit = json.loads((args.output_root/'analysis_protocol.json').read_text())
        self.assertTrue(any(r.get('strategy') == 'normal_only' for r in audit['skipped']))
        for p, before in protected.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),before)
        with self.assertRaises(FileExistsError):
            diagnostic.analyze(args)

    def test_conflicts_missing_required_and_changed_metadata_fail(self):
        self.prepare()
        args = self.args()
        csv = self.root/'new/group/clip_b16_baseline_normal.csv'
        frame = pd.read_csv(csv)
        frame.raw_margin += .001  # same fixed decisions and AUC, different raw predictions
        frame.to_csv(csv,index=False)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            diagnostic.discover(args)
        args.evaluation_roots = [self.root/'old']
        csv = self.root/'old/group/clip_b16_mean_teacher_normal.csv'
        frame = pd.read_csv(csv); frame.loc[0,'video_id'] = 'wrong'; frame.to_csv(csv,index=False)
        with self.assertRaisesRegex(ValueError,'membership differs'):
            diagnostic.discover(args)
        csv.unlink()
        with self.assertRaisesRegex(FileNotFoundError,'missing'):
            diagnostic.discover(args)

    def test_notebook_matches_md_and_code_cells_have_no_python_fence(self):
        md = (ROOT/'docs/colab/analyze_smoothing_error_tradeoff_seed42.md').read_text()
        blocks = re.findall(r'```python\n(.*?)```',md,re.S)
        nb = json.loads((ROOT/'notebooks/analyze_smoothing_error_tradeoff_seed42.ipynb').read_text())
        actual = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(blocks,actual)
        self.assertEqual(len(actual),4)
        from IPython.core.interactiveshell import InteractiveShell
        for source in actual:
            ast.parse(InteractiveShell.instance().input_transformer_manager.transform_cell(source))
            self.assertFalse(source.startswith('python\n'))


if __name__ == '__main__':
    unittest.main()
