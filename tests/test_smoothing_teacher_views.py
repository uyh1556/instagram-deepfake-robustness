"""Small CPU checks of view isolation, matched budgets, restart and saved references."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_mean_teacher as mt
import ffpp_c23_smoothing_teacher_views as views
import test_smoothing_mean_teacher as fixture
import test_smoothing_preservation as pixels


class TeacherViewTests(unittest.TestCase):
    setUp = fixture.MeanTeacherTests.setUp
    tearDown = fixture.MeanTeacherTests.tearDown

    def new_args(self, old):
        a = copy.copy(old)
        a.runs_root, a.manifest_root, a.output_root = self.root/'view_runs', self.root/'view_manifests', self.root/'view_results'
        a.hard_manifest_root, a.hard_evaluations = old.manifest_root, old.output_root
        a.strategies = list(views.STRATEGIES)
        return a

    def test_exact_views_and_label_free_matched_exposure_plans(self):
        a = self.args
        _, _, target, cfg, _, _ = mt.adaptation.dataset_inputs(a)
        original_plan = mt.target_plan(target, 200, 42)
        a._target_cache = mt.common.pixel_cache.BilateralCache(a.target_train_cache/'pixels', cfg['bilateral'], {'test':True})
        original_hooks = (mt.PROTOCOL, mt.TargetViews, mt.target_plan, mt.configuration, mt.pseudo_loss)
        for strategy in views.STRATEGIES:
            with views.view_protocol(strategy):
                one = mt.target_plan(target.assign(class_index=0, method='real'), 200, 42)
                two = mt.target_plan(target.assign(class_index=1, method='fake'), 200, 42)
                self.assertEqual(one, two)
                self.assertTrue(all('class_index' not in r and 'method' not in r for r in one))
                for new, old in zip(one, original_plan):
                    self.assertEqual({k:new[k] for k in ['crop_path','video_id','roi_status','flip']},
                                     {k:old[k] for k in ['crop_path','video_id','roi_status','flip']})
                    self.assertEqual(new['scheduled_condition'], old['condition'])
                    self.assertEqual(new['teacher_condition'], 'normal')
                    self.assertEqual(new['condition'], 'normal' if strategy == 'normal_only' else old['condition'])
                for status in ['detected','fallback_letterbox']:
                    row = next(r for r in one if r['roi_status'] == status and r['scheduled_condition'] == 'smooth_strong')
                    for flip in [False, True]:
                        row = {**row, 'flip':flip}
                        teacher, student = mt.TargetViews([row], a, pixels.build()[1], cfg, False)[0]
                        with Image.open(a.target_root/row['crop_path']) as opened:
                            raw = opened.convert('RGB')
                        changed = (mt.adaptation.smooth(raw, cfg['conditions']['smooth_strong'], cfg['bilateral'])
                                   if strategy == 'clean_to_mixed' and status == 'detected' else raw)
                        teacher_expected = raw.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else raw
                        student_expected = changed if flip else changed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        self.assertTrue(torch.equal(teacher, pixels.build()[1](teacher_expected)))
                        self.assertTrue(torch.equal(student, pixels.build()[1](student_expected)))
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with views.view_protocol('normal_only'):
                raise RuntimeError('intentional')
        self.assertEqual(original_hooks, (mt.PROTOCOL, mt.TargetViews, mt.target_plan, mt.configuration, mt.pseudo_loss))

    def test_training_and_evaluation_reuse_without_touching_previous_runs(self):
        old = self.args
        mt.train(old)
        mt.evaluate(old)
        protected = {str(p):mt.base.common.digest(p) for p in
            [*old.runs_root.rglob('*.pt'), *old.output_root.glob('*.csv'), *old.manifest_root.glob('*.csv')]}
        a = self.new_args(old)
        views.train(a)
        legacy = json.loads((old.runs_root/mt.run_name('clip_b16','mean_teacher',42)/'history.json').read_text())
        for strategy in a.strategies:
            run = a.runs_root/f'clip_b16_{strategy}_{views.PROTOCOL}_seed42'
            state = torch.load(run/'last.pt', weights_only=True)
            self.assertEqual(state['config']['teacher_condition'], 'normal')
            self.assertEqual(state['config']['target_weight'], .5)
            self.assertEqual(state['config']['pseudo_label_kind'], 'hard')
            self.assertEqual([h['source_paths_sha256'] for h in state['history']], [h['source_paths_sha256'] for h in legacy])
            for h in state['history']:
                self.assertEqual(h['target_exposures'], 16)
                if strategy == 'normal_only':
                    self.assertEqual(h['target_condition_counts'], {'normal':16})
            old_plan = pd.read_csv(old.manifest_root/'target_exposures_epoch01.csv')
            new_plan = pd.read_csv(a.manifest_root/strategy/'target_exposures_epoch01.csv')
            self.assertTrue(old_plan[['crop_path','video_id','roi_status','flip']].equals(
                new_plan[['crop_path','video_id','roi_status','flip']]))
            self.assertEqual(old_plan.condition.tolist(), new_plan.scheduled_condition.tolist())
            # Exercise trained-student inference even if random tiny data selects epoch0.
            mt.base.common.save(run/'best.pt', {key:state[key] for key in
                ['identity','config','model_state_dict','epoch','best']} | {'validation':state['history'][-1]['validation']})
            done = json.loads((run/'completed.json').read_text())
            done['selected_epoch'] = state['epoch']
            mt.base.common.json_write(run/'completed.json', done)
        hashes = {str(p):mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        views.train(a)
        self.assertEqual(hashes, {str(p):mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        with patch.object(mt, 'infer', wraps=mt.infer) as infer:
            views.evaluate(a)
            self.assertEqual(infer.call_count, 8)
            self.assertTrue(all(any(strategy in call.args[-1] for strategy in views.STRATEGIES) for call in infer.call_args_list))
        table = pd.read_csv(a.output_root/'comparison_with_reference.csv')
        self.assertEqual(len(table), 24)
        self.assertEqual(set(table.strategy), {*views.REFERENCE_STRATEGIES, *views.STRATEGIES})
        summary = pd.read_csv(a.output_root/'domain_and_smoothing.csv')
        self.assertEqual(len(summary), 6)
        normal = table[table.condition.eq('normal')].set_index('strategy')
        strong = table[table.condition.eq('smooth_strong')].set_index('strategy')
        for row in summary.to_dict('records'):
            s = row['strategy']
            self.assertAlmostEqual(row['normal_balanced_error_rate'], normal.loc[s,'balanced_error_rate'])
            self.assertAlmostEqual(row['strong_minus_normal_real_fpr'], strong.loc[s,'real_fpr']-normal.loc[s,'real_fpr'])
            self.assertTrue(0 <= row['pooled_auc'] <= 1)
        before = {p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        with patch.object(mt, 'infer', side_effect=AssertionError('Should reuse saved inference')):
            views.evaluate(a)
        self.assertEqual(before, {p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        for path, digest in protected.items():
            self.assertEqual(digest, mt.base.common.digest(Path(path)))
        # Reject incompatible old references BEFORE running new GPU inference.
        path = old.output_root/'clip_b16_mean_teacher_normal.json'
        original = path.read_text()
        report = json.loads(original)
        report['manifest_sha256'] = 'wrong'
        mt.base.common.json_write(path, report)
        try:
            with patch.object(mt, 'infer', side_effect=AssertionError('Should fail before inference')):
                with self.assertRaisesRegex(FileNotFoundError, 'Matching saved'):
                    views.evaluate(a)
        finally:
            path.write_text(original)

    def test_exact_restart_and_three_model_support(self):
        old = self.args
        old.strategies = ['mean_teacher']
        mt.train(old)
        a = self.new_args(old)
        a.models = list(mt.base.PREPROCESS)
        views.train(a)
        golden = {}
        for model in a.models:
            for strategy in a.strategies:
                golden[(model,strategy)] = torch.load(
                    a.runs_root/f'{model}_{strategy}_{views.PROTOCOL}_seed42/last.pt', weights_only=True)
        a.runs_root = self.root/'resumed_views'
        save = mt.base.common.save
        def interrupt(path, state):
            save(path, state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(mt.base.common, 'save', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                views.train(a)
        views.train(a)
        for (model,strategy), state in golden.items():
            resumed = torch.load(a.runs_root/f'{model}_{strategy}_{views.PROTOCOL}_seed42/last.pt', weights_only=True)
            self.assertEqual(state['history'], resumed['history'])
            for key in ['model_state_dict','teacher_state_dict']:
                self.assertTrue(all(torch.equal(v,resumed[key][k]) for k,v in state[key].items()))

    def test_notebook_matches_md_and_has_direct_progress_commands(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_teacher_views_seed42'
        text = (views.ROOT/'docs/colab'/f'{stem}.md').read_text()
        blocks = re.findall(r'```python\n(.*?)```', text, re.S)
        notebook = json.loads((views.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 7)
        self.assertEqual(code, blocks)
        for cell in code:
            self.assertFalse(cell.lstrip().startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(cell))
        self.assertIn("MODELS=['dinov2_b14']", blocks[1])
        self.assertIn('!python -u', blocks[4])
        self.assertIn('!python -u', blocks[5])
        self.assertNotIn('--target-weight', blocks[4])


if __name__ == '__main__':
    unittest.main()
