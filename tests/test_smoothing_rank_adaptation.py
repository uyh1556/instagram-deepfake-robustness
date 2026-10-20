"""Small CPU integration checks for the matched rank adaptation comparison."""
import ast
import copy
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_rank_adaptation as method
import test_smoothing_mean_teacher as fixture


class RankAdaptationTests(unittest.TestCase):
    setUp = fixture.MeanTeacherTests.setUp
    tearDown = fixture.MeanTeacherTests.tearDown

    def setup_references(self):
        old = copy.copy(self.args)
        old.models = ['dinov2_b14']
        old.strategies = ['augmentation']
        method.engine.train(old)
        ce = copy.copy(old)
        ce.hard_manifest_root = old.manifest_root
        ce.runs_root = self.root/'ce_runs'
        ce.manifest_root = self.root/'ce_manifests'
        ce.output_root = self.root/'ce_results'
        ce.ranking_weight, ce.delta_cap_ratio = .25, .05
        ce.strategies = ['ce_rank']
        method.rank.train(ce)
        with method.rank.protocol_context():
            method.engine.evaluate(ce)
        a = copy.copy(ce)
        a.ce_rank_runs_root, a.ce_rank_evaluations = ce.runs_root, ce.output_root
        a.runs_root, a.manifest_root, a.output_root = self.root/'new_runs', self.root/'new_manifests', self.root/'new_results'
        a.strategies = list(method.STRATEGIES)
        return a

    def test_target_label_isolation_loss_and_metadata(self):
        a = self.args
        _, _, target, _, _, _ = method.adaptation.dataset_inputs(a)
        labelled = method.labelled_target(target, a.target_root)
        plan = method.engine.target_plan(target, 32, 42)
        self.assertEqual(plan, method.engine.target_plan(labelled.assign(class_index=1), 32, 42))
        self.assertTrue(all('class_index' not in p for p in plan))
        student = torch.zeros(3, 2, requires_grad=True)
        teacher = torch.tensor([[8., -8.], [0., 0.], [-8., 8.]], requires_grad=True)
        loss, stats, _ = method.target_objective(student, teacher, None, 'rank_ts', .9)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(student.grad[1], torch.zeros(2)))
        self.assertEqual(stats.tolist(), [2, 1, 1, 1])
        student = torch.zeros(3, 2, requires_grad=True)
        loss, stats, _ = method.target_objective(student, None, torch.tensor([0, 1, 1]), 'rank_sda', .9)
        loss.backward()
        self.assertGreater(student.grad[1].abs().sum().item(), 0)
        self.assertEqual(stats.tolist(), [1, 2, 1, 2])
        bad = target.copy(); bad.loc[0, 'video_id'] = 'test_video'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            method.labelled_target(bad, a.target_root)

    def test_train_resume_and_reuse_existing_ce(self):
        a = self.setup_references()
        protected = {str(p):method.base.common.digest(p) for root in [a.ce_rank_runs_root, a.ce_rank_evaluations, a.baseline_runs_root]
                     for p in root.rglob('*') if p.is_file()}
        method.prepare(a)
        method.train(a)
        histories = []
        for strategy in a.strategies:
            run = a.runs_root/method.run_name('dinov2_b14', strategy, 42)
            state = torch.load(run/'last.pt', weights_only=True)
            histories.append(state['history'])
            self.assertEqual(state['teacher_state_dict'] is None, strategy == 'rank_sda')
            self.assertEqual(state['config']['target_labels_used'], strategy == 'rank_sda')
            if strategy == 'rank_ts':
                self.assertIsNone(state['config']['target_label_sha256'])
                self.assertIsNone(state['config']['target_class_counts'])
            self.assertTrue(all(h['target_exposures'] == 16 for h in state['history']))
            if strategy == 'rank_sda':
                self.assertTrue(all(h['target_selected_fraction'] == 1 for h in state['history']))
            done = json.loads((run/'completed.json').read_text())
            self.assertEqual(done['best_mean_auc'], max([json.loads((run/'epoch0_validation.json').read_text())['mean_auc']]
                + [h['validation']['mean_auc'] for h in state['history']]))
        for field in ['source_paths_sha256', 'source_batches_sha256', 'target_plan_sha256']:
            self.assertEqual([h[field] for h in histories[0]], [h[field] for h in histories[1]])
        ce_history = json.loads(next(a.ce_rank_runs_root.rglob('history.json')).read_text())
        self.assertEqual([h['source_batches_sha256'] for h in histories[0]], [h['source_batches_sha256'] for h in ce_history])
        saved = {str(p):method.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        method.train(a)
        self.assertEqual(saved, {str(p):method.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        # Exercise trained checkpoint inference even when a tiny fixture selects
        # epoch0. Only isolated temporary test outputs are changed here.
        run = a.runs_root/method.run_name('dinov2_b14', 'rank_sda', 42)
        last = torch.load(run/'last.pt', weights_only=True)
        method.base.common.save(run/'best.pt', {k:last[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']}
            | {'validation':last['history'][-1]['validation']})
        done = json.loads((run/'completed.json').read_text()); done['selected_epoch'] = last['epoch']
        method.base.common.json_write(run/'completed.json', done)
        method.evaluate(a)
        table = pd.read_csv(a.output_root/'comparison_with_ce_rank.csv')
        self.assertEqual(len(table), 16)
        self.assertEqual(set(table.strategy), {'baseline', 'ce_rank', 'rank_ts', 'rank_sda'})
        trained_report = json.loads((a.output_root/'dinov2_b14_rank_sda_normal.json').read_text())
        self.assertGreater(trained_report['selected_epoch'], 0)
        self.assertNotIn('reused_predictions', trained_report)
        a.strategies = ['rank_ts']
        times = {p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        method.evaluate(a)
        self.assertEqual(len(pd.read_csv(a.output_root/'comparison_with_ce_rank.csv')), 16)
        self.assertEqual(times, {p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        a.runs_root = self.root/'resumed'
        save = method.base.common.save
        def interrupt(path, state):
            save(path, state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(method.base.common, 'save', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'intentional'):
            method.train(a)
        method.train(a)
        restored = torch.load(a.runs_root/method.run_name('dinov2_b14', 'rank_ts', 42)/'last.pt', weights_only=True)
        golden = torch.load(self.root/'new_runs'/method.run_name('dinov2_b14', 'rank_ts', 42)/'last.pt', weights_only=True)
        self.assertEqual(golden['history'], restored['history'])
        for field in ['model_state_dict', 'teacher_state_dict']:
            self.assertTrue(all(torch.equal(v, restored[field][k]) for k, v in golden[field].items()))
        # With target weight zero the SAME source objective/optimizer must
        # reproduce the completed CE+Rank student exactly, not just its sampler.
        zero = copy.copy(a)
        zero.runs_root = self.root/'zero_target'
        zero.target_weight = 0.
        with patch.object(method, 'labelled_target', side_effect=AssertionError('TS requested GT')):
            method.train(zero)
        state = torch.load(zero.runs_root/method.run_name('dinov2_b14', 'rank_ts', 42)/'last.pt', weights_only=True)
        ce_state = torch.load(next(a.ce_rank_runs_root.rglob('last.pt')), weights_only=True)
        self.assertTrue(all(torch.equal(v, state['model_state_dict'][k]) for k, v in ce_state['model_state_dict'].items()))
        for path, digest in protected.items():
            self.assertEqual(digest, method.base.common.digest(Path(path)))
        changed = copy.copy(a); changed.ranking_weight = .5
        with self.assertRaisesRegex(ValueError, 'matched control'):
            method.prepare(changed)

    def test_colab(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_rank_adaptation_seed42'
        text = (method.ROOT/'docs/colab'/f'{stem}.md').read_text()
        notebook = json.loads((method.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        cells = re.findall(r'```python\n(.*?)```', text, re.S)
        self.assertEqual(cells, [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'])
        self.assertEqual(len(cells), 7)
        for cell in cells:
            self.assertFalse(cell.lstrip().startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(cell))


if __name__ == '__main__':
    unittest.main()
