"""CPU checks of selection, matched random quotas, pixels, EMA and restart."""
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
import ffpp_c23_smoothing_stable_uda as stable
import test_smoothing_mean_teacher as fixture
import test_smoothing_preservation as pixels


class StableSelectionTests(unittest.TestCase):
    setUp = fixture.MeanTeacherTests.setUp
    tearDown = fixture.MeanTeacherTests.tearDown

    def new_args(self, old):
        a = copy.copy(old)
        a.runs_root, a.manifest_root, a.output_root = self.root/'stable_runs', self.root/'stable_manifests', self.root/'stable_results'
        a.hard_manifest_root, a.hard_evaluations = old.manifest_root, old.output_root
        a.strategies, a.teacher_batch_size = list(stable.STRATEGIES), 5
        a.target_weight = .5
        return a

    def test_masks_random_quotas_and_all_batch_loss(self):
        # Both classes include a stable high-confidence, an unstable high-confidence,
        # and a stable low-confidence example. Only first/high-confidence may gate.
        observed = torch.tensor([[8., -8.], [8., -8.], [.1, 0.],
                                 [-8., 8.], [-8., 8.], [0., .1]])
        teacher = observed[:, None].repeat(1, 4, 1).requires_grad_(True)
        with torch.no_grad():
            teacher[1, 2] = torch.tensor([-8., 8.])
            teacher[4, 3] = torch.tensor([8., -8.])
            # Non-observed views need not be confident, merely agree.
            teacher[0, 1] = torch.tensor([.01, 0.])
        masks = stable.selection_masks(teacher, .9, 'stable_agreement', 42)
        self.assertEqual(masks[2].tolist(), [True, True, False, True, True, False])
        self.assertEqual(masks[3].tolist(), [True, False, False, True, False, False])
        for seed in range(30):
            labels, _, candidate, agree, selected = stable.selection_masks(teacher, .9, 'random_agreement_count', seed)
            for label in [0, 1]:
                self.assertEqual(int((selected & labels.eq(label)).sum()), int((agree & labels.eq(label)).sum()))
            self.assertFalse((selected & ~candidate).any())
            again = stable.selection_masks(teacher, .9, 'random_agreement_count', seed)[-1]
            self.assertTrue(torch.equal(selected, again))
        before = torch.get_rng_state().clone()
        student = torch.zeros(6, 2, requires_grad=True)
        log = stable.SelectionLog('stable_agreement')
        loss, stats, _ = log.loss(student, teacher, .9)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertAlmostEqual(loss.item(), 2*np.log(2)/6, places=6)
        self.assertEqual(stats.tolist(), [3, 3, 1, 1])
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(student.grad[[1, 2, 4, 5]], torch.zeros(4, 2)))
        empty = torch.zeros(2, 2, requires_grad=True)
        loss, stats, _ = stable.SelectionLog('random_agreement_count').loss(empty, torch.zeros(2, 4, 2), .9)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats[2:].sum(), 0)
        self.assertTrue(torch.equal(empty.grad, torch.zeros_like(empty)))
        # The random control is allowed to select an unstable but confident example.
        self.assertTrue(any((stable.selection_masks(teacher, .9, 'random_agreement_count', s)[-1] & ~masks[3]).any() for s in range(30)))

    def test_native_views_observed_order_fallback_and_ema(self):
        a = self.args
        _, _, target, cfg, _, _ = mt.adaptation.dataset_inputs(a)
        a._target_cache = mt.common.pixel_cache.BilateralCache(a.target_train_cache/'pixels', cfg['bilateral'], {'test': True})
        original_plan = mt.target_plan(target, 100, 42)
        self.assertEqual(original_plan, mt.target_plan(target.assign(class_index=1, method='anything'), 100, 42))
        for status in ['detected', 'fallback_letterbox']:
            for condition in mt.CONDITIONS:
                row = next(r for r in original_plan if r['roi_status'] == status)
                for flip in [False, True]:
                    row = {**row, 'condition': condition, 'flip': flip}
                    with Image.open(a.target_root/row['crop_path']) as image:
                        raw = image.convert('RGB')
                    order = [condition, *[c for c in mt.CONDITIONS if c != condition]]
                    ds = stable.TargetViews([row], a, pixels.build()[1], cfg, False)
                    with patch.object(a._target_cache, 'filtered', wraps=a._target_cache.filtered) as filtered:
                        teacher, student = ds[0]
                        self.assertEqual(filtered.call_count, int(status == 'detected'))
                    for i, c in enumerate(order):
                        expected = (mt.adaptation.smooth(raw, cfg['conditions'][c], cfg['bilateral'])
                                    if status == 'detected' and c != 'normal' else raw)
                        if flip:
                            expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        self.assertTrue(torch.equal(teacher[i], pixels.build()[1](expected)))
                    self.assertTrue(torch.equal(teacher[0].flip(-1), student))
                    if status != 'detected':
                        self.assertTrue(torch.equal(teacher, teacher[0].expand_as(teacher)))
        model = pixels.Tiny().eval()
        teacher = stable.MultiViewTeacher(copy.deepcopy(model), 3).eval().requires_grad_(False)
        x = torch.rand(2, 4, 3, 12, 12)
        expected = model(x.flatten(0, 1)).reshape(2, 4, 2)
        self.assertTrue(torch.allclose(expected, teacher(x), atol=1e-6))
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.)
        previous = [p.clone() for p in teacher.parameters()]
        mt.update_teacher(teacher, model, .75)
        for old, new, student in zip(previous, teacher.parameters(), model.parameters()):
            self.assertTrue(torch.allclose(new, .75*old+.25*student))

    def test_training_restart_and_three_model_paths(self):
        old = self.args
        old.strategies = ['mean_teacher']
        mt.train(old)
        a = self.new_args(old)
        a.models = list(mt.base.PREPROCESS)
        protected = {p: mt.base.common.digest(p) for p in old.runs_root.rglob('*.pt')}
        hooks = (mt.PROTOCOL, mt.configuration, mt.TargetViews, mt.copy, mt.pseudo_loss, mt.base.common.save)
        stable.train(a)
        self.assertEqual(hooks, (mt.PROTOCOL, mt.configuration, mt.TargetViews, mt.copy, mt.pseudo_loss, mt.base.common.save))
        golden = {}
        for model in a.models:
            for strategy in a.strategies:
                run = a.runs_root/f'{model}_{strategy}_{stable.PROTOCOL}_seed42'
                state = torch.load(run/'last.pt', weights_only=True)
                golden[(model, strategy)] = state
                self.assertTrue(all(key.startswith('model.') for key in state['teacher_state_dict']))
                self.assertEqual(state['config']['target_weight'], .5)
                self.assertEqual(state['config']['extension_code_sha256'], mt.base.common.digest(Path(stable.__file__)))
                for h in state['history']:
                    s = h['selection_diagnostics']
                    self.assertEqual(h['target_exposures'], 16)
                    self.assertEqual(s['images'], 16)
                    self.assertEqual(s['selected_real'], s['agreement_real'])
                    self.assertEqual(s['selected_fake'], s['agreement_fake'])
                    if strategy == 'stable_agreement':
                        self.assertEqual(s['selected_unstable'], 0)
                    self.assertTrue((run/f'selection_epoch{h["epoch"]:02d}.csv').is_file())
        hashes = {p: mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        stable.train(a)
        self.assertEqual(hashes, {p: mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        a.runs_root = self.root/'resumed'
        save = mt.base.common.save
        def interrupt(path, state):
            save(path, state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(mt.base.common, 'save', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                stable.train(a)
        stable.train(a)
        for (model, strategy), expected in golden.items():
            actual = torch.load(a.runs_root/f'{model}_{strategy}_{stable.PROTOCOL}_seed42/last.pt', weights_only=True)
            self.assertEqual(expected['history'], actual['history'])
            for key in ['model_state_dict', 'teacher_state_dict']:
                self.assertTrue(all(torch.equal(value, actual[key][name]) for name, value in expected[key].items()))
        self.assertEqual(protected, {p: mt.base.common.digest(p) for p in protected})

    def test_evaluation_reuses_references_and_preserves_old_files(self):
        old = self.args
        mt.train(old)
        mt.evaluate(old)
        protected = {p: mt.base.common.digest(p) for p in [*old.runs_root.rglob('*.pt'), *old.output_root.glob('*.csv')]}
        a = self.new_args(old)
        stable.train(a)
        # Tiny random data can choose epoch0; force a trained state for inference coverage.
        for strategy in a.strategies:
            run = a.runs_root/f'clip_b16_{strategy}_{stable.PROTOCOL}_seed42'
            state = torch.load(run/'last.pt', weights_only=True)
            mt.base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']}
                                | {'validation': state['history'][-1]['validation']})
        with patch.object(mt, 'infer', wraps=mt.infer) as infer:
            stable.evaluate(a)
            self.assertEqual(infer.call_count, 8)
            self.assertTrue(all(any(s in call.args[-1] for s in stable.STRATEGIES) for call in infer.call_args_list))
        table = pd.read_csv(a.output_root/'comparison_with_reference.csv')
        self.assertEqual(len(table), 20)
        self.assertEqual(set(table.strategy), {'baseline', 'augmentation', 'mean_teacher', *stable.STRATEGIES})
        self.assertTrue(np.allclose(table.balanced_error_rate, (table.real_fpr+table.fake_fnr)/2))
        deltas = pd.read_csv(a.output_root/'deltas_vs_reference.csv')
        self.assertEqual(len(deltas), 30)
        self.assertTrue((deltas.both_errors_lower == (deltas.delta_real_fpr.lt(0) & deltas.delta_fake_fnr.lt(0))).all())
        with patch.object(mt, 'infer', side_effect=AssertionError('Must reuse saved predictions')):
            stable.evaluate(a)
        self.assertEqual(protected, {p: mt.base.common.digest(p) for p in protected})
        report = old.output_root/'clip_b16_mean_teacher_normal.json'
        original = report.read_text()
        values = json.loads(original)
        values['manifest_sha256'] = 'wrong'
        mt.base.common.json_write(report, values)
        try:
            with patch.object(mt, 'infer', side_effect=AssertionError('Must fail before new inference')):
                with self.assertRaisesRegex(FileNotFoundError, 'Matching saved'):
                    stable.evaluate(a)
        finally:
            report.write_text(original)

    def test_notebook_matches_md_and_progress_is_direct(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_stable_uda_seed42'
        text = (stable.ROOT/'docs/colab'/f'{stem}.md').read_text()
        blocks = re.findall(r'```python\n(.*?)```', text, re.S)
        notebook = json.loads((stable.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 7)
        self.assertEqual(cells, blocks)
        for cell in cells:
            self.assertFalse(cell.lstrip().startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(cell))
        self.assertIn("MODELS=['xception']", cells[1])
        self.assertIn('!python -u', cells[4])
        self.assertIn('!python -u', cells[5])
        self.assertIn('deltas_vs_reference.csv', cells[6])


if __name__ == '__main__':
    unittest.main()
