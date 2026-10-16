"""Soft targets and reuse of completed hard UDA, with small CPU fixtures."""
import ast
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
import ffpp_c23_smoothing_mean_teacher as mt
import ffpp_c23_smoothing_mean_teacher_soft as soft
import test_smoothing_mean_teacher as fixture


class SoftTeacherTests(unittest.TestCase):
    setUp = fixture.MeanTeacherTests.setUp
    tearDown = fixture.MeanTeacherTests.tearDown

    def test_soft_probability_gradient_mask_and_engine_restoration(self):
        probabilities = torch.tensor([[.08,.92],[.5,.5],[.92,.08]])
        teacher = probabilities.log().requires_grad_(True)
        student = torch.tensor([[.01,.99],[.2,.8],[.4,.6]]).log().requires_grad_(True)
        loss, stats, confidence = soft.soft_pseudo_loss(student,teacher,.9)
        loss.backward()
        expected = (student.detach().softmax(1)-probabilities)/3
        expected[1] = 0
        self.assertTrue(torch.allclose(student.grad,expected,atol=1e-7))
        self.assertGreater(student.grad[0,1].item(),0.)  # .99 should move DOWN to .92.
        self.assertIsNone(teacher.grad)
        self.assertEqual(stats.tolist(),[2,1,1,1])
        self.assertAlmostEqual(confidence.item(),2.34,places=6)
        x = probabilities.log().requires_grad_(True)
        matched, _, _ = soft.soft_pseudo_loss(x,teacher,.9)
        matched.backward()
        self.assertGreater(matched.item(),0.)  # Teacher entropy, not incorrect agreement.
        self.assertTrue(torch.allclose(x.grad,torch.zeros_like(x),atol=1e-7))
        x = torch.zeros(2,2,requires_grad=True)
        zero, _, _ = soft.soft_pseudo_loss(x,torch.zeros_like(x),.9)
        zero.backward()
        self.assertEqual(zero.item(),0.)
        self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))
        original = mt.pseudo_loss
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with soft.soft_protocol():
                self.assertIs(mt.pseudo_loss,soft.soft_pseudo_loss)
                raise RuntimeError('intentional')
        self.assertIs(mt.pseudo_loss,original)
        self.assertEqual(mt.PROTOCOL,soft.HARD_PROTOCOL)

    def test_train_reuse_comparison_resume_and_frozen_membership(self):
        old = copy.copy(self.args)
        old.strategies = ['augmentation','mean_teacher']
        mt.train(old)
        mt.evaluate(old)
        protected = {str(p):mt.base.common.digest(p) for p in
            [*old.runs_root.rglob('*.pt'),*old.output_root.glob('*.csv')]}
        a = copy.copy(old)
        a.runs_root, a.manifest_root, a.output_root = self.root/'soft_runs', self.root/'soft_manifests', self.root/'soft_results'
        a.hard_manifest_root, a.hard_evaluations = old.manifest_root, old.output_root
        a.strategies = list(soft.WEIGHTS)
        soft.train(a)
        histories = []
        for strategy, weight in soft.WEIGHTS.items():
            run = a.runs_root/f'clip_b16_{strategy}_{soft.PROTOCOL}_seed42'
            state = torch.load(run/'last.pt',weights_only=True)
            self.assertEqual(state['config']['target_weight'],weight)
            self.assertEqual(state['config']['pseudo_label_kind'],'soft')
            histories.append(state['history'])
        hard = json.loads((old.runs_root/mt.run_name('clip_b16','mean_teacher',42)/'history.json').read_text())
        for history in histories:
            for key in ['source_paths_sha256','target_plan_sha256','target_condition_counts']:
                self.assertEqual([h[key] for h in history],[h[key] for h in hard])
        self.assertEqual(histories[0][0]['mean_target_weight'],2*histories[1][0]['mean_target_weight'])
        hashes = {str(p):mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        soft.train(a)
        self.assertEqual(hashes,{str(p):mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        # Tiny random fixtures can select epoch0. Exercise trained-student inference too.
        run = a.runs_root/f'clip_b16_soft_w05_{soft.PROTOCOL}_seed42'
        trained = torch.load(run/'last.pt',weights_only=True)
        mt.base.common.save(run/'best.pt', {key:trained[key] for key in
            ['identity','config','model_state_dict','epoch','best']} | {'validation':trained['history'][-1]['validation']})
        done = json.loads((run/'completed.json').read_text())
        done['selected_epoch'] = trained['epoch']
        mt.base.common.json_write(run/'completed.json',done)
        soft.evaluate(a)
        combined = pd.read_csv(a.output_root/'comparison_with_hard.csv')
        self.assertEqual(len(combined),20)
        self.assertEqual(set(combined.strategy),{'baseline','augmentation','mean_teacher',*soft.WEIGHTS})
        baseline = json.loads((a.output_root/'clip_b16_baseline_normal.json').read_text())
        self.assertIn('reused_predictions',baseline)
        self.assertNotIn('reused_predictions',json.loads((a.output_root/'clip_b16_soft_w05_normal.json').read_text()))
        before = {p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        soft.evaluate(a)
        self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        bad = old.output_root/'clip_b16_mean_teacher_normal.json'
        original_report = bad.read_text()
        values = json.loads(original_report)
        values['manifest_sha256'] = 'different'
        mt.base.common.json_write(bad,values)
        try:
            with self.assertRaisesRegex(FileNotFoundError,'Matching saved'):
                soft.evaluate(a)
        finally:
            bad.write_text(original_report)
        golden = torch.load(a.runs_root/f'clip_b16_soft_w025_{soft.PROTOCOL}_seed42/last.pt',weights_only=True)
        a.runs_root = self.root/'resumed_soft'
        a.strategies = ['soft_w025']
        save = mt.base.common.save
        def interrupt(path,state):
            save(path,state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(mt.base.common,'save',side_effect=interrupt), self.assertRaisesRegex(RuntimeError,'intentional'):
            soft.train(a)
        soft.train(a)
        resumed = torch.load(a.runs_root/f'clip_b16_soft_w025_{soft.PROTOCOL}_seed42/last.pt',weights_only=True)
        self.assertEqual(golden['history'],resumed['history'])
        for key in ['model_state_dict','teacher_state_dict']:
            self.assertTrue(all(torch.equal(v,resumed[key][k]) for k,v in golden[key].items()))
        for path,digest in protected.items():
            self.assertEqual(digest,mt.base.common.digest(Path(path)))
        target = old.manifest_root/'target_unlabelled.csv'
        frame = pd.read_csv(target)
        frame = frame.iloc[::-1]
        frame.to_csv(target,index=False)
        with self.assertRaisesRegex(ValueError,'membership differs'):
            soft.train(a)

    def test_notebook_matches_md_and_cli(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem = 'train_test_ffpp_c23_smoothing_mean_teacher_soft_seed42'
        text = (soft.ROOT/'docs/colab'/f'{stem}.md').read_text()
        import re
        blocks = re.findall(r'```python\n(.*?)```',text,re.S)
        notebook = json.loads((soft.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        cells = [c for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),7)
        self.assertEqual([''.join(c['source']) for c in cells],blocks)
        for cell in cells:
            code = ''.join(cell['source'])
            self.assertFalse(code.lstrip().startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(code))
        self.assertIn("MODELS=['clip_b16','xception']",blocks[1])
        self.assertNotIn('--target-weight',blocks[4])  # Weights belong to named conditions.


if __name__ == '__main__':
    unittest.main()
