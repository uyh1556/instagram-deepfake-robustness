"""CPU checks for paired source regularization, historical reuse and exact resume."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_mean_teacher as mt
import ffpp_c23_smoothing_mean_teacher_pair as pair
import test_smoothing_mean_teacher as fixture


class PairedTeacherTests(unittest.TestCase):
    setUp = fixture.MeanTeacherTests.setUp
    tearDown = fixture.MeanTeacherTests.tearDown

    def test_binary_margin_huber_gradients_and_fallback(self):
        logits=torch.tensor([[0.,0.],[0.,1.],[0.,2.],[0.,8.]],requires_grad=True)
        value=pair.source_pair_loss(logits,2,torch.tensor([True,False]))
        self.assertAlmostEqual(value.item(),.75)
        value.backward()
        self.assertTrue(torch.equal(logits.grad[1],torch.zeros(2)))
        self.assertTrue(torch.equal(logits.grad[3],torch.zeros(2)))
        self.assertGreater(logits.grad[0,0].item(),0.)
        self.assertLess(logits.grad[0,1].item(),0.)
        self.assertLess(logits.grad[2,0].item(),0.)
        self.assertGreater(logits.grad[2,1].item(),0.)
        # Both current predictions move toward one another; neither is detached.
        equal=torch.tensor([[0.,2.],[3.,5.]],requires_grad=True)
        zero=pair.source_pair_loss(equal,1,torch.tensor([True]))
        zero.backward()
        self.assertEqual(zero.item(),0.)
        self.assertTrue(torch.equal(equal.grad,torch.zeros_like(equal)))
        empty=pair.source_pair_loss(logits,2,torch.tensor([False,False]))
        self.assertEqual(empty.item(),0.)

    def test_training_matches_hard_at_zero_then_resumes_and_reuses_results(self):
        old=copy.copy(self.args)
        old.strategies=['augmentation','mean_teacher']
        mt.train(old)
        mt.evaluate(old)
        protected={str(p):mt.base.common.digest(p) for p in
            [*old.runs_root.rglob('*.pt'),*old.output_root.glob('*.csv'),Path(mt.__file__)]}
        a=copy.copy(old)
        a.runs_root=self.root/'pair_zero_runs'
        a.manifest_root=self.root/'pair_manifests'
        a.output_root=self.root/'pair_results'
        a.hard_manifest_root=old.manifest_root
        a.hard_evaluations=old.output_root
        a.strategies=['mean_teacher_pair']
        a.pair_weight=0.
        pair.train(a)
        original=torch.load(old.runs_root/mt.run_name('clip_b16','mean_teacher',42)/'last.pt',weights_only=True)
        zero=torch.load(a.runs_root/f'clip_b16_mean_teacher_pair_{pair.PROTOCOL}_seed42/last.pt',weights_only=True)
        for key in ['model_state_dict','teacher_state_dict']:
            self.assertTrue(all(torch.equal(v,zero[key][k]) for k,v in original[key].items()))
        for key in ['source_paths_sha256','target_plan_sha256','target_condition_counts']:
            self.assertEqual([r[key] for r in zero['history']],[r[key] for r in original['history']])
        self.assertEqual(mt.PROTOCOL,pair.reference.HARD_PROTOCOL)
        a.pair_weight=.1
        a.runs_root=self.root/'pair_positive_runs'
        pair.train(a)
        run=a.runs_root/f'clip_b16_mean_teacher_pair_{pair.PROTOCOL}_seed42'
        golden=torch.load(run/'last.pt',weights_only=True)
        self.assertGreater(golden['history'][0]['source_pair_huber'],0.)
        self.assertTrue(any(not torch.equal(v,zero['model_state_dict'][k])
                            for k,v in golden['model_state_dict'].items()))
        before=mt.base.common.digest(run/'last.pt')
        pair.train(a)
        self.assertEqual(before,mt.base.common.digest(run/'last.pt'))
        # Exercise a trained student's inference even if the tiny fixture selected epoch0.
        mt.base.common.save(run/'best.pt',{k:golden[k] for k in
            ['identity','config','model_state_dict','epoch','best']}|{'validation':golden['history'][-1]['validation']})
        done=json.loads((run/'completed.json').read_text())
        done['selected_epoch']=golden['epoch']
        mt.base.common.json_write(run/'completed.json',done)
        pair.evaluate(a)
        table=pd.read_csv(a.output_root/'comparison_with_hard.csv')
        self.assertEqual(len(table),16)
        self.assertEqual(set(table.strategy),{'baseline','augmentation','mean_teacher','mean_teacher_pair'})
        self.assertEqual(len(pd.read_csv(a.output_root/'smoothing_changes.csv')),4)
        times={p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        pair.evaluate(a)
        self.assertEqual(times,{p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        a.runs_root=self.root/'pair_resumed_runs'
        save=mt.base.common.save
        def interrupt(path,state):
            save(path,state)
            if path.name=='last.pt' and state['epoch']==1:
                raise RuntimeError('intentional interruption')
        with patch.object(mt.base.common,'save',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'intentional'):
            pair.train(a)
        pair.train(a)
        resumed=torch.load(a.runs_root/f'clip_b16_mean_teacher_pair_{pair.PROTOCOL}_seed42/last.pt',weights_only=True)
        self.assertEqual(golden['history'],resumed['history'])
        for key in ['model_state_dict','teacher_state_dict']:
            self.assertTrue(all(torch.equal(v,resumed[key][k]) for k,v in golden[key].items()))
        for path,digest in protected.items():
            self.assertEqual(digest,mt.base.common.digest(Path(path)))

    def test_notebook_and_cli(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem='train_test_ffpp_c23_smoothing_mean_teacher_pair_seed42'
        md=(pair.ROOT/'docs/colab'/f'{stem}.md').read_text()
        blocks=re.findall(r'```python\n(.*?)```',md,re.S)
        nb=json.loads((pair.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),7)
        self.assertEqual(blocks,code)
        for cell in code:
            ast.parse(TransformerManager().transform_cell(cell))
            self.assertFalse(cell.lstrip().startswith('python\n'))
        self.assertIn("MODELS=['xception']",code[1])
        self.assertIn('--pair-weight 0.1',code[4])
        self.assertNotIn('soft_w',md)
        self.assertIn('!python -u',code[4])
        self.assertIn('!python -u',code[5])


if __name__=='__main__':
    unittest.main()

