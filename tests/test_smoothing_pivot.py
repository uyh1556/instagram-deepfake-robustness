"""Small CPU execution/gradient checks; no GPU performance claims."""
import copy
import ast
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_pivot as pivot
import test_smoothing_preservation as fixture


class PivotTests(unittest.TestCase):
    def setUp(self):
        fixture.PreservationTests.setUp(self)
        a=self.args
        a.validation_cache=self.root/'validation_cache'
        a.strategies=list(pivot.STRATEGIES)
        a.encoder_lr=a.head_lr=.0001
        a.critic_lr=.001
        a.adversarial_weight=.1
        a.baseline_evaluation_root=None
        self.extra=[patch.object(pivot.base,'stage',side_effect=lambda model,*args:model.train().requires_grad_(True)),
                    patch.object(pivot.base,'head',side_effect=lambda model,name:model.fc)]
        for item in self.extra:item.start()

    def tearDown(self):
        for item in reversed(self.extra):item.stop()
        fixture.PreservationTests.tearDown(self)

    def test_gradient_reversal_class_heads_and_fallback(self):
        torch.manual_seed(42)
        critic=pivot.ConditionalCritic(8)
        labels=torch.tensor([0,0,1,1])
        strengths=torch.tensor([0,1,2,3])
        detected=torch.ones(4,dtype=torch.bool)
        logits=torch.randn(4,2,requires_grad=True)
        loss,_=pivot.critic_loss(critic,logits,labels,strengths,detected,1.)
        loss.backward()
        reverse_gradient=logits.grad.clone()
        reverse_head_gradient=critic.heads[0][0].weight.grad.clone()
        critic.zero_grad();logits.grad=None
        loss,_=pivot.critic_loss(critic,logits,labels,strengths,detected,-1.)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad,-reverse_gradient))
        self.assertTrue(torch.equal(critic.heads[0][0].weight.grad,reverse_head_gradient))
        critic.zero_grad();logits.grad=None
        zero,_=pivot.critic_loss(critic,logits,labels,strengths,torch.zeros_like(detected),1.)
        zero.backward()
        self.assertTrue(torch.equal(logits.grad,torch.zeros_like(logits)))
        # An auxiliary classifier must not learn strength from untouched fallbacks.
        pool,_,_=pivot.shared.source_inputs(self.args)
        ds=pivot.PairedSource(pool,self.args,fixture.build()[1],self.cfg,0)
        for i,row in pool.reset_index(drop=True).iterrows():
            clean,changed,label,strength,valid=ds[i]
            self.assertIn(strength,[1,2,3])
            if row.roi_status!='detected':
                self.assertFalse(valid)
                self.assertTrue(torch.equal(clean,changed))

    def test_train_evaluate_skip_and_baselines_preserved(self):
        a=self.args
        actual_target=a.target_root
        a.target_root=self.root/'CDF_is_not_available_during_training'
        pivot.train(a)
        pivot.train(a)
        a.target_root=actual_target
        for name in a.models:
            histories=[]
            for strategy in a.strategies:
                run=a.runs_root/pivot.run_name(name,strategy,a.seed)
                best=torch.load(run/'best.pt',weights_only=True)
                reference=json.loads((run/'epoch0_validation.json').read_text())
                self.assertLessEqual(best['best'],reference['mean_ber'])
                self.assertFalse(best['config']['target_training'])
                last=torch.load(run/'last.pt',weights_only=True)
                histories.append([x['epoch_sample_sha256'] for x in last['history']])
                self.assertTrue(any(not torch.equal(last['model_state_dict'][k],
                    torch.load(a.baseline_runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed42/best.pt',weights_only=True)['model_state_dict'][k])
                    for k in last['model_state_dict']))
            self.assertEqual(histories[0],histories[1])
            self.assertEqual(pivot.base.common.digest(a.baseline_runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed42/best.pt'),self.initial_hashes[name])
        from argparse import Namespace
        a.baseline_evaluation_root=self.root/'baseline_evaluation'
        ea=Namespace(data_root=a.target_root, manifest=self.test, protocol=a.protocol, cache_root=a.cache_root,
                     output_root=a.baseline_evaluation_root, runs_root=a.baseline_runs_root, models=a.models,
                     seed=42,batch_size=a.batch_size,workers=0,device='cpu')
        pivot.evaluation.prepare(ea)
        pivot.evaluation.evaluate(ea)
        pivot.evaluate(a)
        before={p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*.csv') if p.name not in ['comparison.csv','condition_means.csv','mixture_summary.csv','cross_condition_auc.csv']}
        pivot.evaluate(a)
        self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*.csv') if p.name in before})
        table=pd.read_csv(a.output_root/'comparison.csv')
        self.assertEqual(len(table),len(a.models)*(len(a.strategies)+1)*4)
        self.assertTrue(np.allclose(table.balanced_error_rate,.5*(table.real_fpr+table.fake_fnr)))
        mixture=pd.read_csv(a.output_root/'mixture_summary.csv')
        cross=pd.read_csv(a.output_root/'cross_condition_auc.csv')
        for row in mixture.itertuples():
            selected=table[(table.model==row.model)&(table.strategy==row.strategy)]
            self.assertAlmostEqual(row.balanced_error_rate,selected.balanced_error_rate.mean())
            selected_cross=cross[(cross.model==row.model)&(cross.strategy==row.strategy)]
            self.assertEqual(len(selected_cross),16)
            self.assertAlmostEqual(row.pooled_auc,selected_cross.auc.mean())
        for path in a.output_root.glob('*_normal.csv'):
            frame=pd.read_csv(path)
            self.assertIn('raw_margin',frame)
            self.assertTrue(np.isfinite(frame.raw_margin).all())

    def test_resume_matches_completed_epoch(self):
        a=self.args
        a.models=['clip_b16']
        pivot.train(a)
        expected={s:copy.deepcopy(torch.load(a.runs_root/pivot.run_name('clip_b16',s,42)/'last.pt',weights_only=True)) for s in a.strategies}
        a.runs_root=self.root/'resume_runs'
        real_validate=pivot.validate
        calls=0
        def interrupted(*args):
            nonlocal calls
            calls+=1
            if calls==3:raise InterruptedError('after epoch1 checkpoint')
            return real_validate(*args)
        with patch.object(pivot,'validate',side_effect=interrupted):
            with self.assertRaises(InterruptedError):pivot.train(a)
        pivot.train(a)
        for strategy in a.strategies:
            actual=torch.load(a.runs_root/pivot.run_name('clip_b16',strategy,42)/'last.pt',weights_only=True)
            self.assertEqual(expected[strategy]['history'],actual['history'])
            for key,value in expected[strategy]['model_state_dict'].items():
                self.assertTrue(torch.equal(value,actual['model_state_dict'][key]))
            for key,value in expected[strategy]['critic_state_dict'].items():
                self.assertTrue(torch.equal(value,actual['critic_state_dict'][key]))


    def test_notebook_code_cells_and_direct_training(self):
        from IPython.core.inputtransformer2 import TransformerManager
        path=pivot.ROOT/'notebooks/train_test_ffpp_c23_smoothing_pivot_seed42.ipynb'
        nb=json.loads(path.read_text())
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),7)
        for cell in code:
            self.assertNotEqual(cell.lstrip().splitlines()[0],'python')
            ast.parse(TransformerManager().transform_cell(cell))
        self.assertIn('!python -u',code[3])
        self.assertIn('--val-images 1200',code[3])
        self.assertNotIn('non_test',code[4])


if __name__=='__main__':unittest.main()
