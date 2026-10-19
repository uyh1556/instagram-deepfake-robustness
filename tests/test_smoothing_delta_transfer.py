"""Bounded CPU checks for new objectives, label isolation and Colab workflow."""
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
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_delta_transfer as delta
import test_smoothing_mean_teacher as fixture
import test_smoothing_preservation as pixels


class DeltaTransferTests(unittest.TestCase):
    setUp=fixture.MeanTeacherTests.setUp
    tearDown=fixture.MeanTeacherTests.tearDown

    def test_bounds_gradients_ranking_and_selection(self):
        f=torch.tensor([[[3.,4.],[6.,8.]],[[0.,2.],[0.,0.]]],requires_grad=True)
        donor=torch.tensor([[30.,40.],[0.,.01]],requires_grad=True)
        d=delta.bounded_delta(f,donor,.05)
        self.assertFalse(d.requires_grad)
        self.assertTrue((d.norm(dim=-1)<=f.detach().norm(dim=-1)*.05+1e-6).all())
        self.assertTrue((d.norm(dim=-1)<=donor.detach().norm(dim=-1)[:,None]+1e-6).all())
        self.assertTrue(torch.equal(delta.bounded_delta(f,donor,0.),torch.zeros_like(f)))
        (f+d).square().mean().backward()
        self.assertIsNone(donor.grad)
        self.assertGreater(f.grad.abs().sum().item(),0)
        labels=torch.tensor([0,1])
        logits=torch.tensor([[[0.,2.],[0.,1.]],[[0.,-2.],[0.,-1.]]],requires_grad=True)
        loss,ce,rank,_=delta.classification_and_ranking(logits,labels,.25)
        rank.backward()
        self.assertTrue((logits.grad[0,:,1]>0).all())
        self.assertTrue((logits.grad[1,:,1]<0).all())
        doubled=delta.classification_and_ranking(torch.cat([logits,logits],1),labels,.25)
        self.assertTrue(torch.allclose(loss,doubled[0]))
        shifted=logits.detach().clone();shifted[...,1]+=20
        self.assertTrue(torch.allclose(rank,delta.classification_and_ranking(shifted,labels,.25)[2]))
        self.assertTrue(delta.is_better({'mean_auc':.81,'mean_ber':.4},{'auc':.8,'ber':.2}))
        self.assertFalse(delta.is_better({'mean_auc':.79,'mean_ber':.1},{'auc':.8,'ber':.2}))
        self.assertTrue(delta.is_better({'mean_auc':.8,'mean_ber':.1},{'auc':.8,'ber':.2}))

    def test_donor_pixels_same_flip_no_labels_and_fallback(self):
        a=self.args
        _,_,target,cfg,data,_=delta.adaptation.dataset_inputs(a)
        ident={'test':'delta'}
        observed=delta.common.pixel_cache.BilateralCache(self.root/'observed',cfg['bilateral'],ident)
        probe=delta.common.pixel_cache.BilateralCache(self.root/'probe',cfg['bilateral'],ident)
        plan=delta.engine.target_plan(target.assign(class_index=0),32,42)
        self.assertEqual(plan,delta.engine.target_plan(target.assign(class_index=1),32,42))
        self.assertTrue(all('class_index' not in r and 'method' not in r for r in plan))
        row=target[target.roi_status.eq('detected')].iloc[0].to_dict()
        item={**row,'condition':'smooth_medium','flip':True}
        ds=delta.DonorViews([item],a.target_root,pixels.build()[1],cfg,observed,probe)
        x,y,valid=ds[0]
        with Image.open(a.target_root/row['crop_path']) as im:
            normal=im.convert('RGB')
        first=delta.adaptation.smooth(normal,.5,cfg['bilateral'])
        second=delta.adaptation.smooth(first,.25,cfg['bilateral'])
        t=pixels.build()[1]
        self.assertTrue(torch.equal(x,t(first.transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
        self.assertTrue(torch.equal(y,t(second.transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
        self.assertTrue(valid)
        item={**target[target.roi_status.ne('detected')].iloc[0].to_dict(),'condition':'smooth_strong','flip':False}
        x,y,valid=delta.DonorViews([item],a.target_root,t,cfg,observed,probe)[0]
        self.assertFalse(valid);self.assertTrue(torch.equal(x,y))

    def test_train_resume_evaluate_reuse_three_models(self):
        old=copy.copy(self.args)
        old.models=['xception','clip_b16','dinov2_b14']
        old.strategies=['augmentation','mean_teacher']
        delta.engine.train(old)
        delta.engine.evaluate(old)
        protected={str(p):delta.base.common.digest(p) for p in
            [*old.runs_root.rglob('*.pt'),*old.output_root.glob('*.csv'),Path(delta.engine.__file__)]}
        a=copy.copy(old)
        a.runs_root=self.root/'delta_runs';a.manifest_root=self.root/'delta_manifests';a.output_root=self.root/'delta_results'
        a.hard_manifest_root=old.manifest_root;a.hard_evaluations=old.output_root
        a.ranking_weight=.25;a.delta_cap_ratio=.05;a.strategies=list(delta.STRATEGIES)
        delta.train(a)
        histories=[]
        for name in a.models:
            for strategy in a.strategies:
                run=a.runs_root/delta.run_name(name,strategy,42)
                state=torch.load(run/'last.pt',weights_only=True)
                histories.append(state['history'])
                self.assertEqual(state['ema_state_dict'] is None,strategy=='ce_rank')
                self.assertTrue(any(not torch.equal(v,state['model_state_dict'][k]) for k,v in
                    torch.load(a.baseline_runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed42/best.pt',weights_only=True)['model_state_dict'].items()))
                for h in state['history']:
                    self.assertEqual(h['target_exposures'],16 if strategy=='target_delta' else 0)
                    self.assertLessEqual(h['applied_delta_relative_norm'],.050001)
                best=torch.load(run/'best.pt',weights_only=True)
                done=json.loads((run/'completed.json').read_text())
                self.assertEqual(done['best_mean_auc'],max([json.loads((run/'epoch0_validation.json').read_text())['mean_auc']]
                    +[h['validation']['mean_auc'] for h in state['history']]))
                self.assertEqual(best['epoch'],done['selected_epoch'])
        self.assertTrue(all([h['source_batches_sha256'] for h in hist]==[h['source_batches_sha256'] for h in histories[0]] for hist in histories))
        hashes={str(p):delta.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        delta.train(a)
        self.assertEqual(hashes,{str(p):delta.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        # Explicitly exercise trained-state inference even if this tiny fixture
        # happens to select epoch0. Only isolated temporary fixture files change.
        run=a.runs_root/delta.run_name('xception','target_delta',42)
        last=torch.load(run/'last.pt',weights_only=True)
        delta.base.common.save(run/'best.pt',{k:last[k] for k in
            ['identity','config','model_state_dict','epoch','best']}|{'validation':last['history'][-1]['validation']})
        done=json.loads((run/'completed.json').read_text());done['selected_epoch']=last['epoch']
        delta.base.common.json_write(run/'completed.json',done)
        delta.evaluate(a)
        table=pd.read_csv(a.output_root/'comparison_with_reference.csv')
        self.assertEqual(len(table),72)
        self.assertEqual(set(table.strategy),{'baseline','augmentation','mean_teacher',*delta.STRATEGIES})
        report=json.loads((a.output_root/'xception_target_delta_normal.json').read_text())
        self.assertGreater(report['selected_epoch'],0)
        self.assertNotIn('reused_predictions',report)
        before={p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        # One requested strategy must retain the completed others in the summary.
        a.strategies=['target_delta']
        delta.evaluate(a)
        self.assertEqual(len(pd.read_csv(a.output_root/'comparison_with_reference.csv')),72)
        self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        a.models=['clip_b16'];a.runs_root=self.root/'resumed'
        golden=torch.load(self.root/'delta_runs'/delta.run_name('clip_b16','target_delta',42)/'last.pt',weights_only=True)
        save=delta.base.common.save
        def interrupt(path,state):
            save(path,state)
            if path.name=='last.pt' and state['epoch']==1:
                raise RuntimeError('intentional interruption')
        with patch.object(delta.base.common,'save',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'intentional'):
            delta.train(a)
        delta.train(a)
        restored=torch.load(a.runs_root/delta.run_name('clip_b16','target_delta',42)/'last.pt',weights_only=True)
        self.assertEqual(golden['history'],restored['history'])
        for field in ['model_state_dict','ema_state_dict']:
            self.assertTrue(all(torch.equal(v,restored[field][k]) for k,v in golden[field].items()))
        for path,digest in protected.items():
            self.assertEqual(digest,delta.base.common.digest(Path(path)))

    def test_colab_cells_match_md_and_parse(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem='train_test_ffpp_c23_smoothing_delta_transfer_seed42'
        md=(delta.ROOT/'docs/colab'/f'{stem}.md').read_text()
        nb=json.loads((delta.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        blocks=re.findall(r'```python\n(.*?)```',md,re.S)
        self.assertEqual(blocks,[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'])
        self.assertEqual(len(blocks),7)
        for code in blocks:
            ast.parse(TransformerManager().transform_cell(code))


if __name__=='__main__':
    unittest.main()
