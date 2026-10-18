"""Small CPU checks for per-image worst-view source training and matched reuse."""
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
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_mean_teacher as mt
import ffpp_c23_smoothing_worst_view as worst
import test_smoothing_mean_teacher as fixture


class WorstViewTests(unittest.TestCase):
    setUp=fixture.MeanTeacherTests.setUp
    tearDown=fixture.MeanTeacherTests.tearDown

    def test_per_image_selection_correct_class_gradient_and_ties(self):
        labels=torch.tensor([0,1])
        logits=torch.tensor([[2.,0.],[0.,0.],[0.,2.],[0.,3.]],requires_grad=True)
        loss,stats=worst.worst_source_loss(logits,labels)
        expected=F.cross_entropy(torch.stack([logits[2],logits[1]]),labels)
        self.assertTrue(torch.equal(loss,expected))
        self.assertEqual(stats[2:].tolist(),[1.,0.,1.,0.,1.,1.])
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0],torch.zeros(2)))
        self.assertTrue(torch.equal(logits.grad[3],torch.zeros(2)))
        self.assertGreater(logits.grad[2,1].item(),0.)  # Smoothed real: lower fake logit.
        self.assertLess(logits.grad[1,1].item(),0.)  # Clean fake: raise fake logit.
        tied=torch.zeros(2,2,requires_grad=True)
        loss,stats=worst.worst_source_loss(tied,torch.tensor([0]))
        loss.backward()
        self.assertTrue(torch.equal(tied.grad[0],tied.grad[1]))
        self.assertTrue(torch.allclose(tied.grad,torch.tensor([[-.25,.25],[-.25,.25]])))
        self.assertEqual(stats[3].item(),1.)

    def test_three_model_train_resume_and_reused_old_results(self):
        old=copy.copy(self.args)
        old.models=['xception','clip_b16','dinov2_b14']
        old.strategies=['augmentation','mean_teacher']
        mt.train(old)
        mt.evaluate(old)
        protected={str(p):mt.base.common.digest(p) for p in
            [*old.runs_root.rglob('*.pt'),*old.output_root.glob('*.csv'),Path(mt.__file__)]}
        a=copy.copy(old)
        a.runs_root=self.root/'worst_runs'
        a.manifest_root=self.root/'worst_manifests'
        a.output_root=self.root/'worst_results'
        a.hard_manifest_root=old.manifest_root
        a.hard_evaluations=old.output_root
        a.strategies=list(worst.STRATEGIES)
        worst.train(a)
        for name in a.models:
            old_ts=torch.load(old.runs_root/mt.run_name(name,'mean_teacher',42)/'last.pt',weights_only=True)
            for strategy in a.strategies:
                run=a.runs_root/f'{name}_{strategy}_{worst.PROTOCOL}_seed42'
                state=torch.load(run/'last.pt',weights_only=True)
                enabled=strategy=='worst_view_ts'
                self.assertEqual(state['config']['target_enabled'],enabled)
                self.assertEqual(state['teacher_state_dict'] is not None,enabled)
                for row in state['history']:
                    self.assertGreaterEqual(row['source_ce']+1e-6,
                                            .5*(row['source_clean_ce']+row['source_smoothed_ce']))
                    self.assertEqual(row['target_exposures'],16 if enabled else 0)
                    if not enabled:
                        self.assertEqual(row['target_ce'],0.)
                        self.assertEqual(row['mean_target_weight'],0.)
                        self.assertEqual(row['pseudo_selected_counts'],{'real':0,'fake':0})
                for key in ['source_paths_sha256']:
                    self.assertEqual([r[key] for r in state['history']],[r[key] for r in old_ts['history']])
                if enabled:
                    self.assertEqual([r['target_plan_sha256'] for r in state['history']],
                                     [r['target_plan_sha256'] for r in old_ts['history']])
        hashes={str(p):mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')}
        worst.train(a)
        self.assertEqual(hashes,{str(p):mt.base.common.digest(p) for p in a.runs_root.rglob('*.pt')})
        run=a.runs_root/f'clip_b16_worst_view_ts_{worst.PROTOCOL}_seed42'
        golden=torch.load(run/'last.pt',weights_only=True)
        mt.base.common.save(run/'best.pt',{k:golden[k] for k in
            ['identity','config','model_state_dict','epoch','best']}|{'validation':golden['history'][-1]['validation']})
        done=json.loads((run/'completed.json').read_text())
        done['selected_epoch']=golden['epoch']
        mt.base.common.json_write(run/'completed.json',done)
        worst.evaluate(a)
        table=pd.read_csv(a.output_root/'comparison_with_reference.csv')
        self.assertEqual(len(table),60)
        self.assertEqual(set(table.strategy),{'baseline','augmentation','mean_teacher',*worst.STRATEGIES})
        deltas=pd.read_csv(a.output_root/'deltas_vs_matched.csv')
        self.assertEqual(len(deltas),24)
        self.assertTrue((deltas.reference_strategy==deltas.strategy.map(worst.MATCHED_REFERENCE)).all())
        times={p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        worst.evaluate(a)
        self.assertEqual(times,{p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        a.models=['clip_b16']
        a.strategies=['worst_view_ts']
        a.runs_root=self.root/'worst_resumed'
        save=mt.base.common.save
        def interrupt(path,state):
            save(path,state)
            if path.name=='last.pt' and state['epoch']==1:
                raise RuntimeError('intentional interruption')
        with patch.object(mt.base.common,'save',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'intentional'):
            worst.train(a)
        worst.train(a)
        resumed=torch.load(a.runs_root/f'clip_b16_worst_view_ts_{worst.PROTOCOL}_seed42/last.pt',weights_only=True)
        self.assertEqual(golden['history'],resumed['history'])
        for key in ['model_state_dict','teacher_state_dict']:
            self.assertTrue(all(torch.equal(v,resumed[key][k]) for k,v in golden[key].items()))
        self.assertEqual(mt.PROTOCOL,worst.reference.HARD_PROTOCOL)
        for path,digest in protected.items():
            self.assertEqual(digest,mt.base.common.digest(Path(path)))

    def test_notebook_matches_md(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem='train_test_ffpp_c23_smoothing_worst_view_seed42'
        text=(worst.ROOT/'docs/colab'/f'{stem}.md').read_text()
        blocks=re.findall(r'```python\n(.*?)```',text,re.S)
        nb=json.loads((worst.ROOT/'notebooks'/f'{stem}.ipynb').read_text())
        cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),7)
        self.assertEqual(cells,blocks)
        for code in cells:
            ast.parse(TransformerManager().transform_cell(code))
            self.assertFalse(code.lstrip().startswith('python\n'))
        self.assertIn("MODELS=['xception']",cells[1])
        self.assertIn("STRATEGIES=['worst_view_aug','worst_view_ts']",cells[1])
        self.assertNotIn('--pair-weight',text)
        self.assertIn('!python -u',cells[4])
        self.assertIn('!python -u',cells[5])


if __name__=='__main__':
    unittest.main()

