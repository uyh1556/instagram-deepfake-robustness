"""Small CPU tests for label isolation, EMA, native views, resume and evaluation."""
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import ffpp_c23_smoothing_mean_teacher as mt
import test_smoothing_preservation as fixture


class MeanTeacherTests(unittest.TestCase):
    def setUp(self):
        fixture.PreservationTests.setUp(self)
        a = self.args
        a.models = ['clip_b16']
        a.strategies = list(mt.STRATEGIES)
        a.validation_cache = self.root/'val_cache'
        a.train_cache = self.root/'train_cache'
        a.target_train_cache = self.root/'target_cache'
        a.evaluation_pixel_cache = self.root/'evaluation_pixels'
        a.reference_evaluations = None
        a.target_images = 16
        a.encoder_lr = a.head_lr = .0001
        a.ema_decay, a.pseudo_threshold, a.target_weight, a.ramp_epochs = .9, .5, .5, 2.
        self.extra = [patch.object(mt.base, 'stage', side_effect=lambda model,*args: model.train().requires_grad_(True)),
                      patch.object(mt.base, 'head', side_effect=lambda model,name: model.fc)]
        for p in self.extra:
            p.start()
        rng = np.random.default_rng(900)
        for label, kind in enumerate(['real','fake']):
            rows = []
            for i in range(12):
                path = f'pool_{kind}_{i}.png'
                Image.fromarray(rng.integers(0,256,(24,24,3),dtype=np.uint8)).save(a.target_root/path)
                rows.append(dict(crop_path=path,video_id=f'pool_{kind}_{i//2}',roi_status='detected' if i else 'fallback_letterbox',
                                 split='non_test_pool',class_index=str(label),method=kind))
            pd.DataFrame(rows).to_csv(a.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv',index=False)

    def tearDown(self):
        for p in reversed(self.extra):
            p.stop()
        fixture.PreservationTests.tearDown(self)

    def test_pseudo_mask_gradient_and_ema(self):
        student = torch.zeros(3,2,requires_grad=True)
        teacher = torch.tensor([[8.,-8.],[0.,0.],[-8.,8.]],requires_grad=True)
        loss, stats, _ = mt.pseudo_loss(student,teacher,.9)
        self.assertAlmostEqual(loss.item(), 2*np.log(2)/3, places=6)
        self.assertEqual(stats.tolist(), [2,1,1,1])
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(student.grad[1],torch.zeros(2)))
        self.assertLess(student.grad[0,0],0)
        self.assertLess(student.grad[2,1],0)
        x = torch.zeros(2,2,requires_grad=True)
        zero, _, _ = mt.pseudo_loss(x,torch.zeros(2,2),.9)
        zero.backward()
        self.assertEqual(zero.item(),0.)
        self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))
        s, t = torch.nn.Linear(2,2), torch.nn.Linear(2,2)
        with torch.no_grad():
            s.weight.fill_(2); t.weight.fill_(0)
        mt.update_teacher(t,s,.75)
        self.assertTrue(torch.equal(t.weight,torch.full_like(t.weight,.5)))
        self.assertFalse(t.training)

    def test_target_plan_ignores_labels_and_views_match_pixels(self):
        a = self.args
        _, _, target, cfg, _, _ = mt.adaptation.dataset_inputs(a)
        with_labels = target.assign(class_index=0,method='all_real')
        other_labels = target.assign(class_index=1,method='all_fake')
        plan = mt.target_plan(with_labels,200,42)
        self.assertEqual(plan,mt.target_plan(other_labels,200,42))
        self.assertTrue(all('class_index' not in row and 'method' not in row for row in plan))
        self.assertEqual(set(p['condition'] for p in plan),set(mt.CONDITIONS))
        # Exact legacy composition, including native mask, before resize/flip.
        identity = {'test':True}
        a._target_cache = mt.common.pixel_cache.BilateralCache(a.target_train_cache/'one',cfg['bilateral'],identity)
        a._target_probe_cache = mt.common.pixel_cache.BilateralCache(a.target_train_cache/'two',cfg['bilateral'],identity)
        row = target[target.roi_status.eq('detected')].iloc[0].to_dict()
        plan = [{**row,'condition':'smooth_strong','flip':False}]
        ds = mt.TargetViews(plan,a,fixture.build()[1],cfg,True)
        observed, changed = ds[0]
        with Image.open(a.target_root/row['crop_path']) as opened:
            original = opened.convert('RGB')
        expected = mt.adaptation.smooth(original,cfg['conditions']['smooth_strong'],cfg['bilateral'])
        extra = mt.adaptation.smooth(expected,cfg['conditions']['smooth_weak'],cfg['bilateral'])
        transform = fixture.build()[1]
        self.assertTrue(torch.equal(observed,transform(expected)))
        self.assertTrue(torch.equal(changed,transform(extra.transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
        fallback = target[target.roi_status.ne('detected')].iloc[0].to_dict()
        dataset = mt.TargetViews([{**fallback,'condition':'smooth_strong','flip':False}],a,transform,cfg,True)
        observed, changed = dataset[0]
        self.assertTrue(torch.equal(observed.flip(-1),changed))

    def test_train_resume_evaluate_and_protocols(self):
        a = self.args
        # Produce the exact same baseline and tiny 3-strategy experiment without external downloads.
        mt.train(a)
        hashes = {}
        for strategy in a.strategies:
            run = a.runs_root/mt.run_name('clip_b16',strategy,42)
            hashes[strategy] = mt.base.common.digest(run/'best.pt')
            last = torch.load(run/'last.pt',weights_only=True)
            original = torch.load(a.baseline_runs_root/'clip_b16_ffpp_c23_roi16_baseline_v1_seed42/best.pt',weights_only=True)
            self.assertTrue(any(not torch.equal(last['model_state_dict'][k],v) for k,v in original['model_state_dict'].items()))
            if strategy != 'augmentation':
                self.assertEqual(last['history'][0]['target_exposures'],16)
                self.assertEqual(sum(last['history'][0]['teacher_predicted_counts'].values()),16)
                self.assertIsNotNone(last['teacher_state_dict'])
        histories = [json.loads((a.runs_root/mt.run_name('clip_b16',s,42)/'history.json').read_text()) for s in a.strategies]
        self.assertEqual([h['source_paths_sha256'] for h in histories[0]],[h['source_paths_sha256'] for h in histories[1]])
        self.assertEqual([h['target_plan_sha256'] for h in histories[1]],[h['target_plan_sha256'] for h in histories[2]])
        frozen = pd.read_csv(a.manifest_root/'target_unlabelled.csv')
        self.assertEqual(list(frozen),['crop_path','video_id','roi_status'])
        self.assertFalse(set(frozen.video_id)&set(pd.read_csv(self.test).video_id))
        mt.train(a)
        for strategy in a.strategies:
            self.assertEqual(hashes[strategy],mt.base.common.digest(a.runs_root/mt.run_name('clip_b16',strategy,42)/'best.pt'))
        mt.evaluate(a)
        comparison = pd.read_csv(a.output_root/'comparison.csv')
        self.assertEqual(len(comparison),16)
        self.assertTrue(np.allclose(comparison.balanced_error_rate,.5*(comparison.real_fpr+comparison.fake_fnr)))
        before = {p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        mt.evaluate(a)
        self.assertEqual(before,{p.name:p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        # Reuse saved baseline raw-margin results without a forward pass.
        a.reference_evaluations = a.output_root
        a.output_root = self.root/'results_reused'
        mt.evaluate(a)
        reference = json.loads((a.output_root/'clip_b16_baseline_normal.json').read_text())
        self.assertIn('reused_predictions',reference)
        # An interrupted completed epoch restores student, EMA, optimizer and sampling exactly.
        strategy = 'mean_teacher_smoothing'
        golden = torch.load(a.runs_root/mt.run_name('clip_b16',strategy,42)/'last.pt',weights_only=True)
        a.runs_root = self.root/'resumed_runs'
        a.strategies = [strategy]
        save = mt.base.common.save
        def interrupt(path,state):
            save(path,state)
            if path.name == 'last.pt' and state['epoch'] == 1:
                raise RuntimeError('intentional interruption')
        with patch.object(mt.base.common,'save',side_effect=interrupt), self.assertRaisesRegex(RuntimeError,'intentional'):
            mt.train(a)
        mt.train(a)
        resumed = torch.load(a.runs_root/mt.run_name('clip_b16',strategy,42)/'last.pt',weights_only=True)
        self.assertEqual(golden['history'],resumed['history'])
        for key in ['model_state_dict','teacher_state_dict']:
            self.assertTrue(all(torch.equal(v,resumed[key][k]) for k,v in golden[key].items()))
        baseline = a.baseline_runs_root/'clip_b16_ffpp_c23_roi16_baseline_v1_seed42/best.pt'
        self.assertEqual(mt.base.common.digest(baseline),self.initial_hashes['clip_b16'])

    def test_notebook_cells_and_raw_auc(self):
        from IPython.core.interactiveshell import InteractiveShell
        path = mt.ROOT/'notebooks/train_test_ffpp_c23_smoothing_mean_teacher_seed42.ipynb'
        notebook = json.loads(path.read_text())
        codecells = [c for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(len(codecells),7)
        for cell in codecells:
            source = ''.join(cell['source'])
            self.assertFalse(source.lstrip().startswith('python\n'))
            ast.parse(InteractiveShell.instance().input_transformer_manager.transform_cell(source))
        frame = pd.DataFrame({'class_index':[0,0,1,1]})
        metrics = mt.raw_metrics(frame,np.ones(4),np.array([20.,21.,22.,23.]))
        self.assertEqual(metrics['auc'],1.)
        self.assertEqual(metrics['auc_probability'],.5)


if __name__ == '__main__':
    unittest.main()
