"""Paired error identities, diagnostic GT isolation, final EMA and notebook checks."""
import argparse
import ast
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tests'))
import analyze_smoothing_uda_failures as errors
import diagnose_smoothing_pseudo_labels as pseudo
import test_smoothing_error_tradeoff as legacy_tests


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_repair_harm_and_smoothing_pair_identities(self):
        y = [0]*4+[1]*4
        def frame(scores):
            return pd.DataFrame(dict(crop_path=[f'{i}.png' for i in range(8)],
                video_id=[str(i) for i in range(8)], class_index=list(map(str,y)),
                roi_status=['detected']*8, raw_margin=scores, probability_fake=[.5]*8))
        baseline = frame([-1,1,-1,1,1,-1,1,-1])
        candidate = frame([-1,-1,1,1,1,1,-1,-1])
        normal = frame([-1,-1,-1,1,1,1,1,-1])
        paired = errors.paired_rows('clip_b16','mean_teacher','smooth_strong',baseline,candidate,normal)
        self.assertEqual(paired.repaired.sum(),2)
        self.assertEqual(paired.new_error.sum(),2)
        self.assertEqual(paired.baseline_smoothing_induced_error.sum(),2)
        self.assertEqual((paired.repaired & paired.baseline_smoothing_induced_error).sum(),2)
        for label in ['0','1']:
            row = errors.transition_summary(paired[paired.class_index.eq(label)])
            self.assertEqual(row['repaired_errors'],1)
            self.assertEqual(row['new_errors'],1)
            self.assertEqual(row['error_reduction_pp'],0)

    def test_saved_discovery_alignment_conflicts_and_preservation(self):
        fixture = legacy_tests.TradeoffTests()
        fixture.root, fixture.y = self.root, np.array([0]*4+[1]*4)
        for strategy in ['baseline','mean_teacher','stable_agreement']:
            for condition in errors.CONDITIONS:
                fixture.write_record(self.root/'saved', strategy, condition,
                    [-1,.6,.2,-.4,.8,-.6,.5,.1], reverse=strategy == 'stable_agreement')
        originals = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root/'saved').iterdir()}
        args = argparse.Namespace(evaluation_roots=[self.root/'saved'],output_root=self.root/'output',
            models=['clip_b16'],strategies=['mean_teacher','stable_agreement'],expected_images=8)
        args.output_root.mkdir()
        errors.analyze(args)
        table = pd.read_csv(args.output_root/'error_transitions.csv')
        self.assertTrue(table.new_errors.eq(0).all())
        self.assertEqual(len(table),16)
        self.assertEqual(len(pd.read_csv(args.output_root/'paired_image_errors.csv')),64)
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path,sha in originals.items()))
        with self.assertRaises(FileExistsError): errors.analyze(args)
        path = self.root/'saved/clip_b16_stable_agreement_smooth_strong.csv'
        frame = pd.read_csv(path,dtype={'video_id':str})
        frame.loc[0,'video_id'] = 'wrong-video'
        frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'Membership'):
            errors.discover(args)

    def inputs(self):
        target = self.root/'target'
        (target/'manifests').mkdir(parents=True)
        frame = pd.DataFrame(dict(crop_path=[f'frames/v{i//2:03}/{i}.png' for i in range(8)],
            video_id=[f'v{i//2:03}' for i in range(8)], roi_status=['detected']*7+['fallback']))
        from PIL import Image
        for i,path in enumerate(frame.crop_path):
            dest=target/path; dest.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(12,16),(80+i,100,120)).save(dest)
        labels = ['0']*4+['1']*4
        paths=[]
        for kind, subset in [('real',slice(0,4)),('fake',slice(4,8))]:
            p=target/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv'
            part=frame.iloc[subset].copy(); part['class_index']=labels[subset]; part['split']='non_test_pool'
            part.to_csv(p,index=False); paths.append(p)
        folder=self.root/'hard_manifest/xception'; folder.mkdir(parents=True)
        frame.to_csv(folder/'target_unlabelled.csv',index=False)
        manifest_hash=hashlib.sha256(json.dumps(frame.crop_path.tolist(),sort_keys=True).encode()).hexdigest()
        frozen=dict(protocol=pseudo.SNAPSHOTS['mean_teacher'],seed=42,excluded_video_ids=['test-video'],data={
            'target_paths_sha256':manifest_hash,'non_test_manifest_sha256':[errors.saved.digest(p) for p in paths],
            'test_manifest_sha256':'official-test-fixture'})
        errors.write_json(folder/'protocol.json',frozen)
        analysis=self.root/'errors'; analysis.mkdir()
        pd.DataFrame(dict(crop_path=['test.png'],video_id=['test-video'],class_index=[0],roi_status=['detected'])).to_csv(analysis/'test_membership.csv',index=False)
        errors.write_json(analysis/'analysis_protocol.json',dict(
            test_membership_sha256=errors.saved.digest(analysis/'test_membership.csv'),
            evaluation_signature={'manifest_sha256':'official-test-fixture'}))
        args=argparse.Namespace(hard_manifest_root=self.root/'hard_manifest',analysis_root=analysis,
            target_root=target,output_root=self.root/'pseudo',seed=42,models=['xception'],
            strategies=['mean_teacher','stable_agreement'],hard_runs_root=self.root/'runs/hard',
            stable_runs_root=self.root/'runs/stable',target_train_cache=self.root/'cache',batch_size=3,workers=0,
            device='cpu',protocol=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json',
            drive_data=self.root/'drive',archive_cache=self.root/'archive_cache')
        return args,frame,frozen,paths

    def test_target_labels_are_diagnostic_only_and_leakage_is_rejected(self):
        args,frame,frozen,_=self.inputs()
        unlabelled,labels,_,_=pseudo.labelled_target(args)
        self.assertEqual(list(unlabelled),['crop_path','video_id','roi_status'])
        self.assertEqual(labels.tolist(),[0]*4+[1]*4)
        frozen['excluded_video_ids'].append(frame.video_id.iloc[0])
        errors.write_json(self.root/'hard_manifest/xception/protocol.json',frozen)
        with self.assertRaisesRegex(ValueError,'overlaps'):
            pseudo.labelled_target(args)

    def test_prepare_extracts_only_frozen_members_and_never_test_data(self):
        args,frame,_,paths=self.inputs()
        args.drive_data.mkdir()
        sizes={}
        for kind,path in zip(['real','fake'],paths):
            archive=args.drive_data/f'celebdf_v2_roi16_non_test_{kind}_v1.tar'
            with tarfile.open(archive,'w') as tf:
                tf.add(path,arcname=pseudo.PREFIX+'manifests/'+path.name)
                members = frame.crop_path.iloc[:4] if kind == 'real' else frame.crop_path.iloc[4:]
                for relative in members:
                    tf.add(args.target_root/relative,arcname=pseudo.PREFIX+relative)
                other=tarfile.TarInfo(pseudo.PREFIX+'frames/unrequested.png'); other.size=4
                tf.addfile(other,io.BytesIO(b'none'))
            sizes[kind]=archive.stat().st_size
        args.target_root=self.root/'prepared'
        with patch.dict(pseudo.ARCHIVES,sizes):
            pseudo.prepare(args)
        self.assertEqual(len(list((args.target_root/'frames').rglob('*.png'))),8)
        self.assertFalse((args.target_root/'frames/unrequested.png').exists())
        self.assertFalse(list(args.archive_cache.glob('*.tar')))
        # Repeated prepare uses existing crops and does not reopen TARs.
        with patch.object(tarfile,'open',side_effect=AssertionError('No TAR read needed')):
            pseudo.prepare(args)

    def test_final_ema_inference_reuse_and_quality(self):
        import torch
        import ffpp_c23_smoothing_mean_teacher as engine
        import ffpp_c23_smoothing_stable_uda as stable
        args,frame,frozen,_=self.inputs()
        def build(name,pretrained=False):
            model=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,2))
            def transform(image):
                return torch.from_numpy(np.array(image.resize((8,8)),dtype=np.float32)).permute(2,0,1)/255
            return model,transform
        template,_=build('xception')
        with torch.no_grad():
            template[-1].weight.zero_(); template[-1].bias.copy_(torch.tensor([-2.,2.]))
        cfg0=json.loads(args.protocol.read_text())
        for strategy in args.strategies:
            protocol=pseudo.SNAPSHOTS[strategy]
            root=args.hard_runs_root if strategy == 'mean_teacher' else args.stable_runs_root
            run=root/f'xception_{strategy}_{protocol}_seed42'; run.mkdir(parents=True)
            cfg=dict(model='xception',strategy=strategy,protocol=protocol,seed=42,data=frozen['data'],
                preprocessing=engine.base.PREPROCESS['xception'],code_sha256=engine.code_hashes(),
                smoothing=cfg0,opencv=engine.evaluation.cv2.__version__,pseudo_threshold=.9)
            if strategy == 'stable_agreement':
                cfg['extension_code_sha256']=errors.saved.digest(Path(stable.__file__))
            identity=engine.evaluation.key(cfg)
            weights=template.state_dict()
            if strategy == 'stable_agreement': weights={'model.'+k:v for k,v in weights.items()}
            torch.save(dict(config=cfg,identity=identity,epoch=3,teacher_state_dict=weights),run/'last.pt')
            errors.write_json(run/'config.json',cfg)
            errors.write_json(run/'completed.json',dict(identity=identity,epochs_completed=3,selected_epoch=1))
        with patch.object(engine.base,'build_model',side_effect=build):
            pseudo.evaluate(args)
        quality=pd.read_csv(args.output_root/'pseudo_quality.csv')
        selected=quality[quality.grouping.eq('all')&quality.rule.eq('protocol_selected')]
        self.assertEqual(len(selected),8)
        self.assertTrue(selected.selected.eq(8).all())
        self.assertTrue(selected.selected_accuracy.eq(.5).all())
        self.assertTrue(selected.selected_wrong_real_as_fake.eq(4).all())
        saved=pd.read_csv(args.output_root/'xception_mean_teacher.csv',dtype={'video_id':str})
        self.assertEqual(len(saved),32)
        self.assertEqual(saved[saved.roi_status.eq('fallback')].pseudo_label.nunique(),1)
        self.assertAlmostEqual(saved[saved.condition.eq('normal')].video_uniform_weight.sum(),1)
        with patch.object(engine.base,'build_model',side_effect=AssertionError('Reuse must not run models')):
            pseudo.evaluate(args)
        # A changed snapshot cannot silently reuse a previous diagnostic.
        run=args.hard_runs_root/'xception_mean_teacher_ffpp_c23_smoothing_mean_teacher_v1_seed42'
        state=torch.load(run/'last.pt',weights_only=True)
        state['teacher_state_dict']['2.bias']=torch.tensor([2.,-2.])
        torch.save(state,run/'last.pt')
        with self.assertRaisesRegex(ValueError,'Saved snapshot diagnostic differs'):
            pseudo.evaluate(args)

    def test_notebook_matches_md_and_parses_without_python_fence_tokens(self):
        from IPython.core.inputtransformer2 import TransformerManager
        stem='analyze_smoothing_uda_failures_seed42'
        md=(ROOT/f'docs/colab/{stem}.md').read_text()
        blocks=re.findall(r'```python\n(.*?)\n```',md,re.S)
        book=json.loads((ROOT/f'notebooks/{stem}.ipynb').read_text())
        cells=[''.join(c['source']).rstrip() for c in book['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),8)
        self.assertEqual(cells,blocks)
        for cell in cells:
            self.assertFalse(cell.startswith('python\n'))
            ast.parse(TransformerManager().transform_cell(cell))
        self.assertIn('!python -u',cells[6])
        self.assertNotIn(' train ', '\n'.join(cells))


if __name__ == '__main__':
    unittest.main()
