"""Check score reuse, pairing/provenance checks and seen/unseen aggregation without neural inference."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import evaluate_pair_nsf_v2 as p


class PairNsfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(model='xception', seeds=[42,43,44], train_filters=p.FILTERS,
            evaluation_root=self.root/'reference', output_root=self.root/'output')
        self.frame = pd.DataFrame(dict(crop_path=['a','b','c','d'], video_id=['a','b','c','d'],
            class_index=[0,0,1,1], raw_margin=[-2.,-1.,1.,2.], checkpoint_sha256=['pairhash']*4))
        self.scores = pd.DataFrame(dict(crop_path=self.frame.crop_path, score=[-.2,.1,.3,.7]))
        self.done = dict(identity={'source_checkpoint_sha256':'pairhash'}, rank_val_mean=.3,rank_val_std=2.,
                         scorer_val_mean=.1,scorer_val_std=.4,weight=.2)
        source = self.args.evaluation_root/'seed42/cdf_test';source.mkdir(parents=True)
        for filt in p.FILTERS:
            for ev in p.FILTERS:
                for i,c in enumerate(p.CONDITIONS):
                    stem = 'normal' if c=='normal' else ev+'_'+c
                    f = self.frame.copy();f.raw_margin+=.15*i
                    f.to_csv(source/f'xception_aug_pair@{filt}_{stem}.csv',index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_saved_predictions_only_and_shared_auxiliary(self):
        with patch.object(p.nsf,'load',side_effect=AssertionError('Unexpected model inference')):
            rows = p.fuse_one(self.args,42,'bilateral',self.done,self.scores)
        self.assertEqual(len(rows),4)
        dest=self.args.output_root/'seed42/cdf_test'
        normal=pd.read_csv(dest/'xception_aug_pair_nsf@bilateral_normal.csv')
        strong=pd.read_csv(dest/'xception_aug_pair_nsf@bilateral_fast_global_smooth_strong.csv')
        expected=.8*(self.frame.raw_margin-.3)/2+.2*(self.scores.score-.1)/.4
        np.testing.assert_allclose(normal.raw_margin,expected)
        # Smoothing changes only the pair score; cached eye/mouth scores remain unchanged.
        np.testing.assert_allclose(strong.raw_margin-normal.raw_margin,.8*.45/2)

    def test_rejects_misordered_or_stale_predictions(self):
        path=self.args.evaluation_root/'seed42/cdf_test/xception_aug_pair@bilateral_weighted_median_smooth_strong.csv'
        self.frame.iloc[::-1].to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'order/labels'):
            p.fuse_one(self.args,42,'bilateral',self.done,self.scores)
        f=self.frame.copy();f['checkpoint_sha256']='wrong';f.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'Stale prediction'):
            p.fuse_one(self.args,42,'bilateral',self.done,self.scores)

    def test_report_and_ppt_matrix_aggregation(self):
        for seed in self.args.seeds:
            old=[];new=[]
            methods=['baseline','aug_single','aug_pair','aug_pair_rank','aug_pair_rank_fc100_ramp2',p.nsf.LABEL,p.LABEL]
            for method in methods:
                trains=['-'] if method=='baseline' else p.FILTERS
                for train in trains:
                    for ev in p.FILTERS:
                        gain=.04 if method==p.LABEL else .02 if method==p.nsf.LABEL else 0.
                        same=train==ev
                        cross=(.8 if same else .5)+gain
                        row=dict(model='xception',strategy=method+'@'+train,method=method,train_filter=train,filter=ev,
                            normal_auc=.8+gain,fp_cross_auc=cross,fn_cross_auc=cross+.01,max_gap=.1,
                            real_shift=.1,fake_shift=.1)
                        row.update({'fp_flip@normal5':.03,'fn_flip@normal5':.02})
                        (new if method==p.LABEL else old).append(row)
            for root,rows in [(self.args.evaluation_root,old),(self.args.output_root,new)]:
                folder=root/f'seed{seed}'/'cdf_test';folder.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
        with patch.object(p,'export_plot'):
            head,means=p.report(self.args)
        r=head.set_index('method').loc[p.LABEL]
        self.assertAlmostEqual(r.diag_worse_mean,.84)
        self.assertAlmostEqual(r.heldout_worse_mean,.54)
        self.assertAlmostEqual(r.all_worse_mean,.615)
        gain=pd.read_csv(self.args.output_root/'ppt_tables/fusion_comparisons.csv').iloc[-1]
        self.assertAlmostEqual(gain.all_cross_gain,.02)
        self.assertEqual(gain.cells_improved,16)
        self.assertTrue((self.args.output_root/'ppt_tables/slide25_same_filter.csv').is_file())

    def test_calibration_reuses_original_validation_statistics(self):
        a=self.args
        a.runs_root=self.root/'runs';a.source_root=self.root/'source';a.device='cpu';a.work_dir=self.root/'work'
        name=p.nsf.diagonal.run_name('xception','bilateral','aug_pair',42)
        run=a.runs_root/name;run.mkdir(parents=True)
        (run/'best.pt').write_bytes(b'fixture checkpoint')
        pair_sha=p.digest(run/'best.pt')
        cfg=dict(strategy='aug_pair',model='xception',train_filter='bilateral',seed=42,initial_checkpoint_sha256='baselinehash')
        (run/'config.json').write_text(json.dumps(cfg));(run/'completed.json').write_text(json.dumps({'identity':'pairidentity'}))
        info=dict(baseline_sha256='baselinehash',pickle_sha256='probesha',val_paths_sha256='valsha',val_mean=.1,val_std=.4)
        raw=np.array([-2.,-1.,1.,2.])
        with patch.object(p.nsf.diagonal,'completed_checkpoint',return_value=({},pair_sha,3)), \
             patch.object(p.nsf,'load',return_value=(None,None)), \
             patch.object(p.nsf.fusion,'Inputs',return_value=None), \
             patch.object(p.nsf.diagonal.engine,'infer',return_value=(None,raw)):
            done=p.calibration(a,42,'bilateral',self.frame,info)
        self.assertEqual(done['selected_epoch'],3)
        self.assertAlmostEqual(done['rank_val_mean'],raw.mean())
        self.assertAlmostEqual(done['rank_val_std'],raw.std())
        with patch.object(p.nsf,'load',side_effect=AssertionError('Repeated validation inference')):
            self.assertEqual(p.calibration(a,42,'bilateral',self.frame,info),done)
        (run/'best.pt').write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError,'provenance changed'):
            p.calibration(a,42,'bilateral',self.frame,info)


if __name__=='__main__':unittest.main()
