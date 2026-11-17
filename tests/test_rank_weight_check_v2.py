"""Rank-weight check: run/config/sample guards, fusion with B on the input, reproduction checks and the headline."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import rank_weight_check_v2 as rw

MODEL = 'xception'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def history(epochs, offset=0):
    return [dict(epoch=e, source_ce=.1+offset, source_rank=.02, source_paths_sha256=f'p{e}', source_batches_sha256=f'b{e}')
            for e in range(1, epochs+1)]


def config(filt, weight, strategy=rw.RANK, **extra):
    return dict(dict(protocol='x', model=MODEL, train_filter=filt, strategy=strategy, seed=42, baseline_seed=42,
                     initial_checkpoint_sha256='base', ranking_weight=weight, fc_weight=0., epochs=10, patience=3,
                     code_sha256={'a': str(weight)}, versions={}, torch='t'), **extra)


class Fixture:
    """Synthetic thesis outputs (pair, rank .25, pair+NSF smooth_all) and one new alpha=1 run with its predictions."""
    def __init__(self, root):
        self.args = argparse.Namespace(model=MODEL, alphas=[1.], runs_base=root/'new_runs', original_runs=root/'runs',
                                       original_eval=root/'eval', pair_nsf_root=root/'pair_nsf', smoothed_b_root=root/'sb',
                                       output_root=root/'out')
        a, rng, n = self.args, np.random.default_rng(3), 120
        self.y = np.r_[np.zeros(45, int), np.ones(75, int)]
        self.paths = [f'v{i}/f.png' for i in range(n)]
        info = dict(settings=rw.nsf.SETTINGS, pickle_sha256='p', val_mean=.1, val_std=.5, val_paths_sha256='v', baseline_sha256='base')
        write(a.original_runs/'nonskin_probe'/f'{MODEL}_face_inner_seed42.json', info)
        write(a.smoothed_b_root/'scores/identity.json', dict(identity=dict(model=MODEL, scorer_pickle_sha256='p')))
        b_raw = {'normal': rng.normal(size=n)+self.y}
        pd.DataFrame(dict(crop_path=self.paths, clean=b_raw['normal'])).to_csv(a.smoothed_b_root/'scores/cdf_test_normal.csv', index=False)
        for f in rw.FILTERS:
            b_raw[f] = b_raw['normal']+rng.normal(scale=.2, size=n)
            pd.DataFrame(dict(crop_path=self.paths, smooth_pix=b_raw[f]+9, smooth_all=b_raw[f])).to_csv(
                a.smoothed_b_root/f'scores/cdf_test_{f}.csv', index=False)
        bz = {v: (s-info['val_mean'])/info['val_std'] for v, s in b_raw.items()}
        summary, sb_rows = [], []
        for alpha in [None, rw.REFERENCE_ALPHA, 1.]:
            label = rw.PAIR if alpha is None else rw.RANK
            for f in rw.FILTERS:
                sha = f'{label}{alpha}{f}'
                root_runs = a.original_runs if alpha in (None, rw.REFERENCE_ALPHA) else a.runs_base/rw.tag(alpha)
                run = rw.run_dir(root_runs, MODEL, f, label)
                write(run/'history.json', history(4 if alpha == 1. else 6, offset=0 if alpha is None else .01))
                write(run/'completed.json', dict(identity='i', selected_epoch=2))
                write(run/'config.json', config(f, alpha or 0., label))
                cal = dict(identity=dict(source_checkpoint_sha256=sha), rank_val_mean=.3, rank_val_std=1.7, weight=.2)
                write(rw.calibration_path(a, alpha, f), cal)
                base = rng.normal(size=n)+1.2*self.y
                m = {}
                for v in rw.VIEWS:
                    m[v] = base if v == 'normal' else base+rng.normal(scale=.4, size=n)+.6
                    path = rw.prediction_path(a, alpha, f, v)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(dict(crop_path=self.paths, video_id=self.paths, class_index=self.y, raw_margin=m[v],
                                      checkpoint_sha256=sha)).to_csv(path, index=False)
                z = {v: (m[v]-cal['rank_val_mean'])/cal['rank_val_std'] for v in rw.VIEWS}
                for ev in rw.FILTERS:
                    if alpha != 1.:
                        r = rw.rc.metrics(self.y, m['normal'], m[ev])
                        summary.append(dict(method=label, train_filter=f, filter=ev, fp_cross_auc=r['fp_cross_auc'], fn_cross_auc=r['fn_cross_auc']))
                    if alpha is None:
                        r = rw.rc.metrics(self.y, .8*z['normal']+.2*bz['normal'], .8*z[ev]+.2*bz[ev])
                        sb_rows.append(dict(variant='smooth_all', train_filter=f, filter=ev, fp_cross_auc=r['fp_cross_auc'], fn_cross_auc=r['fn_cross_auc']))
        pd.DataFrame(summary).to_csv(a.original_eval/'seed42/cdf_test/summary.csv', index=False)
        (a.smoothed_b_root/'seed42').mkdir(parents=True)
        pd.DataFrame(sb_rows).to_csv(a.smoothed_b_root/'seed42/summary.csv', index=False)


class AnalyzeTests(unittest.TestCase):
    def test_analyze_reproduces_references_and_reports_every_alpha(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(Path(temp))
            head = rw.analyze(fx.args)
            self.assertEqual(head.name.tolist(), ['동시 증강', '교차 순위 α=0.25', '교차 순위 α=1',
                                                  '동시 증강 + 눈·입 (제안 방법)', '교차 순위 α=0.25 + 눈·입', '교차 순위 α=1 + 눈·입'])
            h = head.set_index('method')
            self.assertEqual(int(h.loc[rw.PAIR, 'cells_over_pair']), 0)
            self.assertEqual(int(h.loc[rw.PAIR+'+nsf', 'cells_over_proposed']), 0)
            self.assertEqual(h.loc[rw.method(1.), 'selected_epochs'], '2/2/2/2')
            self.assertAlmostEqual(float(h.loc[rw.method(1.), 'weighted_rank_over_ce']), 1*.02/.11)
            self.assertAlmostEqual(float(h.loc[rw.method(.25), 'weighted_rank_over_ce']), .25*.02/.11)
            self.assertTrue(np.isnan(h.loc[rw.PAIR, 'weighted_rank_over_ce']))
            rows = pd.read_csv(fx.args.output_root/'rank_weight_rows.csv')
            self.assertEqual(len(rows), 3*2*16)
            self.assertEqual(int(rows.held_out.sum()), 3*2*12)

    def test_reproduction_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(Path(temp))
            path = fx.args.smoothed_b_root/'seed42/summary.csv'
            pd.read_csv(path).assign(fp_cross_auc=lambda t: t.fp_cross_auc+.001).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                rw.analyze(fx.args)
            fx = Fixture(Path(temp)/'b')
            path = fx.args.original_eval/'seed42/cdf_test/summary.csv'
            pd.read_csv(path).assign(fn_cross_auc=lambda t: t.fn_cross_auc-.001).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                rw.analyze(fx.args)

    def test_prediction_from_another_checkpoint_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(Path(temp))
            path = rw.prediction_path(fx.args, 1., 'fast_global', 'domain_transform')
            pd.read_csv(path).assign(checkpoint_sha256='other').to_csv(path, index=False)
            with self.assertRaises(ValueError):
                rw.analyze(fx.args)


class SelectionTests(unittest.TestCase):
    def tables(self, root, all16):
        """all16: {model: {alpha: score}} -> dev_headline.csv in each backbone's own output folder."""
        for m, scores in all16.items():
            d = root/f'exp_{m}'/'dev'; d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(model=m, alpha=a, all16=s) for a, s in scores.items()]).to_csv(d/'dev_headline.csv', index=False)
        return argparse.Namespace(model='clip_b16', alphas=[1., 2., 4.], output_root=root/'exp_clip_b16')

    def test_waits_until_all_three_backbones_have_dev_results(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.tables(Path(temp), {'clip_b16': {.25: .6, 1.: .7, 2.: .7, 4.: .7}})
            self.assertIsNone(rw.select(args))
            with self.assertRaises(FileNotFoundError):
                rw.require_decision(args)

    def test_keeps_thesis_alpha_unless_clearly_better(self):
        with tempfile.TemporaryDirectory() as temp:
            base = {.25: .600, 1.: .603, 2.: .604, 4.: .598}           # best +.004 < .005 -> keep .25
            args = self.tables(Path(temp), {m: base for m in rw.BACKBONES})
            self.assertEqual(rw.select(args)['chosen_alpha'], .25)
            self.assertEqual(rw.require_decision(args)['best_alpha'], 2.)
        with tempfile.TemporaryDirectory() as temp:
            args = self.tables(Path(temp), {'xception': {.25: .50, 1.: .52, 2.: .51, 4.: .50},
                                            'clip_b16': {.25: .70, 1.: .70, 2.: .71, 4.: .69},
                                            'dinov2_b14': {.25: .60, 1.: .60, 2.: .61, 4.: .60}})
            decision = rw.select(args)                                  # mean: .25 .600, 1 .6067, 2 .61 -> 2 (+.01)
            self.assertEqual(decision['chosen_alpha'], 2.)
            self.assertAlmostEqual(decision['scores']['2'], .61)

    def test_decision_is_frozen_against_changed_dev_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.tables(root, {m: {.25: .6, 1.: .62, 2.: .6, 4.: .6} for m in rw.BACKBONES})
            rw.select(args)
            self.tables(root, {'xception': {.25: .6, 1.: .5, 2.: .6, 4.: .6}})
            with self.assertRaises(ValueError):
                rw.require_decision(args)                                # test refused: dev changed after the choice
            with self.assertRaises(ValueError):
                rw.select(args)                                          # and the earlier decision is not overwritten

    def test_alpha_sets_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            tabs = {m: {.25: .6, 1.: .6, 2.: .6, 4.: .6} for m in rw.BACKBONES}
            tabs['xception'] = {.25: .6, 1.: .6}
            with self.assertRaises(ValueError):
                rw.select(self.tables(Path(temp), tabs))


class RunGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.temp.name))
        for alpha in [rw.REFERENCE_ALPHA, 1.]:
            for f in rw.FILTERS:
                (rw.run_dir(rw.runs_root(self.fx.args, alpha), MODEL, f)/'best.pt').write_bytes(b'x')

    def tearDown(self):
        self.temp.cleanup()

    def test_new_run_passes_when_only_the_weight_differs(self):
        run = rw.check_run(self.fx.args, 1., 'bilateral', 'base')
        self.assertEqual(run.parent.name, 'alpha1')

    def test_other_setting_or_samples_differ(self):
        run = rw.run_dir(self.fx.args.runs_base/'alpha1', MODEL, 'bilateral')
        write(run/'config.json', config('bilateral', 1., patience=4))
        with self.assertRaises(ValueError):
            rw.check_run(self.fx.args, 1., 'bilateral', 'base')
        write(run/'config.json', config('bilateral', 1.))
        h = history(4); h[2]['source_batches_sha256'] = 'changed'
        write(run/'history.json', h)
        with self.assertRaises(ValueError):
            rw.check_run(self.fx.args, 1., 'bilateral', 'base')

    def test_wrong_weight_or_baseline(self):
        with self.assertRaises(FileNotFoundError):
            rw.check_run(self.fx.args, 2., 'bilateral', 'base')      # alpha2 not trained yet
        run = rw.run_dir(self.fx.args.runs_base/'alpha1', MODEL, 'bilateral')
        write(run/'config.json', config('bilateral', 2.))           # folder says alpha1, config says 2
        with self.assertRaises(ValueError):
            rw.check_run(self.fx.args, 1., 'bilateral', 'base')
        write(run/'config.json', config('bilateral', 1.))
        with self.assertRaises(ValueError):
            rw.check_run(self.fx.args, 1., 'bilateral', 'other_baseline')

    def test_parse_args_rejects_thesis_weight_and_shared_folders(self):
        common = ['analyze', '--model', MODEL, '--runs-base', 'n', '--original-runs', 'r', '--original-eval', 'e',
                  '--pair-nsf-root', 'p', '--smoothed-b-root', 's', '--output-root', 'o']
        self.assertEqual(rw.parse_args(common).alphas, [1., 2., 4.])
        with self.assertRaises(SystemExit):
            rw.parse_args(common+['--alphas', '.25', '1'])
        with self.assertRaises(SystemExit):
            rw.parse_args([*common[:-1], 'e'])


if __name__ == '__main__':
    unittest.main()
