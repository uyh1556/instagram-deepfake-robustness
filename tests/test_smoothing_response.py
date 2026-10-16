"""CPU execution checks for the matched smoothing-response experiment."""
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
import ffpp_c23_smoothing_response as response
import summarize_smoothing_response as summary
import test_smoothing_preservation as fixture


class ResponseTests(unittest.TestCase):
    def setUp(self):
        fixture.PreservationTests.setUp(self)
        self.args.models = ['clip_b16']
        self.args.strategies = list(response.STRATEGIES)
        self.args.extra_strength = .25

    def tearDown(self):
        fixture.PreservationTests.tearDown(self)

    def test_response_mask_initialization_and_native_fallback(self):
        torch.manual_seed(42)
        control = response.ResponseClassifier(4, 8, False)
        torch.manual_seed(42)
        candidate = response.ResponseClassifier(4, 8, True)
        self.assertEqual(sum(p.numel() for p in control.parameters()), sum(p.numel() for p in candidate.parameters()))
        for k, v in control.state_dict().items():
            self.assertTrue(torch.equal(v, candidate.state_dict()[k]))
        f, m = torch.randn(6, 2, 4), torch.randn(6, 2)
        self.assertTrue(torch.equal(control(f, m), m[:, 0]))
        self.assertTrue(torch.equal(candidate(f, m), m[:, 0]))
        with torch.no_grad():
            for head in [control, candidate]:
                head.net[0].weight.zero_(); head.net[0].bias.zero_()
                head.net[0].weight[:, 4] = 1.
                head.net[-1].weight.fill_(.1)
        changed = f.clone(); changed[:, 1, 0] += 10.
        self.assertTrue(torch.equal(control(f, m), control(changed, m)))
        self.assertFalse(torch.equal(candidate(f, m), candidate(changed, m)))
        pool, _, _ = response.shared.source_inputs(self.args)
        plan = response.training_plan(pool, 42, self.cfg)
        self.assertTrue(plan.iloc[::2].condition.eq('normal').all())
        self.assertEqual(plan.iloc[::2].flip.tolist(), plan.iloc[1::2].flip.tolist())
        dataset = response.ProbeViews(plan, self.args.source_root, fixture.build()[1], self.cfg, .25)
        for i in plan.index[plan.roi_status.ne('detected')]:
            views, _ = dataset[i]
            self.assertTrue(torch.equal(views[0], views[1]))

    def test_train_evaluate_reuse_and_original_preserved(self):
        a = self.args
        response.train(a)
        self.assertFalse((a.manifest_root/'target_unlabelled.csv').exists())
        for strategy in a.strategies:
            run = a.runs_root/response.run_name('clip_b16', strategy, a.seed)
            selected = torch.load(run/'best.pt', weights_only=True)
            self.assertLessEqual(selected['best'], json.loads((run/'baseline_source_validation.json').read_text())['mean_ber'])
            self.assertFalse(selected['config']['target_training'])
        response.train(a)
        response.evaluate(a)
        paths = list(a.output_root.glob('*.csv'))
        before = {p.name: p.stat().st_mtime_ns for p in paths if p.name not in ['comparison.csv','condition_means.csv']}
        response.evaluate(a)
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in paths if p.name in before})
        table = pd.read_csv(a.output_root/'comparison.csv')
        self.assertTrue({'baseline','current_only','response'}.issubset(set(table.strategy)))
        self.assertTrue(np.allclose(table.balanced_error_rate, (table.real_fpr+table.fake_fnr)/2))
        # Match the actual legacy Aug producer: flat metrics without images/precision.
        a.reference_evaluations.mkdir()
        (a.reference_evaluations/'evaluation_protocol.json').write_bytes((a.output_root/'evaluation_protocol.json').read_bytes())
        for condition in self.cfg['conditions']:
            source = a.output_root/f'clip_b16_baseline_{condition}.json'
            report = json.loads(source.read_text())
            for field in ['images','precision','model','condition','strategy']:
                report.pop(field, None)
            dest = a.reference_evaluations/f'clip_b16_augmentation_{condition}.json'
            dest.write_text(json.dumps(report))
            dest.with_suffix('.csv').write_bytes(source.with_suffix('.csv').read_bytes())
        with patch.object(response,'features_for',side_effect=AssertionError('must not infer')):
            summary.summarize(a)
        merged = pd.read_csv(a.output_root/'comparison.csv')
        self.assertEqual(int(merged.strategy.eq('augmentation').sum()),4)
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in paths if p.name in before})
        for path in a.output_root.glob('clip_b16_*_smooth_strong.csv'):
            frame = pd.read_csv(path)
            self.assertTrue(frame.probability_fake.between(0,1).all())
            self.assertTrue(frame.loc[frame.roi_status.ne('detected'),'response_margin'].eq(0).all())
        original = a.baseline_runs_root/'clip_b16_ffpp_c23_roi16_baseline_v1_seed42/best.pt'
        self.assertEqual(response.base.common.digest(original), self.initial_hashes['clip_b16'])

    def test_resume_matches_uninterrupted_weights(self):
        a = self.args
        response.train(a)
        expected = {s: copy.deepcopy(torch.load(a.runs_root/response.run_name('clip_b16',s,42)/'last.pt',weights_only=True))
                    for s in a.strategies}
        a.runs_root = self.root/'interrupted_runs'
        real_extract = response.features_for
        def interrupted(*args, **kwargs):
            if args[-1] == 'train_epoch02':
                raise InterruptedError('simulated interruption after completed epoch1')
            return real_extract(*args, **kwargs)
        with patch.object(response,'features_for',side_effect=interrupted):
            with self.assertRaises(InterruptedError):
                response.train(a)
        for strategy in a.strategies:
            run = a.runs_root/response.run_name('clip_b16',strategy,42)
            self.assertFalse((run/'completed.json').exists())
        response.train(a)
        for strategy in a.strategies:
            actual = torch.load(a.runs_root/response.run_name('clip_b16',strategy,42)/'last.pt',weights_only=True)
            self.assertEqual(expected[strategy]['history'],actual['history'])
            for key,value in expected[strategy]['head_state_dict'].items():
                self.assertTrue(torch.equal(value,actual['head_state_dict'][key]),key)


if __name__ == '__main__':
    unittest.main()
