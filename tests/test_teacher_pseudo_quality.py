"""CPU checks for the teacher pseudo-label diagnostic: the TS rule scoring and a tiny end-to-end run."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_teacher_pseudo_quality as teacher
import test_smoothing_preservation as fixture


class TeacherQualityTests(unittest.TestCase):
    setUp = fixture.PreservationTests.setUp
    tearDown = fixture.PreservationTests.tearDown

    def test_pseudo_rule_counts(self):
        frame = pd.DataFrame(dict(class_index=['0', '0', '0', '1', '1', '1'],
                                  kind=['Celeb-real', 'YouTube-real', 'Celeb-real'] + ['Celeb-synthesis']*3))
        p = np.array([.02, .95, .6, .97, .99, .05])
        row = teacher.pseudo_rows(frame, p, .9)
        self.assertEqual((row['selected_real'], row['selected_fake']), (2, 3))
        self.assertAlmostEqual(row['selected_fraction'], 5/6)
        self.assertAlmostEqual(row['pseudo_accuracy'], 3/5)
        self.assertAlmostEqual(row['fake_side_error'], 1/3)     # one true real among three "fake" pseudo-labels
        self.assertAlmostEqual(row['real_side_error'], 1/2)     # one true fake among two "real" pseudo-labels
        self.assertAlmostEqual(row['true_real_called_fake'], 1/3)
        self.assertAlmostEqual(row['YouTube-real_called_fake'], 1.)
        self.assertAlmostEqual(row['Celeb-real_called_fake'], 0.)

    def test_end_to_end_and_reuse(self):
        a = copy.copy(self.args)
        test = pd.read_csv(self.test, dtype=str)
        membership = test.assign(kind=np.where(test.class_index.eq('1'), 'Celeb-synthesis', 'Celeb-real'), partition='adapt_train')
        extra = membership.iloc[:2].assign(partition='adapt_val')
        a.membership = self.root/'membership_labelled.csv'
        pd.concat([membership, extra.assign(crop_path='unused_'+extra.crop_path)]).to_csv(a.membership, index=False)
        a.models, a.strategies, a.thresholds = ['xception'], list(teacher.style.STRATEGIES), [.9, .95]
        a.ce_rank_runs_root, a.pixel_cache = self.root/'ce_runs', self.root/'pixels'
        run = a.ce_rank_runs_root/teacher.rank.run_name('xception', 'ce_rank', 42)
        run.mkdir(parents=True)
        torch.save(dict(config=dict(strategy='ce_rank', initial_checkpoint_sha256=self.initial_hashes['xception']),
                        model_state_dict={k: v+.1 for k, v in fixture.Tiny().state_dict().items()}), run/'best.pt')
        teacher.evaluate(a)
        table = pd.read_csv(a.output_root/'summary.csv')
        self.assertEqual(len(table), 2*2*5)                       # strategies x thresholds x (4 conditions + all)
        self.assertEqual(set(table.condition), {*teacher.CONDITIONS, 'all'})
        self.assertTrue((table[table.condition.eq('all')].frames == 4*len(membership)).all())
        self.assertTrue((table[table.condition.ne('all')].frames == len(membership)).all())
        times = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        teacher.evaluate(a)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})


if __name__ == '__main__':
    unittest.main()
