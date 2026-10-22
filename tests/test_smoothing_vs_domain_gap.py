"""Synthetic checks for the domain/smoothing AUC split; no models or data."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_smoothing_vs_domain_gap as gap


def grid(strategy, base, shift):
    # AUC rises when the fake is smoothed more than the real, falls in the opposite direction.
    return pd.DataFrame([dict(model='m', strategy=strategy, real_condition=r, fake_condition=f,
                              auc=base+shift*(j-i)) for i, r in enumerate(gap.CONDITIONS)
                         for j, f in enumerate(gap.CONDITIONS)])


class GapSplitTests(unittest.TestCase):
    def test_symmetric_shift_costs_no_pooled_auc_but_shows_in_sides(self):
        table = gap.analyse([grid('baseline', .70, .03), grid(gap.REFERENCE, .85, 0.)], gap.REFERENCE).set_index('strategy')
        row = table.loc['baseline']
        self.assertAlmostEqual(row.shift_cost, 0.)
        self.assertAlmostEqual(row.domain_gap, .15)
        self.assertAlmostEqual(row.gap_to_reference, row.domain_gap+row.smoothing_gap)
        self.assertAlmostEqual(row.strong_real_vs_normal_fake, .61)
        self.assertLess(row.fp_side, row.matched)
        self.assertGreater(row.fn_side, row.matched)


if __name__ == '__main__':
    unittest.main()
