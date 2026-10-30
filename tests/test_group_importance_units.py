"""Summary contract for group permutation: the null must match block width."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import pandas as pd

from run_group_importance import summarise_permutation


def runs_frame():
    rows = []
    controls = {'narrow': [0.01, 0.02, 0.03, 0.04],
                'wide': [0.60, 0.70, 0.80, 0.90]}
    observed = {'narrow': 0.50, 'wide': 0.95}
    for group, values in controls.items():
        rows.append(dict(task='t', model='m', group=group,
                         condition='permute_group', control_index=-1,
                         delta_log_loss=observed[group], seed=42))
        for index, value in enumerate(values):
            rows.append(dict(task='t', model='m', group=group,
                             condition='permute_random', control_index=index,
                             delta_log_loss=value, seed=42))
    return pd.DataFrame(rows)


class MatchedNullTest(unittest.TestCase):
    def test_p_uses_only_the_same_group_controls(self):
        summary = summarise_permutation(runs_frame()).set_index('group')
        self.assertAlmostEqual(summary.loc['narrow', 'p_empirical'], 1 / 5)
        self.assertAlmostEqual(summary.loc['wide', 'p_empirical'], 1 / 5)

    def test_null_mean_is_per_group(self):
        summary = summarise_permutation(runs_frame()).set_index('group')
        self.assertAlmostEqual(summary.loc['narrow', 'null_mean'], 0.025)
        self.assertAlmostEqual(summary.loc['wide', 'null_mean'], 0.75)

    def test_rank_orders_observed_effects(self):
        summary = summarise_permutation(runs_frame()).set_index('group')
        self.assertEqual(int(summary.loc['wide', 'rank']), 1)
        self.assertEqual(int(summary.loc['narrow', 'rank']), 2)


if __name__ == '__main__':
    unittest.main()
