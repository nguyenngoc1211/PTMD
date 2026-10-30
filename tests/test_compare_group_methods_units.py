"""Contracts for the cross-method group importance comparison."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from compare_group_methods import (correlation_table, featurewise_scores,
                                   permutation_scores, rank_table)


class PermutationScoresTest(unittest.TestCase):
    def test_p_uses_matched_controls_of_same_group(self):
        runs = pd.DataFrame([
            dict(model='xgboost', group='a', condition='permute_group', delta_log_loss=1.0),
            dict(model='xgboost', group='a', condition='permute_random', delta_log_loss=0.0),
            dict(model='xgboost', group='a', condition='permute_random', delta_log_loss=0.5),
            dict(model='xgboost', group='a', condition='permute_random', delta_log_loss=2.0),
            dict(model='xgboost', group='b', condition='permute_group', delta_log_loss=0.5),
            dict(model='xgboost', group='b', condition='permute_random', delta_log_loss=0.0),
            dict(model='xgboost', group='b', condition='permute_random', delta_log_loss=0.1),
        ])
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(runs).to_csv(Path(directory) / 'permutation_runs.csv', index=False)
            scores = permutation_scores(directory, 'xgboost')
        self.assertAlmostEqual(scores.loc['a', 'permutation_p_empirical'], 0.5)
        self.assertAlmostEqual(scores.loc['b', 'permutation_p_empirical'], 1 / 3)


class FeaturewiseScoresTest(unittest.TestCase):
    def test_features_are_summed_inside_group_then_averaged_over_runs(self):
        rows = []
        for repeat, delta in ((0, 1.0), (1, 3.0)):
            rows.extend([
                dict(model='xgboost', split_seed=42, model_seed=0, repeat=repeat,
                     feature='a.f1', delta_log_loss=delta),
                dict(model='xgboost', split_seed=42, model_seed=0, repeat=repeat,
                     feature='a.f2', delta_log_loss=delta),
                dict(model='xgboost', split_seed=42, model_seed=0, repeat=repeat,
                     feature='b.f1', delta_log_loss=0.0),
            ])
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(rows).to_csv(Path(directory) / 'stability_runs.csv', index=False)
            scores = featurewise_scores(directory, 'xgboost')
        self.assertAlmostEqual(scores.loc['a', 'featurewise_delta_log_loss'], 4.0)
        self.assertAlmostEqual(scores.loc['b', 'featurewise_delta_log_loss'], 0.0)


class CorrelationTest(unittest.TestCase):
    def test_identical_rankings_give_spearman_one(self):
        groups = ['pslist', 'dlllist', 'handles', 'ldrmodules', 'malfind',
                  'psxview', 'modules', 'svcscan', 'callbacks']
        values = np.arange(len(groups), dtype=float)
        scores = pd.DataFrame({f'{m}_score': values for m in
                               ('ablation', 'permutation', 'cfi', 'featurewise')},
                              index=groups)
        ranks = rank_table(scores)
        table = correlation_table(ranks, {'all_groups': groups})
        self.assertEqual(len(table), 6)
        self.assertTrue(np.allclose(table.spearman, 1.0))

    def test_missing_method_is_skipped(self):
        groups = ['a', 'b', 'c']
        scores = pd.DataFrame({
            'ablation_score': [3, 2, 1],
            'permutation_score': [3, np.nan, 2],
            'cfi_score': [1, 2, 3],
            'featurewise_score': [3, 2, 1],
        }, index=groups)
        ranks = rank_table(scores)
        table = correlation_table(ranks, {'all_groups': groups})
        pairs = set(zip(table.method_a, table.method_b))
        self.assertNotIn(('permutation', 'cfi'), pairs)


if __name__ == '__main__':
    unittest.main()
