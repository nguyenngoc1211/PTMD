"""Regression tests for the multiclass conditional-importance module."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from cfi_family import (build_task, cfi_table, class_profile, group_importance,
                        multiclass_logloss_rows, subset_positions)

FEATURES = ['handles.nfile', 'svcscan.nservices', 'malfind.commitCharge']


def synthetic_frame(n=900, seed=0):
    """Three malware types where only ``handles.nfile`` separates them."""
    rng = np.random.default_rng(seed)
    types = np.repeat(['Ransomware', 'Spyware', 'Trojan'], n // 3)
    shift = {'Ransomware': 0.0, 'Spyware': 4.0, 'Trojan': 8.0}
    frame = pd.DataFrame({
        'handles.nfile': np.array([shift[t] for t in types]) + rng.normal(0, 0.5, len(types)),
        'svcscan.nservices': rng.normal(0, 1, len(types)),
        'malfind.commitCharge': rng.normal(0, 1, len(types)),
    })
    frame['__type'] = types
    frame['__family'] = types
    frame['__label'] = 1
    return frame


class MulticlassLoglossTest(unittest.TestCase):
    def test_matches_sklearn(self):
        y = np.array([0, 1, 2, 1])
        p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3],
                      [0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
        expected = log_loss(y, p, labels=[0, 1, 2])
        self.assertAlmostEqual(multiclass_logloss_rows(y, p).mean(), expected, places=12)


class BuildTaskTest(unittest.TestCase):
    def test_malware3_drops_benign_and_sorts_classes(self):
        frame = synthetic_frame()
        benign = frame.iloc[[0]].copy()
        benign['__type'] = 'Benign'
        benign['__family'] = 'Benign'
        benign['__label'] = 0
        full = pd.concat([frame, benign], ignore_index=True)
        subset, codes, classes = build_task(full, 'malware3')
        self.assertEqual(classes, ['Ransomware', 'Spyware', 'Trojan'])
        self.assertEqual(len(subset), len(frame))
        self.assertNotIn('Benign', set(subset['__type']))
        self.assertEqual(sorted(set(codes.tolist())), [0, 1, 2])

    def test_type4_keeps_benign(self):
        frame = synthetic_frame()
        benign = frame.iloc[[0]].copy()
        benign['__type'] = 'Benign'
        benign['__family'] = 'Benign'
        benign['__label'] = 0
        full = pd.concat([frame, benign], ignore_index=True)
        _, codes, classes = build_task(full, 'type4')
        self.assertEqual(classes, ['Benign', 'Ransomware', 'Spyware', 'Trojan'])
        self.assertEqual(len(set(codes.tolist())), 4)


class SubsetPositionsTest(unittest.TestCase):
    def test_deterministic_sorted_and_stratified(self):
        y = np.repeat([0, 1, 2], 50)
        first = subset_positions(y, 30, 42)
        second = subset_positions(y, 30, 42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(sorted(first.tolist()), first.tolist())
        self.assertEqual(np.bincount(y[first]).tolist(), [10, 10, 10])

    def test_full_size_returns_everything(self):
        y = np.repeat([0, 1], 5)
        np.testing.assert_array_equal(subset_positions(y, 100, 42), np.arange(10))


class GroupImportanceTest(unittest.TestCase):
    def test_sums_by_group_and_counts_significant(self):
        importance = pd.DataFrame({
            'feature': ['handles.a', 'handles.b', 'svcscan.c'],
            'conditional_importance': [1.0, 2.0, 4.0],
            'conditional_rank': [3, 2, 1],
            'q_value': [0.5, 0.01, 0.001],
        })
        out = group_importance(importance, 0.05).set_index('forensic_group')
        self.assertEqual(out.index[0], 'svcscan')
        self.assertAlmostEqual(out.loc['handles', 'conditional_importance'], 3.0)
        self.assertEqual(out.loc['handles', 'n_significant'], 1)
        self.assertEqual(out.loc['svcscan', 'n_significant'], 1)
        self.assertEqual(int(out.loc['handles', 'best_rank']), 2)


class ClassProfileTest(unittest.TestCase):
    def test_columns_are_classes_and_rows_are_features(self):
        frame = synthetic_frame()
        subset, codes, classes = build_task(frame, 'malware3')
        X = subset[FEATURES].reset_index(drop=True)
        medians, means = class_profile(X, codes, classes, ['handles.nfile'])
        self.assertEqual(list(medians.columns), classes)
        self.assertEqual(list(medians.index), ['handles.nfile'])
        # The separating feature must show an increasing mean across classes.
        self.assertLess(means.loc['handles.nfile', 'Ransomware'],
                        means.loc['handles.nfile', 'Trojan'])


class ConditionalImportanceTest(unittest.TestCase):
    def test_informative_feature_ranks_first_and_noise_stays_flat(self):
        frame = synthetic_frame()
        subset, codes, classes = build_task(frame, 'malware3')
        X = subset[FEATURES].to_numpy()
        model = XGBClassifier(n_estimators=60, max_depth=4, random_state=0, n_jobs=1)
        model.fit(X, codes)
        repeats = 3
        positions = np.arange(len(X))
        rng = np.random.default_rng(1)
        columns = {'test_position': np.repeat(positions, repeats),
                   'replicate': np.tile(np.arange(1, repeats + 1), len(X))}
        for j, feature in enumerate(FEATURES):
            if j == 0:
                # Break the association by drawing values from other classes.
                columns[feature] = np.repeat(rng.permutation(X[:, 0]), repeats)
            else:
                columns[feature] = np.repeat(X[:, j], repeats)
        out = cfi_table(model, X, codes, pd.DataFrame(columns), FEATURES,
                        positions, repeats, 0.05)
        self.assertEqual(out.iloc[0]['feature'], 'handles.nfile')
        self.assertGreater(out.iloc[0]['conditional_importance'], 0)
        noise = out[out['feature'] != 'handles.nfile']['conditional_importance']
        self.assertLess(noise.abs().max(), out.iloc[0]['conditional_importance'])
        self.assertEqual(int(out.iloc[0]['conditional_rank']), 1)

    def test_coverage_drops_when_draws_are_missing(self):
        frame = synthetic_frame(n=90)
        subset, codes, classes = build_task(frame, 'malware3')
        X = subset[FEATURES].to_numpy()
        model = XGBClassifier(n_estimators=20, max_depth=3, random_state=0, n_jobs=1)
        model.fit(X, codes)
        repeats = 2
        positions = np.arange(len(X))
        columns = {'test_position': np.repeat(positions, repeats),
                   'replicate': np.tile(np.arange(1, repeats + 1), len(X))}
        for j, feature in enumerate(FEATURES):
            values = np.repeat(X[:, j], repeats)
            if j == 0:
                values[:10] = np.nan
            columns[feature] = values
        out = cfi_table(model, X, codes, pd.DataFrame(columns), FEATURES,
                        positions, repeats, 0.05)
        coverage = out.set_index('feature')['coverage']
        self.assertAlmostEqual(coverage['handles.nfile'], (len(X) - 5) / len(X))


if __name__ == '__main__':
    unittest.main()
