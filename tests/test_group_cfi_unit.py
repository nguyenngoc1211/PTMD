"""Contract tests for Group CFI, Group Permutation and draw alignment.

These tests run without R: they feed crafted draws to the Python side and
check the mathematical contract that the R sampler is supposed to satisfy.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import numpy as np
import pandas as pd

import run_group_cfi
import run_group_importance
from run_group_cfi import align_group, group_importance
from run_group_importance import permutation


class ToyTask:
    def __init__(self, frame, y, name='toy'):
        self.name = name
        self.features = list(frame.columns)
        self.frame = frame
        self.y = np.asarray(y)
        self.n_classes = int(len(np.unique(self.y)))
        self.classes = [str(i) for i in range(self.n_classes)]

    def features_of(self, group):
        return [c for c in self.features if c.split('.')[0] == group]


class SingleFeatureTask(ToyTask):
    def features_of(self, group):
        return [group]


class RecordingModel:
    """Scores rows as a logistic function of the row sum and records inputs."""

    def __init__(self, scale=20.0):
        self.scale = scale
        self.seen = []

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.seen.append(X.copy())
        p = 1.0 / (1.0 + np.exp(-self.scale * (X.sum(axis=1) - 1.0)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def make_task(n=12, test_rows=5):
    a = np.linspace(0.1, 0.9, n)
    frame = pd.DataFrame({'pslist.a': a, 'pslist.b': 1.0 - a})
    y = np.arange(n) % 2
    return ToyTask(frame, y), np.arange(n - test_rows, n)


def joint_draws(task, test_rows, repeats, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for position, _ in enumerate(test_rows):
        for r in range(1, repeats + 1):
            a = rng.uniform(0.15, 0.85)
            rows.append(dict(test_position=position, replicate=r,
                             **{'pslist.a': a, 'pslist.b': 1.0 - a}))
    return pd.DataFrame(rows)


class DrawAlignmentTests(unittest.TestCase):
    def test_align_group_enforces_contract_and_order(self):
        frame = pd.DataFrame({'test_position': [0, 0, 1, 1],
                              'replicate': [2, 1, 1, 2],
                              'a': [10.0, 11.0, 20.0, 21.0],
                              'b': [100.0, 110.0, 200.0, 210.0]})
        block = align_group(frame, ['a', 'b'], [0, 1], 2)
        np.testing.assert_array_equal(block[:, :, 0], [[11.0, 10.0], [20.0, 21.0]])
        np.testing.assert_array_equal(block[:, :, 1], [[110.0, 100.0], [200.0, 210.0]])

    def test_align_group_rejects_duplicates_and_extras_but_allows_missing(self):
        frame = pd.DataFrame({'test_position': [0, 0], 'replicate': [1, 1],
                              'a': [1.0, 2.0]})
        with self.assertRaises(ValueError):
            align_group(frame, ['a'], [0], 2)
        frame = pd.DataFrame({'test_position': [5], 'replicate': [1], 'a': [1.0]})
        with self.assertRaises(ValueError):
            align_group(frame, ['a'], [0], 1)
        frame = pd.DataFrame({'test_position': [0], 'replicate': [1], 'a': [1.0]})
        block = align_group(frame, ['a'], [0, 1], 2)
        self.assertEqual(block.shape, (2, 2, 1))
        self.assertTrue(np.isnan(block[1]).all())


class GroupCfiTests(unittest.TestCase):
    def run_cfi(self, task, draws, groups, test_rows, repeats=4, model=None):
        with tempfile.TemporaryDirectory() as tmp:
            old_draws = run_group_cfi.DRAWS
            old_groups = run_group_cfi.GROUPS
            run_group_cfi.DRAWS = Path(tmp)
            run_group_cfi.GROUPS = groups
            try:
                for name, frame in draws.items():
                    frame.to_csv(Path(tmp) / f'draws_{task.name}_42_{name}.csv',
                                 index=False)
                model = model or RecordingModel()
                cfi, diagnostics = group_importance(task, model, test_rows,
                                                    repeats, 42)
            finally:
                run_group_cfi.DRAWS = old_draws
                run_group_cfi.GROUPS = old_groups
        return cfi, diagnostics, model

    def test_group_cfi_replaces_whole_group_simultaneously(self):
        task, test_rows = make_task()
        draws = {'pslist': joint_draws(task, test_rows, 4)}
        cfi, diagnostics, model = self.run_cfi(task, draws, ['pslist'], test_rows)
        for matrix in model.seen:
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(int(cfi.n_features.iloc[0]), 2)
        self.assertAlmostEqual(float(cfi.conditional_importance.iloc[0]), 0.0,
                               places=12)
        self.assertEqual(float(cfi.p_value.iloc[0]), 1.0)
        self.assertEqual(float(cfi.coverage.iloc[0]), 1.0)
        self.assertTrue(bool(diagnostics.gate_pass.iloc[0]))

    def test_group_cfi_does_not_sum_featurewise(self):
        task, test_rows = make_task()
        repeats = 4
        joint = joint_draws(task, test_rows, repeats)
        rng = np.random.default_rng(7)
        single = {}
        for column in task.features:
            rows = []
            for position, _ in enumerate(test_rows):
                for r in range(1, repeats + 1):
                    rows.append(dict(test_position=position, replicate=r,
                                     **{column: rng.uniform(0.15, 0.85)}))
            single[column] = pd.DataFrame(rows)

        joint_cfi, _, _ = self.run_cfi(task, {'pslist': joint}, ['pslist'],
                                       test_rows, repeats)
        feature_task = SingleFeatureTask(task.frame, task.y)
        feature_cfi, _, _ = self.run_cfi(feature_task, single,
                                         ['pslist.a', 'pslist.b'], test_rows,
                                         repeats,
                                         model=RecordingModel())
        joint_value = float(joint_cfi.conditional_importance.iloc[0])
        additive = float(feature_cfi.conditional_importance.sum())
        self.assertAlmostEqual(joint_value, 0.0, places=12)
        self.assertGreater(additive, 0.1)
        self.assertGreater(feature_cfi.conditional_importance.min(), 0.0)

    def test_invalid_draws_are_reported_not_dropped_silently(self):
        task, test_rows = make_task()
        draws = joint_draws(task, test_rows, 4)
        for position in (0, 1):
            mask = draws.test_position == position
            draws.loc[mask, ['pslist.a', 'pslist.b']] = np.nan
        cfi, diagnostics, _ = self.run_cfi(task, {'pslist': draws}, ['pslist'],
                                           test_rows)
        self.assertAlmostEqual(float(cfi.coverage.iloc[0]), 3 / 5)
        self.assertEqual(int(cfi.n_valid.iloc[0]), 3)
        self.assertFalse(bool(diagnostics.gate_pass.iloc[0]))
        self.assertIn('coverage', diagnostics.gate_reasons.iloc[0])


class MultiClassModel:
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        logits = np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))])
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class GroupPermutationTests(unittest.TestCase):
    def test_permutation_supports_multiclass(self):
        rng = np.random.default_rng(11)
        n = 30
        frame = pd.DataFrame({'pslist.a': rng.uniform(0, 1, n),
                              'handles.c': rng.uniform(0, 1, n)})
        task = ToyTask(frame, np.arange(n) % 3)
        test = np.arange(n - 20, n)
        model = MultiClassModel()
        runs, rows, _ = permutation(task, model, [0, 1], test,
                                    {'pslist': ['pslist.a']},
                                    repeats=2, seed=42, controls=1)
        self.assertEqual(rows.shape, (len(test), 1))
        self.assertTrue(np.isfinite(runs.delta_log_loss).all())
        self.assertEqual(set(runs.condition), {'permute_group', 'permute_random'})

    def test_permutation_preserves_within_group_structure(self):
        rng = np.random.default_rng(3)
        n = 30
        frame = pd.DataFrame({
            'pslist.a': rng.uniform(0, 1, n),
            'pslist.b': rng.uniform(0, 1, n),
            'handles.c': rng.uniform(0, 1, n),
            'handles.d': rng.uniform(0, 1, n),
            'handles.e': rng.uniform(0, 1, n),
        })
        task = ToyTask(frame, np.arange(n) % 2)
        test = np.arange(n - 20, n)
        index = list(range(5))
        model = RecordingModel()
        runs, rows, _ = permutation(task, model, index, test,
                                    {'pslist': ['pslist.a', 'pslist.b']},
                                    repeats=3, seed=42, controls=0)
        original = frame.iloc[test][['pslist.a', 'pslist.b']].to_numpy()
        original_pairs = sorted(map(tuple, original.round(12)))
        for matrix in model.seen:
            pairs = sorted(map(tuple, matrix[:, :2].round(12)))
            self.assertEqual(pairs, original_pairs)
        self.assertFalse(any(np.array_equal(matrix[:, :2], original)
                             for matrix in model.seen[2:]))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs.group.iloc[0], 'pslist')
        self.assertEqual(rows.shape, (len(test), 1))
        self.assertEqual(list(rows.columns), ['pslist'])


if __name__ == '__main__':
    unittest.main()
