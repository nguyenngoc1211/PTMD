"""Guards for the source-aware split and for what must never reach the model.

These tests exist because the audit found the verified pipeline's own
verification passed 20 checks while 99.8% of test specimens were also in train:
every check was at row or vector level, none at source level.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import numpy as np
import pandas as pd

from split_manager import (MIXED, locked_path, source_aware_indices,
                           unit_labels, units_for)
from group_protocol import (GROUPS, TASKS, build_task, group_map,
                            load_base_frame, parse_category)

METADATA_COLUMNS = {'Class', 'Category', '__label', '__type', '__family',
                    '__specimen', '__run', '__category', '__conflict', '__unit'}
INTACT = None


def setUpModule():
    """Load the real dataset once; every source-level claim is checked on it."""
    global INTACT
    INTACT = load_base_frame()


class SyntheticSplitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.units, self.y = [], []
        for label in (0, 1):
            for specimen in range(60):
                size = int(rng.integers(1, 11))
                for _ in range(size):
                    self.units.append(f'spec{label}_{specimen}')
                    self.y.append(label)
        self.units = np.asarray(self.units, dtype=object)
        self.y = np.asarray(self.y)

    def test_no_unit_crosses_the_boundary(self):
        train, test = source_aware_indices(self.units, self.y, seed=7)
        self.assertFalse(set(self.units[train]) & set(self.units[test]))
        self.assertEqual(len(train) + len(test), len(self.y))
        self.assertEqual(np.intersect1d(train, test).size, 0)

    def test_test_fraction_is_close_to_target(self):
        train, test = source_aware_indices(self.units, self.y, test_size=0.2, seed=7)
        self.assertAlmostEqual(len(test) / len(self.y), 0.2, delta=0.03)

    def test_stratification_holds_per_class(self):
        train, test = source_aware_indices(self.units, self.y, test_size=0.2, seed=7)
        for label in (0, 1):
            share = (self.y[test] == label).mean()
            self.assertAlmostEqual(share, 0.5, delta=0.03,
                                   msg=f'class {label} share in test')

    def test_split_is_reproducible_and_seed_sensitive(self):
        first = source_aware_indices(self.units, self.y, seed=7)
        again = source_aware_indices(self.units, self.y, seed=7)
        other = source_aware_indices(self.units, self.y, seed=8)
        np.testing.assert_array_equal(first[0], again[0])
        np.testing.assert_array_equal(first[1], again[1])
        self.assertFalse(np.array_equal(first[1], other[1]))

    def test_mixed_label_unit_is_one_stratum_and_stays_whole(self):
        units = np.array(['a', 'a', 'b', 'b', 'c', 'c'], dtype=object)
        y = np.array([0, 1, 0, 0, 1, 1])
        unique, strata, sizes = unit_labels(units, y)
        mixed = dict(zip(unique, strata))
        self.assertEqual(mixed['a'], MIXED)
        self.assertEqual(mixed['b'], '0')
        train, test = source_aware_indices(units, y, test_size=0.5, seed=1)
        self.assertFalse(set(units[train]) & set(units[test]))
        self.assertTrue({'a'} <= set(units[test]) or {'a'} <= set(units[train]))

    def test_missing_units_are_rejected(self):
        with self.assertRaises(ValueError):
            source_aware_indices(np.array(['a', None], dtype=object), [0, 1], seed=1)


class RealDataSplitTests(unittest.TestCase):
    """The claims that the old verification could not make."""

    def test_every_malware_row_has_a_parsed_specimen(self):
        frame, features = INTACT
        malware = frame[frame['__label'] == 1]
        self.assertEqual(int(malware['__specimen'].isna().sum()), 0,
                         'a malware row lost its specimen id')
        self.assertEqual(malware['__specimen'].nunique(), 2887)

    def test_random_split_leaks_specimens(self):
        """Documents the defect: this is expected to FAIL on a fixed split."""
        frame, features = INTACT
        task = build_task(frame, features, 'binary')
        units = units_for(task)
        train, test = task.split(42, strategy='random')
        shared = set(units[train]) & set(units[test])
        self.assertGreater(len(shared), 2000,
                           'random split no longer leaks; update TECHNICAL_AUDIT §3')

    def test_source_aware_split_does_not_leak_specimens(self):
        frame, features = INTACT
        for task_name in TASKS:
            task = build_task(frame, features, task_name)
            units = units_for(task)
            for seed in (42, 43, 44):
                train, test = task.split(seed, strategy='source_aware')
                self.assertEqual(set(units[train]) & set(units[test]), set(),
                                 f'{task_name}/{seed} leaked a specimen')

    def test_source_aware_split_is_reproducible_from_data_and_seed(self):
        frame, features = INTACT
        task = build_task(frame, features, 'family16')
        units = units_for(task)
        first = source_aware_indices(units, task.y, seed=42)
        second = source_aware_indices(units, task.y, seed=42)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_locked_split_file_matches_recomputation(self):
        frame, features = INTACT
        path = locked_path('source_aware_v1', 'binary', 42)
        if not path.exists():
            self.skipTest('source_aware_v1 not materialised in this checkout')
        import joblib
        payload = joblib.load(path)
        task = build_task(frame, features, 'binary')
        expected = source_aware_indices(units_for(task), task.y, seed=42)
        np.testing.assert_array_equal(payload['train_indices'], expected[0])
        np.testing.assert_array_equal(payload['test_indices'], expected[1])


class NoLabelInPredictorsTests(unittest.TestCase):
    def test_feature_list_excludes_labels_and_metadata(self):
        frame, features = INTACT
        self.assertEqual(len(features), 55)
        self.assertFalse(METADATA_COLUMNS & set(features))
        for name in features:
            self.assertRegex(name, r'^[a-z]+\.[A-Za-z0-9_]+$')

    def test_every_feature_belongs_to_one_of_nine_groups(self):
        frame, features = INTACT
        mapping = group_map(features)
        self.assertEqual(set(mapping), set(GROUPS))
        self.assertEqual(sum(len(v) for v in mapping.values()), 55)
        self.assertEqual(sorted(mapping), sorted(GROUPS))

    def test_parse_category_never_used_as_a_feature(self):
        self.assertEqual(parse_category('Benign')[:2], ('Benign', 'Benign'))
        parsed = parse_category('Trojan-Zeus-' + 'a' * 64 + '-3.raw')
        self.assertEqual(parsed[:2], ('Trojan', 'Zeus'))
        self.assertEqual(parsed[2], 'a' * 64)
        self.assertEqual(parse_category('garbage'), (None, None, None, None))


if __name__ == '__main__':
    unittest.main()
