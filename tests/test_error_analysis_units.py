"""Contracts for closed-set error analysis: variant filtering and confusion tables."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from group_protocol import TaskData
from run_error_analysis import (confusion_frames, malware_only, pair_checks,
                                top_confusions)


def toy_task():
    frame = pd.DataFrame({'f': [0.0, 1.0, 2.0, 3.0]})
    return TaskData(name='toy', frame=frame, features=['f'],
                    y=np.array([0, 1, 2, 1]), classes=['Benign', 'CWS', 'Gator'],
                    label_column='__family')


class MalwareOnlyTest(unittest.TestCase):
    def test_drops_benign_and_renumbers(self):
        task = toy_task()
        reduced, keep = malware_only(task)
        self.assertEqual(keep.tolist(), [False, True, True, True])
        self.assertEqual(reduced.classes, ['CWS', 'Gator'])
        self.assertEqual(reduced.y.tolist(), [0, 1, 0])
        self.assertEqual(len(reduced.frame), 3)

    def test_keeps_feature_values_aligned(self):
        reduced, _ = malware_only(toy_task())
        self.assertEqual(reduced.frame['f'].tolist(), [1.0, 2.0, 3.0])


class ConfusionTest(unittest.TestCase):
    def test_normalization_is_per_true_class(self):
        matrix = np.array([[2, 1], [0, 3]])
        counts, normalized = confusion_frames(matrix, ['a', 'b'])
        self.assertEqual(counts.to_numpy().tolist(), [[2, 1], [0, 3]])
        self.assertAlmostEqual(normalized.loc['a', 'b'], 1 / 3)
        self.assertAlmostEqual(normalized.loc['b', 'b'], 1.0)

    def test_top_confusion_excludes_diagonal(self):
        matrix = np.array([[5, 1, 0], [2, 4, 3], [0, 0, 6]])
        top = top_confusions(matrix, ['a', 'b', 'c'])
        pairs = set(zip(top.true_label, top.predicted_label))
        self.assertIn(('a', 'b'), pairs)
        self.assertIn(('b', 'c'), pairs)
        self.assertNotIn(('c', 'c'), pairs)
        self.assertFalse(any(t == p for t, p in pairs))

    def test_empty_true_class_is_skipped(self):
        matrix = np.array([[4, 1], [0, 0]])
        top = top_confusions(matrix, ['a', 'b'])
        self.assertEqual(top.true_label.tolist(), ['a'])


class PairCheckTest(unittest.TestCase):
    def test_reports_both_directions(self):
        matrix = np.array([[8, 2, 0], [3, 5, 2], [0, 0, 4]])
        pairs = pair_checks(matrix, ['a', 'b', 'c'], pairs=(('a', 'b'),))
        self.assertEqual(len(pairs), 2)
        forward = pairs.iloc[0]
        backward = pairs.iloc[1]
        self.assertEqual((forward.true_label, forward.predicted_label), ('a', 'b'))
        self.assertAlmostEqual(forward.rate, 0.2)
        self.assertEqual(backward['count'], 3)
        self.assertAlmostEqual(backward.rate, 0.3)

    def test_absent_label_is_skipped(self):
        matrix = np.array([[4, 1], [0, 5]])
        pairs = pair_checks(matrix, ['a', 'b'], pairs=(('a', 'c'),))
        self.assertTrue(pairs.empty)


if __name__ == '__main__':
    unittest.main()
