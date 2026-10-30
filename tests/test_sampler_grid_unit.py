"""Contract tests for the grid sampler's constraint rejection and snapping."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import numpy as np
import pandas as pd

from run_sampler_grid import (constraint_keep, select_constrained,
                              snap_boundary)


def constraints(rows):
    return pd.DataFrame(rows, columns=['lhs', 'op', 'rhs', 'hold_rate'])


class ConstraintKeepTest(unittest.TestCase):
    def setUp(self):
        # Two evidence rows, three proposals per row.
        self.frame = pd.DataFrame({'a': [5.0, 1.0], 'b': [3.0, 2.0],
                                   'unrelated': [0.0, 9.0]})
        self.block = np.array([[1.0, 4.0, 7.0],
                               [0.5, 1.5, 2.5]])

    def test_constraints_not_involving_feature_are_ignored(self):
        # 'unrelated <= a' does not involve b and must not filter b's draws.
        table = constraints([('unrelated', '<=', 'a', 1.0)])
        keep = constraint_keep(self.block, 'b', table, self.frame, 3)
        self.assertTrue(keep.all())

    def test_feature_on_left_side(self):
        table = constraints([('b', '<=', 'a', 1.0)])
        keep = constraint_keep(self.block, 'b', table, self.frame, 3)
        # Row 0: a=5 -> proposals 1,4 accepted; row 1: a=1 -> only 0.5.
        np.testing.assert_array_equal(keep, [[True, True, False],
                                             [True, False, False]])

    def test_feature_on_right_side(self):
        table = constraints([('unrelated', '<=', 'b', 1.0)])
        keep = constraint_keep(self.block, 'b', table, self.frame, 3)
        # Row 0: unrelated=0 -> all; row 1: unrelated=9 -> none.
        np.testing.assert_array_equal(keep, [[True, True, True],
                                             [False, False, False]])

    def test_non_finite_proposals_are_rejected(self):
        block = self.block.copy()
        block[0, 1] = np.nan
        keep = constraint_keep(block, 'b', constraints([]), self.frame, 3)
        self.assertFalse(keep[0, 1])
        self.assertTrue(keep[0, 0] and keep[0, 2])


class SelectionTest(unittest.TestCase):
    def test_keeps_first_accepted_and_pads_with_nan(self):
        block = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        keep = np.array([[False, True, True],
                         [False, False, False],
                         [True, True, True]])
        values, acceptance = select_constrained(block, keep, 2)
        np.testing.assert_array_equal(values[0], [2.0, 3.0])
        self.assertTrue(np.isnan(values[1]).all())
        np.testing.assert_array_equal(values[2], [7.0, 8.0])
        self.assertAlmostEqual(acceptance, 5 / 9)

    def test_acceptance_denominator_is_finite_proposals(self):
        block = np.array([[1.0, np.nan, 3.0]])
        keep = np.array([[True, False, False]])
        _, acceptance = select_constrained(block, keep, 1)
        self.assertAlmostEqual(acceptance, 0.5)


class SnapTest(unittest.TestCase):
    def test_snaps_only_within_tolerance(self):
        block = np.array([[-2.5e-7, -3.0, 0.4]])
        snapped, count, worst = snap_boundary(block, 5e-7)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(worst, 2.5e-7)
        self.assertEqual(snapped[0, 0], 0.0)
        self.assertEqual(snapped[0, 1], -3.0)

    def test_zero_tolerance_is_a_no_op(self):
        block = np.array([[-1e-9, 1.0]])
        snapped, count, worst = snap_boundary(block, 0.0)
        np.testing.assert_array_equal(snapped, block)
        self.assertEqual(count, 0)
        self.assertEqual(worst, 0.0)


if __name__ == '__main__':
    unittest.main()
