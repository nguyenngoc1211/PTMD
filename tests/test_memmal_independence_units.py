"""Contracts for the MemMal-D2024 independence audit."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from verify_memmal_independence import compare


def local_frame():
    return pd.DataFrame({
        'Category': ['Ako', 'Ako', 'CWS'],
        'f1': [1.0, 2.0, 3.0],
        'f2': [0.5, 0.25, 0.125],
        'Class': ['Malware', 'Malware', 'Benign'],
    })


def memmal_frame():
    return pd.DataFrame({
        'Category': ['Ako', 'Ako', 'CWS'],
        'f1': [1.0, 3.0, 9.0],
        'f2': [0.5, 0.125, 0.1],
        'Class': ['Malware'] * 3,
    })


class CompareTest(unittest.TestCase):
    def test_counts_exact_matches_of_unique_rows(self):
        result = compare(memmal_frame(), local_frame())
        self.assertEqual(result['shared_features'], 2)
        self.assertEqual(result['local_malware_rows'], 2)
        self.assertEqual(result['memmal_unique_rows'], 3)
        self.assertEqual(result['matched_unique_rows'], 1)
        self.assertEqual(result['unmatched_unique_rows'], 2)

    def test_duplicate_rows_collapse(self):
        frame = pd.concat([memmal_frame(), memmal_frame().iloc[[0]]], ignore_index=True)
        result = compare(frame, local_frame())
        self.assertEqual(result['memmal_rows'], 4)
        self.assertEqual(result['memmal_unique_rows'], 3)
        self.assertEqual(result['matched_unique_rows'], 1)

    def test_category_is_part_of_identity(self):
        frame = memmal_frame()
        frame.loc[0, 'Category'] = 'CWS'
        result = compare(frame, local_frame())
        self.assertEqual(result['matched_unique_rows'], 0)

    def test_missing_feature_is_fatal(self):
        with self.assertRaises(SystemExit):
            compare(memmal_frame().drop(columns=['f2']), local_frame())


if __name__ == '__main__':
    unittest.main()
