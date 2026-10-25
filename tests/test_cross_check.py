"""Regression checks for the external-benchmark methodology cross-check."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import pandas as pd

from cross_check_benchmark import (build_rows, oacsp_top, stability_split,
                                   topk_overlap)


class CrossCheckTests(unittest.TestCase):
    def test_topk_overlap_counts_and_sorts(self):
        a = ['x', 'y', 'z', 'w']
        b = ['y', 'x', 'q', 'w']
        count, shared = topk_overlap(a, b, 3)
        self.assertEqual(count, 2)
        self.assertEqual(shared, ['x', 'y'])
        self.assertEqual(topk_overlap(a, b, 4)[0], 3)

    def test_oacsp_top_orders_by_gain(self):
        importance = {'features': {'a': 1.0, 'b': 5.0, 'c': 3.0}}
        self.assertEqual(oacsp_top(importance, 2), ['b', 'c'])
        self.assertEqual(oacsp_top(None, 2), [])

    def test_stability_split_separates_benign_from_malware(self):
        frame = pd.DataFrame([
            {'Perturbation_Percent': 0.5, 'Family': 'Benign',
             'Mean_Prediction_Stability': 1.0},
            {'Perturbation_Percent': 0.5, 'Family': 'Ako',
             'Mean_Prediction_Stability': 0.2},
            {'Perturbation_Percent': 0.5, 'Family': 'CWS',
             'Mean_Prediction_Stability': 0.4},
            {'Perturbation_Percent': 5.0, 'Family': 'Benign',
             'Mean_Prediction_Stability': 0.0},
        ])
        result = stability_split(frame)
        self.assertEqual(result['scale'], 0.5)
        self.assertAlmostEqual(result['benign'], 1.0)
        self.assertAlmostEqual(result['malware'], 0.3)
        self.assertIsNone(stability_split(None))

    def test_build_rows_marks_exact_group_match(self):
        our = dict(raw_rows=100, unique_rows=90, duplicates_removed=10,
                   test_rows=18, binary_f1=1.0, lofo_f1=0.99, control_acc=0.999)
        bench = dict(split={'row_count': 100, 'unique_groups': 90,
                            'locked_test_group_count': 18},
                     audit={'duplicate_group_count': 9}, stability=None)
        oacsp = dict(binary={'accuracy': 0.9999}, family={'accuracy': 0.87},
                     ood={'accuracy': 0.9998}, importance=None)
        rows = build_rows(our, bench, oacsp, (5, []), (7, []))
        by_aspect = {row['Khía cạnh']: row for row in rows}
        self.assertEqual(by_aspect['Số vector đặc trưng duy nhất (nhóm)']['Đánh giá'],
                         'khớp tuyệt đối')
        self.assertEqual(by_aspect['Top-5 feature trùng khớp']['PTMD'], '5/5')
        self.assertTrue(all(set(r) == {'Khía cạnh', 'PTMD', 'Đối chiếu', 'Nguồn',
                                       'Đánh giá'} for r in rows))


if __name__ == '__main__':
    unittest.main()
