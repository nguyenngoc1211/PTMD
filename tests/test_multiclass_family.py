"""Regression checks for the multiclass family attribution study."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from run_multiclass_family import (HierarchicalClassifier, class_maps,
                                   drop_family_conflicts, map_families_to_types,
                                   score, stratified_split)


def hierarchy_frame():
    """Benign plus one malware type holding two separable families."""
    rows = []
    for family, type_, base in (('Benign', 'Benign', 0.0),
                                ('Alpha', 'Trojan', 10.0),
                                ('Beta', 'Trojan', 20.0)):
        for i in range(40):
            rows.append(dict(__family=family, __type=type_,
                             a=base + i * 0.01, b=base * 2 + i * 0.01))
    return pd.DataFrame(rows)


class MulticlassFamilyTests(unittest.TestCase):
    def test_class_maps_keeps_benign_and_is_sorted(self):
        frame = pd.DataFrame({'__family': ['Benign', 'Ako', 'Benign', 'Zeus'],
                              '__type': ['Benign', 'Ransomware', 'Benign', 'Trojan']})
        families, types, family_code, type_code = class_maps(frame)
        self.assertEqual(families, ['Ako', 'Benign', 'Zeus'])
        self.assertEqual(types, ['Benign', 'Ransomware', 'Trojan'])
        self.assertEqual(family_code['Benign'], 1)
        self.assertIn('Benign', families)

    def test_map_families_to_types_groups_families(self):
        frame = pd.DataFrame({'__family': ['Benign', 'Ako', 'Benign', 'Zeus'],
                              '__type': ['Benign', 'Ransomware', 'Benign', 'Trojan']})
        families, types, family_code, type_code = class_maps(frame)
        mapping = map_families_to_types(frame, families, family_code, type_code)
        self.assertEqual(mapping[type_code['Benign']], [family_code['Benign']])
        self.assertEqual(mapping[type_code['Ransomware']], [family_code['Ako']])
        self.assertEqual(mapping[type_code['Trojan']], [family_code['Zeus']])

    def test_stratified_split_is_disjoint_and_stratified(self):
        y = np.array([0] * 60 + [1] * 40)
        frame = pd.DataFrame({'x': np.arange(100)})
        train, test = stratified_split(frame, seed=0, test_size=0.2, y=y)
        self.assertEqual(np.intersect1d(train, test).size, 0)
        self.assertEqual(len(test), 20)
        self.assertLess(abs(y[test].mean() - 0.4), 0.06)
        self.assertEqual(set(y[train]), {0, 1})

    def test_score_macro_f1_matches_manual_computation(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        result = score(y_true, y_pred, 3)
        # class F1: 1.0, 2/3, 0.8 -> macro (1 + 0.666667 + 0.8) / 3
        self.assertAlmostEqual(result['macro_f1'], 0.822222, places=5)
        self.assertAlmostEqual(result['accuracy'], 5 / 6, places=6)
        self.assertAlmostEqual(result['balanced_accuracy'], 2.5 / 3, places=6)

    def test_drop_family_conflicts_removes_ambiguous_vectors(self):
        raw = pd.DataFrame({
            'Category': ['Trojan-Alpha-h-1.raw', 'Trojan-Beta-h-2.raw',
                         'Trojan-Alpha-h-3.raw', 'Benign'],
            'Class': ['Malware', 'Malware', 'Malware', 'Benign'],
            'a': [1.0, 1.0, 2.0, 0.0],
            'b': [5.0, 5.0, 6.0, 4.0],
        })
        # load_frame keeps the first row of every duplicate vector.
        frame = pd.DataFrame({'a': [1.0, 2.0, 0.0], 'b': [5.0, 6.0, 4.0]})
        kept, n_groups, n_rows = drop_family_conflicts(frame, ['a', 'b'], raw=raw)
        self.assertEqual(n_groups, 1)
        self.assertEqual(n_rows, 1)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept['a'].tolist(), [2.0, 0.0])

    def test_hierarchical_combines_type_then_family(self):
        frame = hierarchy_frame()
        features = ['a', 'b']
        families, types, family_code, type_code = class_maps(frame)
        mapping = map_families_to_types(frame, families, family_code, type_code)
        y16 = frame['__family'].map(family_code).to_numpy()
        y4 = frame['__type'].map(type_code).to_numpy()
        model = HierarchicalClassifier(42, False, type_code, family_code, mapping)
        model.fit(frame[features], y4, y16)
        prediction = model.predict(frame[features])
        self.assertEqual(accuracy_score(y16, prediction), 1.0)
        benign_rows = y4 == type_code['Benign']
        self.assertEqual(set(prediction[benign_rows]), {family_code['Benign']})


if __name__ == '__main__':
    unittest.main()
