"""Regression checks for the leave-one-family-out generalization study."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import pandas as pd

from run_generalization import engineer, lofo_folds, control_split


FEATURES = ['a', 'b', 'pslist.nproc', 'handles.nhandles', 'handles.nfile']


def prepared():
    """Two malware families plus benign, mirroring load_frame's output schema."""
    rows = []
    for family in ('Alpha', 'Beta'):
        for i in range(3):
            rows.append(dict(Category=f'Trojan-{family}-deadbeef-{i}.raw',
                             Class='Malware', a=float(i), b=float(2 * i),
                             **{'pslist.nproc': float(i + 1),
                                'handles.nhandles': float(2 * (i + 1)),
                                'handles.nfile': float(i + 1)}))
    for i in range(6):
        rows.append(dict(Category='Benign', Class='Benign', a=10.0 + i, b=20.0 + i,
                         **{'pslist.nproc': 1.0, 'handles.nhandles': 3.0,
                            'handles.nfile': 2.0}))
    df = pd.DataFrame(rows)
    frame = df[FEATURES].copy()
    frame['__label'] = df['Class'].eq('Malware').astype(int).to_numpy()
    frame['__type'] = df['Category'].str.split('-').str[0].to_numpy()
    frame['__family'] = df['Category'].str.split('-').str[1].fillna('Benign').to_numpy()
    return frame


class GeneralizationTests(unittest.TestCase):
    def test_engineer_adds_ratios_and_group_aggregates(self):
        frame = prepared()
        engineered = engineer(frame, FEATURES)
        for column in ('ratio.handles.nhandles', 'group.handles.sum', 'group.handles.mean'):
            self.assertIn(column, engineered.columns)
        # Ratio is normalised by (nproc + 1); first malware row has nproc=1, nhandles=2.
        self.assertAlmostEqual(engineered['ratio.handles.nhandles'].iloc[0], 2.0 / 2.0)
        # Group sum/mean combine the two handles columns.
        self.assertAlmostEqual(engineered['group.handles.sum'].iloc[0], 2.0 + 1.0)
        self.assertAlmostEqual(engineered['group.handles.mean'].iloc[0], (2.0 + 1.0) / 2)
        # Original columns are preserved and the input frame is not mutated.
        self.assertTrue(set(FEATURES).issubset(engineered.columns))
        self.assertNotIn('ratio.handles.nhandles', frame.columns)

    def test_lofo_holds_out_whole_family_and_keeps_benign_split(self):
        frame = prepared()
        label = frame['__label'].to_numpy()
        family = frame['__family'].to_numpy()
        seen = set()
        for name, train, test in lofo_folds(frame, seed=0, test_size=0.5):
            seen.add(name)
            held = np.flatnonzero(family == name)
            other = np.flatnonzero((label == 1) & (family != name))
            self.assertTrue(np.isin(held, test).all(), 'held family must be in test')
            self.assertFalse(np.isin(other, test).any(), 'other families must not leak into test')
            self.assertEqual(np.intersect1d(train, test).size, 0)
            self.assertEqual(set(label[test]), {0, 1})
            self.assertEqual(set(label[train]), {0, 1})
        self.assertEqual(seen, {'Alpha', 'Beta'})

    def test_control_split_is_stratified_and_disjoint(self):
        frame = prepared()
        label = frame['__label'].to_numpy()
        train, test = control_split(frame, seed=0, test_size=0.5)
        self.assertEqual(np.intersect1d(train, test).size, 0)
        self.assertEqual(set(label[test]), {0, 1})


if __name__ == '__main__':
    unittest.main()
