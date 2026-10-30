"""Aggregation contract for Group Ablation: controls must be matched by group."""
import sys
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from group_protocol import score
from run_group_ablation import recall_deltas, summarise


def deltas_frame():
    rows = []
    groups = {'small': (1, 0.100), 'large': (5, 0.020)}
    control_deltas = {'small': [0.001, 0.002, 0.003],
                      'large': [0.500, 0.600, 0.700]}
    def metrics(log_loss, sign=-1):
        return {'delta_log_loss': log_loss, 'delta_accuracy': sign * log_loss,
                'delta_f1': sign * log_loss, 'delta_macro_f1': sign * log_loss,
                'delta_roc_auc': sign * log_loss, 'delta_pr_auc': sign * log_loss,
                'delta_balanced_accuracy': sign * log_loss}

    for group, (size, effect) in groups.items():
        rows.append(dict(task='toy', model='m', seed=42, condition='drop_group',
                         group=group, control_index=-1, n_features=55 - size,
                         **metrics(effect)))
        for k, value in enumerate(control_deltas[group]):
            rows.append(dict(task='toy', model='m', seed=42,
                             condition='random_drop', group=group,
                             control_index=k, n_features=55 - size,
                             **metrics(value)))
    return pd.DataFrame(rows)


class MatchedControlsTest(unittest.TestCase):
    def setUp(self):
        self.summary, _ = summarise(deltas_frame(), Namespace(controls=3))

    def test_null_is_group_specific_not_pooled(self):
        small = self.summary[self.summary.group == 'small'].iloc[0]
        large = self.summary[self.summary.group == 'large'].iloc[0]
        self.assertAlmostEqual(float(small.log_loss_null_mean), 0.002)
        self.assertAlmostEqual(float(large.log_loss_null_mean), 0.600)
        self.assertEqual(int(small.n_controls), 3)

    def test_empirical_p_uses_matched_controls(self):
        small = self.summary[self.summary.group == 'small'].iloc[0]
        # None of the three matched controls (max 0.003) reaches 0.100;
        # a pooled null would have counted the three large-group controls.
        self.assertAlmostEqual(float(small.log_loss_p_empirical), 0.25)

    def test_rank_orders_by_effect(self):
        small = self.summary[self.summary.group == 'small'].iloc[0]
        large = self.summary[self.summary.group == 'large'].iloc[0]
        self.assertEqual(int(small.rank_log_loss), 1)
        self.assertEqual(int(large.rank_log_loss), 2)


class RecallDeltaTest(unittest.TestCase):
    def test_recall_delta_is_condition_minus_full(self):
        runs = pd.DataFrame([
            dict(task='toy', model='m', seed=1, condition='full', group='__all__',
                 control_index=-1, per_class_recall=json.dumps({'a': 0.8, 'b': 0.6})),
            dict(task='toy', model='m', seed=1, condition='drop_group', group='g',
                 control_index=-1, per_class_recall=json.dumps({'a': 0.5, 'b': 0.7})),
        ])
        table = recall_deltas(runs).set_index('label')
        self.assertAlmostEqual(float(table.loc['a', 'delta_recall']), -0.3)
        self.assertAlmostEqual(float(table.loc['b', 'delta_recall']), 0.1)


class MulticlassScoreTest(unittest.TestCase):
    def test_ovr_aucs_are_finite(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(90, 4))
        y = np.repeat([0, 1, 2], 30)
        X[y == 1, 0] += 1.5
        X[y == 2, 1] += 1.5
        model = LogisticRegression(max_iter=500).fit(X, y)
        task = SimpleNamespace(n_classes=3, classes=['a', 'b', 'c'])
        result = score(model, X, y, task)
        for key in ('roc_auc', 'pr_auc', 'macro_f1', 'log_loss'):
            self.assertTrue(np.isfinite(result[key]), key)
        self.assertGreater(result['roc_auc'], 0.5)
        self.assertGreater(result['pr_auc'], 0.33)


if __name__ == '__main__':
    unittest.main()
