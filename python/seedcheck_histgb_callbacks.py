"""Seed replication for the HistGB callbacks effect seen at seed 42.

`ablation_histgb` reported a catastrophic drop in log-loss when `callbacks`
was removed, but with a single split seed. This script refits full vs
drop-callbacks for split seeds 42/43/44 under the same source-aware split and
model configuration, writing `seedcheck.csv` and a manifest so the claim can
be checked without rerunning the whole ablation grid.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT
from group_protocol import group_map, load_tasks, make_model, score

OUT = ROOT / 'results' / 'group_experiments' / 'histgb_callbacks_seedcheck'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    task = load_tasks(['family16'])['family16']
    X = task.frame[task.features].to_numpy(dtype=np.float64)
    mapping = group_map(task.features)
    kept = [f for f in task.features if f not in set(mapping['callbacks'])]
    rows = []
    for seed in (42, 43, 44):
        train, test = task.split(seed, strategy='source_aware')
        for condition, columns in (('full', list(task.features)),
                                   ('drop_callbacks', kept)):
            index = [task.features.index(c) for c in columns]
            model = make_model('histgb', task, seed)
            start = time.time()
            model.fit(X[np.ix_(train, index)], task.y[train])
            metrics = score(model, X[np.ix_(test, index)], task.y[test], task)
            rows.append(dict(task='family16', model='histgb', seed=seed,
                             condition=condition, n_features=len(columns),
                             fit_seconds=round(time.time() - start, 1), **metrics))
            print(f'seed {seed} {condition:<14} '
                  f'log_loss={metrics["log_loss"]:.4f} '
                  f'macro_f1={metrics["macro_f1"]:.4f}', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'seedcheck.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps({
        'script': 'python/seedcheck_histgb_callbacks.py',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dataset_sha256': hashlib.sha256(
            (ROOT / 'data/Obfuscated-MalMem2022.csv').read_bytes()).hexdigest(),
        'task': 'family16', 'model': 'histgb', 'split': 'source_aware',
        'split_version': 'source_aware_v1', 'seeds': [42, 43, 44],
        'note': 'Targeted seed replication for the callbacks effect in '
                'ablation_histgb (seed 42); full vs drop-callbacks only.',
    }, indent=2))
    print(table[['seed', 'condition', 'log_loss', 'macro_f1',
                 'balanced_accuracy', 'roc_auc']].to_string(index=False))


if __name__ == '__main__':
    main()
