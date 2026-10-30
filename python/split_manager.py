"""Source-aware splits that keep every specimen on one side of the boundary.

Why this module exists
----------------------
``group_protocol.TaskData.split`` reproduces the verified pipeline: a random
stratified holdout on rows. The audit (``TECHNICAL_AUDIT.md`` §3) measured that
split and found 99.8% of test malware specimens also present in train, because
CIC-MalMem-2022 contributes ~10 memory snapshots per specimen and the random
split scatters them across both sides. Binary F1 = 1.0000 is therefore partly
an artefact of near-duplicate leakage, and every leave-one-group-out delta
measured on that split is noise on a saturated task.

This module splits on the *source unit* instead of the row:

* malware rows group by the specimen hash parsed out of ``Category``;
* benign rows have no specimen hash, so each is its own singleton unit;
* a unit is never divided between train and test.

Stratification is applied to unit-level strata. A unit whose rows carry more
than one label would be ambiguous to stratify, so it becomes its own ``mixed``
stratum rather than being assigned to one class. In this dataset 19 units are
mixed (the same 19 specimens that produce the family-label conflicts) and 0
units straddle the binary label, so the binary task is exactly stratified and
the multiclass tasks deviate by at most those 19 units.

Locked splits
-------------
``materialise`` writes each split to ``results/splits/<version>/`` together with
a manifest holding the dataset SHA-256, the seed and the resulting row counts.
Downstream experiments read those files rather than recomputing, so a split can
be locked and reused across every phase.

CLI
---
    python split_manager.py --version source_aware_v1 --seeds 42,43,44
    python split_manager.py --verify source_aware_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CONFIG, ROOT
from group_protocol import TASKS, build_task, load_base_frame

MIXED = 'mixed'
SPLIT_ROOT = ROOT / 'results' / 'splits'

# ``strategy`` values accepted by TaskData.split / the experiment scripts.
STRATEGIES = ('random', 'source_aware')


def unit_labels(units, y):
    """Stratum key per unit: its single label, or MIXED when it has several.

    Returns (unique unit ids, matching stratum keys, rows per unit).
    """
    units = np.asarray(units, dtype=object)
    y = np.asarray(y)
    frame = pd.DataFrame({'unit': units, 'y': y})
    agg = frame.groupby('unit', sort=True)['y'].agg(
        lambda s: str(s.iloc[0]) if s.nunique() == 1 else MIXED)
    counts = frame.groupby('unit', sort=True).size()
    order = counts.index.to_numpy()
    return order, agg.loc[order].to_numpy(), counts.loc[order].to_numpy()


def source_aware_indices(units, y, test_size=None, seed=None):
    """Group-aware stratified holdout. Whole units only, never split rows.

    Deterministic for a given (units, y, seed). Returns sorted positional
    indices into the input arrays, matching the shape of
    ``group_protocol.TaskData.split``.
    """
    if test_size is None:
        test_size = CONFIG['test_size']
    if seed is None:
        seed = CONFIG['seed']
    units = np.asarray(units, dtype=object)
    y = np.asarray(y)
    if len(units) != len(y):
        raise ValueError('units and y must be the same length')
    if pd.isna(list(units)).any():
        raise ValueError('units must not contain missing values')

    unique_units, strata, sizes = unit_labels(units, y)
    position = {unit: i for i, unit in enumerate(unique_units)}
    unit_of_row = np.fromiter((position[u] for u in units), dtype=np.int64,
                              count=len(units))

    rng = np.random.default_rng(seed)
    held_out_units = []
    for stratum in pd.unique(strata):
        members = np.flatnonzero(strata == stratum)
        target_rows = round(sizes[members].sum() * test_size)
        order = rng.permutation(members)
        rows, taken = 0, []
        # Draw whole units until the stratum's row target is reached; the last
        # unit may overshoot by at most its own size (~10 rows per specimen).
        for k in order:
            if rows >= target_rows:
                break
            taken.append(k)
            rows += sizes[k]
        held_out_units.append(np.asarray(taken, dtype=np.int64))

    test_units = np.concatenate(held_out_units) if held_out_units else np.empty(0, np.int64)
    train_mask = np.ones(len(unique_units), dtype=bool)
    train_mask[test_units] = False
    test_rows = np.flatnonzero(~train_mask[unit_of_row])
    train_rows = np.flatnonzero(train_mask[unit_of_row])
    if np.intersect1d(train_rows, test_rows).size:
        raise AssertionError('a unit crossed the train/test boundary')
    return np.sort(train_rows), np.sort(test_rows)


def units_for(task):
    """Source unit per row of one task, as stable strings."""
    return task.frame['__unit'].astype(str).to_numpy()


def locked_path(version, task_name, seed, strategy='source_aware'):
    return SPLIT_ROOT / version / f'split_{task_name}_{seed}_{strategy}.joblib'


def materialise(version, tasks, seeds, strategies=('source_aware',), force=False):
    """Write locked split files for every (task, seed, strategy) combination."""
    frame, features = load_base_frame()
    records = []
    data_path = ROOT / CONFIG['data']
    for task_name in tasks:
        task = build_task(frame, features, task_name)
        for seed in seeds:
            for strategy in strategies:
                out = locked_path(version, task_name, seed, strategy)
                if out.exists() and not force:
                    payload = joblib.load(out)
                    if payload.get('strategy') != strategy:
                        raise ValueError(f'{out.name} is {payload["strategy"]}, asked {strategy}')
                    records.append(dict(payload['summary'], reused=True))
                    continue
                if strategy == 'source_aware':
                    train, test = source_aware_indices(units_for(task), task.y,
                                                       seed=seed)
                elif strategy == 'random':
                    train, test = task.split(seed)
                else:
                    raise ValueError(f'unknown strategy {strategy!r}')
                units = units_for(task)
                crossing = set(units[train]) & set(units[test])
                if strategy == 'source_aware' and crossing:
                    raise AssertionError(f'{task_name}/{seed}: {len(crossing)} units cross')
                out.parent.mkdir(parents=True, exist_ok=True)
                summary = dict(task=task_name, seed=int(seed), strategy=strategy,
                               version=version, n_rows=len(task.frame),
                               n_train=int(len(train)), n_test=int(len(test)),
                               test_fraction=round(len(test) / len(task.frame), 6),
                               n_units_train=len(set(units[train])),
                               n_units_test=len(set(units[test])),
                               units_crossing=len(crossing),
                               class_counts_train={str(c): int((task.y[train] == c).sum())
                                                   for c in range(task.n_classes)},
                               class_counts_test={str(c): int((task.y[test] == c).sum())
                                                  for c in range(task.n_classes)},
                               dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                               created=time.strftime('%Y-%m-%dT%H:%M:%S'))
                joblib.dump(dict(train_indices=train, test_indices=test,
                                 strategy=strategy, summary=summary), out)
                records.append(dict(summary, reused=False))
                print(f'[split] {version}/{task_name}/seed{seed}/{strategy}: '
                      f'train={len(train)} test={len(test)} '
                      f'units={summary["n_units_test"]} crossing={len(crossing)}', flush=True)
    manifest = dict(version=version, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                    dataset=str(CONFIG['data']),
                    dataset_sha256=records[0]['dataset_sha256'],
                    unit_definition='specimen hash from Category; benign rows are singletons',
                    splits=records)
    (SPLIT_ROOT / version / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(records).to_csv(SPLIT_ROOT / version / 'split_table.csv', index=False)
    return pd.DataFrame(records)


def load_locked(version, task_name, seed, strategy='source_aware'):
    path = locked_path(version, task_name, seed, strategy)
    if not path.exists():
        raise FileNotFoundError(
            f'no locked split {version}/{task_name}/{seed}; run '
            f'`python split_manager.py --version {version}` first')
    payload = joblib.load(path)
    return np.asarray(payload['train_indices']), np.asarray(payload['test_indices'])


def verify(version, tasks=None, seeds=None):
    """Re-derive every locked split and confirm the files match the data."""
    frame, features = load_base_frame()
    problems, rows = [], []
    manifest_path = SPLIT_ROOT / version / 'split_manifest.json'
    if not manifest_path.exists():
        return [f'missing manifest {manifest_path.relative_to(ROOT)}'], []
    manifest = json.loads(manifest_path.read_text())
    wanted = {(r['task'], r['seed'], r['strategy']) for r in manifest['splits']}
    for task_name in (tasks or TASKS):
        task = build_task(frame, features, task_name)
        units = units_for(task)
        for (t, seed, strategy) in sorted(wanted):
            if t != task_name or strategy != 'source_aware':
                continue
            if seeds and seed not in seeds:
                continue
            train, test = load_locked(version, t, seed, 'source_aware')
            crossing = set(units[train]) & set(units[test])
            # A locked split must be exactly reproducible from data + seed.
            recomputed = source_aware_indices(units, task.y, seed=seed)
            same = (np.array_equal(train, recomputed[0])
                    and np.array_equal(test, recomputed[1]))
            if crossing:
                problems.append(f'{t}/{seed}: {len(crossing)} units cross')
            if not same:
                problems.append(f'{t}/{seed}: locked file does not match recomputation')
            rows.append(dict(task=t, seed=seed, units_crossing=len(crossing),
                             reproducible=same, n_test=len(test)))
    return problems, rows


def leakage_report(seeds=(42, 43, 44)):
    """Side-by-side leakage of the random split and the source-aware split."""
    frame, features = load_base_frame()
    rows = []
    for task_name in TASKS:
        task = build_task(frame, features, task_name)
        units = units_for(task)
        for seed in seeds:
            for strategy in ('random', 'source_aware'):
                if strategy == 'random':
                    train, test = task.split(seed)
                else:
                    train, test = source_aware_indices(units, task.y, seed=seed)
                sp_train = {u for u in units[train] if not u.startswith('__row_')}
                sp_test = {u for u in units[test] if not u.startswith('__row_')}
                shared = sp_train & sp_test
                leaked = int(np.isin(units[test], list(shared)).sum()) if shared else 0
                rows.append(dict(task=task_name, seed=seed, strategy=strategy,
                                 n_test=len(test), n_units_test=len(sp_test),
                                 units_also_in_train=len(shared),
                                 pct_units_leaked=round(100 * len(shared) / max(len(sp_test), 1), 2),
                                 test_rows_from_leaked_unit=leaked,
                                 pct_rows_leaked=round(100 * leaked / len(test), 2)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='source_aware_v1')
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--strategies', default='source_aware,random')
    parser.add_argument('--force', action='store_true',
                        help='rewrite an existing locked split (refused by default)')
    parser.add_argument('--verify', metavar='VERSION')
    parser.add_argument('--leakage-report', action='store_true')
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
    unknown = set(tasks) - set(TASKS)
    if unknown:
        parser.error(f'unknown tasks: {sorted(unknown)}')

    if args.verify:
        problems, rows = verify(args.verify, tasks, seeds)
        print(pd.DataFrame(rows).to_string(index=False))
        print('PROBLEMS:', problems or 'none')
        raise SystemExit(1 if problems else 0)
    if args.leakage_report:
        report = leakage_report(seeds)
        out = SPLIT_ROOT / 'leakage_comparison.csv'
        out.parent.mkdir(exist_ok=True)
        report.to_csv(out, index=False)
        print(report.to_string(index=False))
        print(f'-> {out.relative_to(ROOT)}')
        return
    table = materialise(args.version, tasks, seeds, strategies, args.force)
    print(table[['task', 'seed', 'strategy', 'n_train', 'n_test',
                 'n_units_test', 'units_crossing']].to_string(index=False))


if __name__ == '__main__':
    main()
