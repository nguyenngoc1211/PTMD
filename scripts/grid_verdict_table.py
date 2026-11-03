#!/usr/bin/env python3
"""Rebuild every count quoted in the manuscript from the artifacts on disk.

The paper reports three kinds of number that are easy to get wrong by hand: the
per-cell threshold verdicts (7/12, 8/13, 10/17), the rank-1 tally, and the Spearman
rho table. All three are recomputed here from `permutation_summary.csv` and
`summary.csv`, so a reviewer can re-derive them with one command. Benjamini
Hochberg runs inside the nine groups of each cell, on the log-loss empirical p.

Rho uses `exact_spearman.compare`, which enumerates all 9! orderings, never
scipy's t-approximation: the reliance cross-model pair sits at rho 0.683, where the
two rules disagree (exact 0.05032 versus asymptotic 0.0424).

Run from the repository root: `python3 scripts/grid_verdict_table.py`.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_spearman import compare  # noqa: E402

ROOT = os.environ.get('RAM_EVID_RESULTS', 'results/group_experiments')

# cell key: (estimand, model, task, seed) -> (artifact directory, table, columns)
INTERVENTIONAL = {
    ('reliance', 'xgboost', 'family16', 42): ('importance_v3', 'permutation_summary.csv',
                                              'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'xgboost', 'family16', 43): ('importance_v3_s43', 'permutation_summary.csv',
                                              'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'xgboost', 'family16', 44): ('importance_v3_s44', 'permutation_summary.csv',
                                              'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'logreg', 'family16', 42): ('importance_v3_logreg', 'permutation_summary.csv',
                                             'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'logreg', 'family16', 43): ('importance_v3_logreg_s43', 'permutation_summary.csv',
                                             'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'logreg', 'family16', 44): ('importance_v3_logreg_s44', 'permutation_summary.csv',
                                             'delta_log_loss_mean', 'p_empirical'),
    ('utility', 'xgboost', 'family16', 42): ('ablation_v4', 'summary.csv',
                                             'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'xgboost', 'family16', 43): ('ablation_xgb_s43', 'summary.csv',
                                             'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'xgboost', 'family16', 44): ('ablation_xgb_s44', 'summary.csv',
                                             'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'logreg', 'family16', 42): ('ablation_logreg_v4', 'summary.csv',
                                            'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'logreg', 'family16', 43): ('ablation_logreg_s43', 'summary.csv',
                                            'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'logreg', 'family16', 44): ('ablation_logreg_s44', 'summary.csv',
                                            'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'histgb', 'family16', 42): ('ablation_histgb_v4', 'summary.csv',
                                            'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('reliance', 'histgb', 'family16', 42): ('importance_v3_histgb', 'permutation_summary.csv',
                                             'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'xgboost', 'binary', 42): ('importance_v3_binary_type4', 'permutation_summary.csv',
                                            'delta_log_loss_mean', 'p_empirical'),
    ('reliance', 'xgboost', 'type4', 42): ('importance_v3_binary_type4', 'permutation_summary.csv',
                                           'delta_log_loss_mean', 'p_empirical'),
    ('utility', 'xgboost', 'binary', 42): ('ablation_v4_binary_type4', 'summary.csv',
                                           'log_loss_delta_mean', 'log_loss_p_empirical'),
    ('utility', 'xgboost', 'type4', 42): ('ablation_v4_binary_type4', 'summary.csv',
                                          'log_loss_delta_mean', 'log_loss_p_empirical'),
}

SHAPLEY_DIRS = ['importance_shapley', 'importance_shapley_s43_44', 'importance_shapley_logreg',
                'importance_shapley_logreg_s43_44', 'importance_shapley_s42_type4']

# Cells whose artifact is still being written. A missing file is expected here and only
# here, so a typo in the path of a finished cell still raises instead of quietly shrinking
# every tally. Empty this set once the run lands and the tallies become final.
PENDING = {('reliance', 'histgb', 'family16', 42)}

WIDTH = {'modules': 1, 'dlllist': 2, 'callbacks': 3, 'malfind': 4, 'pslist': 5,
         'ldrmodules': 6, 'svcscan': 7, 'handles': 13, 'psxview': 14}


def bh(pvals):
    """Benjamini-Hochberg adjusted p, step-up enforced, inside one cell."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    out, prev = [0.0] * n, 1.0
    for pos, i in enumerate(reversed(order)):
        prev = min(prev, pvals[i] * n / (n - pos))
        out[i] = prev
    return out


def load_table(folder, name, task, model, expected=9, allow_missing=False):
    path = os.path.join(ROOT, folder, name)
    if not os.path.exists(path):
        if allow_missing:
            return None
        raise SystemExit(f'{path}: artifact missing and the cell is not in PENDING')
    df = pd.read_csv(path)
    df = df[df['task'] == task]
    if 'model' in df.columns:
        df = df[df['model'] == model]
    if len(df) != expected:
        raise SystemExit(f'{path}: expected {expected} rows for {task}/{model}, got {len(df)}')
    if df['group'].duplicated().any():
        raise SystemExit(f'{path}: duplicated group index, refusing a plausible rho')
    return df.set_index('group')


def label(key):
    return f'{key[0]:9s} {key[1]:8s} {key[2]:8s} s{key[3]}'


def prereg(scores, qmap, rmap, alpha):
    """H1-H5 of REPRODUCIBILITY 4.20, cut points written before the artifact existed.

    Kept in the same file as the tallies so the verdict rules cannot be edited after the
    numbers are on disk without the diff being visible.
    """
    key = ('reliance', 'histgb', 'family16', 42)
    print('\npre-registered H1-H5, histgb reliance family16 s42')
    if key not in scores:
        print('  PENDING - artifact absent, so no verdict is emitted')
        return
    q, r, s = qmap[key], rmap[key], scores[key]
    order = list(s.sort_values(ascending=False).index)
    fam = [k for k in qmap if k[2] == 'family16']
    fam_pass = sum(1 for k in fam if qmap[k]['handles'] < alpha)
    checks = []
    checks.append(('H1 callbacks ranks 1, handles ranks 2-3',
                   order[0] == 'callbacks' and r['handles'] in (2, 3),
                   f'order={">".join(order[:3])} handles_rank={int(r["handles"])}'))
    checks.append(('H2 callbacks q_BH < 0.05', q['callbacks'] < alpha,
                   f'q={q["callbacks"]:.4f} delta={s["callbacks"]:+.4f}'))
    if ('reliance', 'xgboost', 'family16', 42) in scores:
        rr = compare('histgb:s42', s, 'xgboost:s42', scores[('reliance', 'xgboost', 'family16', 42)])
        checks.append(('H3 rho vs xgboost in 0.3-0.7 and exact p > 0.05',
                       0.3 <= rr['rho'] <= 0.7 and rr['p_exact_all_orderings'] > 0.05,
                       f"rho={rr['rho']:.4f} exact_p={rr['p_exact_all_orderings']:.6f}"))
    twin = ('utility', 'histgb', 'family16', 42)
    if twin in scores:
        rr = compare('utility:s42', scores[twin], 'reliance:s42', s)
        checks.append(('H4 within-cell utility~reliance rho >= 0.8', rr['rho'] >= 0.8,
                       f"rho={rr['rho']:.4f} exact_p={rr['p_exact_all_orderings']:.6f}"))
    checks.append(('H5 handles passes threshold', q['handles'] < alpha,
                   f'q={q["handles"]:.4f} rank={int(r["handles"])}; '
                   f'family16 handles tally {fam_pass}/{len(fam)}'))
    for name, ok, detail in checks:
        print(f'  {"PASS" if ok else "FAIL"}  {name}  [{detail}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--focus', default='handles', help='group whose verdicts are tallied')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    scores, qmap, rmap = {}, {}, {}
    absent = []
    for key, (folder, name, dcol, pcol) in INTERVENTIONAL.items():
        df = load_table(folder, name, key[2], key[1], allow_missing=key in PENDING)
        if df is None:
            absent.append(key)
            continue
        scores[key] = df[dcol]
        qmap[key] = dict(zip(df.index, bh([float(p) for p in df[pcol]])))
        rmap[key] = df[dcol].rank(ascending=False, method='min').astype(int)
    for key in absent:
        print(f'  {label(key)}  PENDING - artifact not on disk, absent from every tally below')

    for folder in SHAPLEY_DIRS:
        df = pd.read_csv(os.path.join(ROOT, folder, 'shapley_groups.csv'))
        for (task, model, seed), g in df.groupby(['task', 'model', 'seed']):
            scores[('attribution', model, task, int(seed))] = g.set_index('group')['mean_abs_phi']

    print(f'cell verdicts  (BH over 9 groups on log-loss, alpha={args.alpha}, focus={args.focus})')
    tally = dict(total=0, passed=0, rank1=0, grid=0, grid_passed=0, fam=0, fam_passed=0)
    for key in sorted(scores, key=lambda k: (k[0], k[1], k[2], k[3])):
        if key not in qmap:
            continue
        q, r = qmap[key], rmap[key]
        winners = sorted(g for g in q if q[g] < args.alpha)
        tally['total'] += 1
        tally['rank1'] += int(r[args.focus] == 1)
        tally['passed'] += int(q[args.focus] < args.alpha)
        if key[2] == 'family16':
            tally['fam'] += 1
            tally['fam_passed'] += int(q[args.focus] < args.alpha)
        if key[2] == 'family16' and key[1] in ('xgboost', 'logreg'):
            tally['grid'] += 1
            tally['grid_passed'] += int(q[args.focus] < args.alpha)
        print(f'  {label(key)}  rank {int(r[args.focus])}  q={q[args.focus]:.4f}  '
              f'{"PASS" if q[args.focus] < args.alpha else "fail"}  '
              f'delta={scores[key][args.focus]:+.4f}  winners={winners or "-"}')
    print(f"\n{args.focus}: {tally['passed']}/{tally['total']} cells pass threshold, "
          f"{tally['rank1']}/{tally['total']} rank 1, locked 2x2x3 grid "
          f"{tally['grid_passed']}/{tally['grid']}, all family16 interventional cells "
          f"{tally['fam_passed']}/{tally['fam']} (only the last fraction counts extension "
          f"cells; a cell whose artifact is missing is absent from all four fractions)")

    print('\nrho: same task, exactly one dimension changed at a time')
    for estimand in ('reliance', 'utility', 'attribution'):
        cells = sorted((k for k in scores if k[0] == estimand),
                       key=lambda k: (k[2], k[3], k[1]))
        for a, b in combinations(cells, 2):
            same_model, same_seed, same_task = (a[1] == b[1], a[3] == b[3], a[2] == b[2])
            if same_task and same_model and same_seed:
                continue
            if same_task and same_model:
                kind = 'cross-seed'
            elif same_task and same_seed:
                kind = 'cross-model'
            elif same_model and same_seed:
                kind = 'cross-task'
            else:
                continue
            r = compare(f'{a[1]}/{a[3]}', scores[a], f'{b[1]}/{b[3]}', scores[b])
            print(f'  {estimand:11s} {kind:11s} {a[2]:8s} '
                  f'{a[1]}:s{a[3]}~{b[1]}:s{b[3]} '
                  f'rho={r["rho"]:.4f} exact_p={r["p_exact_all_orderings"]:.6f} '
                  f'top1={"yes" if r["top1_match"] else "no "} maxshift={r["max_rank_shift"]}')

    print('\nestimand agreement inside a cell')
    for key in sorted(k for k in qmap if k[2] == 'family16' and k[0] == 'utility'):
        twin = ('reliance' if key[0] == 'utility' else 'utility', key[1], key[2], key[3])
        if twin in scores:
            r = compare(label(key), scores[key], 'vs', scores[twin])
            print(f'  {key[1]:8s} s{key[3]}  utility~reliance rho={r["rho"]:.4f} '
                  f'exact_p={r["p_exact_all_orderings"]:.6f} maxshift={r["max_rank_shift"]}')

    print('\nblock width versus control spread (why threshold verdicts are width-loaded)')
    for key in sorted(k for k in qmap if k[2] == 'family16'):
        folder, name, _, _ = INTERVENTIONAL[key]
        df = load_table(folder, name, key[2], key[1])
        null_col = 'null_sd' if key[0] == 'reliance' else 'log_loss_null_sd'
        width = pd.Series({g: WIDTH[g] for g in df.index})
        r = compare('width', width, 'null_sd', df[null_col].astype(float))
        print(f'  {label(key)}  rho={r["rho"]:.4f} exact_p={r["p_exact_all_orderings"]:.6f}')

    prereg(scores, qmap, rmap, args.alpha)


if __name__ == '__main__':
    main()
