"""Compare group importance across ablation, group permutation, group CFI and
feature-wise permutation.

All comparisons are on the same task/split where possible (family16,
``source_aware_v1``); the model defaults to XGBoost because that is the model
shared by every method. The feature-wise baseline comes from
``stability_analysis.py`` (permutation of one feature at a time, summed inside
its group) and is *not* the same estimand as joint block permutation; the gap
between them is part of the result.

CFI is reported twice: over all nine groups and over the groups that pass the
sampler gate. Spearman correlations involving CFI are only interpretable in the
gate-pass subset; the all-group value is written for transparency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CONFIG, ROOT
from group_protocol import GROUPS

METHODS = ['ablation', 'permutation', 'cfi', 'featurewise']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ablation_scores(path):
    path = Path(path)
    source = path / 'summary_matched.csv'
    if not source.exists():
        source = path / 'summary.csv'
    table = pd.read_csv(source)
    table = table[table.model == 'xgboost']
    return table.set_index('group')[['log_loss_delta_mean', 'log_loss_p_empirical']], source


def permutation_scores(path, model):
    path = Path(path)
    runs = pd.read_csv(path / 'permutation_runs.csv')
    runs = runs[runs.model == model]
    rows = []
    for group, block in runs.groupby('group'):
        observed = block[block.condition == 'permute_group'].delta_log_loss
        null = block[block.condition == 'permute_random'].delta_log_loss
        effect = float(observed.mean())
        rows.append(dict(group=group, permutation_delta_log_loss=effect,
                         permutation_p_empirical=float(
                             (np.sum(null.to_numpy() >= effect) + 1) / (len(null) + 1))))
    return pd.DataFrame(rows).set_index('group')


def cfi_scores(path):
    table = pd.read_csv(Path(path) / 'group_cfi.csv')
    table = table[table.task == 'family16'] if 'task' in table else table
    return table.set_index('group')[['conditional_importance', 'gate_pass', 'gate_reasons']]


def featurewise_scores(path, model):
    table = pd.read_csv(Path(path) / 'stability_runs.csv')
    table = table[table.model == model].copy()
    table['group'] = table.feature.str.split('.').str[0]
    per_repeat = (table.groupby(['split_seed', 'model_seed', 'repeat', 'group'],
                                as_index=False).delta_log_loss.sum())
    means = per_repeat.groupby('group').delta_log_loss.mean()
    return means.rename('featurewise_delta_log_loss').to_frame()


def rank_table(scores):
    frame = pd.DataFrame(index=scores.index)
    for method in METHODS:
        column = f'{method}_score'
        frame[method] = scores[column]
        frame[f'{method}_rank'] = frame[method].rank(ascending=False, method='average')
    return frame


def correlation_table(ranks, subsets):
    rows = []
    for scope, groups in subsets.items():
        block = ranks.loc[[g for g in groups if g in ranks.index]]
        methods = [m for m in METHODS if block[m].notna().sum() >= 3]
        for i, first in enumerate(methods):
            for second in methods[i + 1:]:
                pair = block[[first, second]].dropna()
                if len(pair) < 3:
                    continue
                rho = spearmanr(pair[first], pair[second]).statistic
                rows.append(dict(scope=scope, method_a=first, method_b=second,
                                 n_groups=len(pair), spearman=float(rho)))
    return pd.DataFrame(rows)


def run(args):
    out = ROOT / 'results' / 'group_experiments' / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    ablation, ablation_source = ablation_scores(ROOT / 'results' / 'group_experiments' /
                                               args.ablation)
    permutation = permutation_scores(ROOT / 'results' / 'group_experiments' /
                                     args.importance, args.model)
    cfi = cfi_scores(ROOT / 'results' / 'group_experiments' / args.cfi)
    featurewise = featurewise_scores(ROOT / 'results' / 'stability' / args.stability,
                                     args.model)

    scores = pd.DataFrame(index=sorted(GROUPS))
    scores['ablation_score'] = ablation['log_loss_delta_mean']
    scores['ablation_p'] = ablation['log_loss_p_empirical']
    scores['permutation_score'] = permutation['permutation_delta_log_loss']
    scores['permutation_p'] = permutation['permutation_p_empirical']
    scores['cfi_score'] = cfi['conditional_importance']
    scores['cfi_gate_pass'] = cfi['gate_pass']
    scores['cfi_gate_reasons'] = cfi['gate_reasons']
    scores['featurewise_score'] = featurewise['featurewise_delta_log_loss']
    scores.to_csv(out / 'group_scores.csv')

    ranks = rank_table(scores)
    ranks.to_csv(out / 'group_ranks.csv')

    gate_pass = [g for g in GROUPS if bool(cfi['gate_pass'].get(g, False))]
    correlations = correlation_table(ranks, {'all_groups': GROUPS,
                                             'cfi_gate_pass': gate_pass})
    correlations.to_csv(out / 'spearman.csv', index=False)

    manifest = dict(
        script='python/compare_group_methods.py',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=CONFIG['data'],
        dataset_sha256=sha256(ROOT / CONFIG['data']),
        task='family16', split_version='source_aware_v1', model=args.model,
        ablation_source=str(ablation_source.relative_to(ROOT)),
        importance_dir=args.importance, cfi_dir=args.cfi, stability_dir=args.stability,
        cfi_gate_pass=gate_pass,
        note='Feature-wise scores are single-feature permutations summed inside a '
             'group, not joint block permutations; they answer a different estimand.',
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(scores.round(4).to_string())
    print('\n' + correlations.round(3).to_string(index=False))
    print(f'\nComparison written to {out.relative_to(ROOT)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation', default='ablation_v2')
    parser.add_argument('--importance', default='importance_v2')
    parser.add_argument('--cfi', default='cfi_v3')
    parser.add_argument('--stability', default='stability_v1')
    parser.add_argument('--model', default='xgboost')
    parser.add_argument('--out-name', default='comparison_v1')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
