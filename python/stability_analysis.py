"""Stability of feature and group importance across seeds, splits and models.

Importance is model-agnostic permutation importance (mean increase in row
log-loss when one feature is permuted), computed for every combination of
task x model family x locked split x model seed. Four sources of variation are
separated by only comparing runs that differ in exactly one factor:

* ``model_seed``   same split and model family, different ``random_state``;
* ``split``        same model family and model seed, different locked split;
* ``model``        same split and model seed, different algorithm;
* ``permutation``  different permutation repeats inside one fitted model, the
                   analogue of sampler randomness for CFI. Repeats on the same
                   test rows share the model and the data, so they are never
                   treated as independent samples: their agreement is reported
                   as a within-run noise floor, not as evidence.

Agreement is measured by Spearman correlation of the importance vectors and
Jaccard overlap of the top-10 features, at feature level (55 columns) and at
group level (nine groups, importance summed inside the group). Per-feature and
per-group means/SDs and bootstrap CIs over runs are written to
``results/stability/<out-name>/`` and summarised in ``report/STABILITY_REPORT.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from common import CONFIG, ROOT
from group_protocol import MODELS, TASKS, load_tasks, make_model
from split_manager import load_locked

TOP_K = 10
SOURCE_ORDER = ['model_seed', 'split', 'model', 'permutation']


def stratified_subset(y, size, seed):
    index = np.arange(len(y))
    if size >= len(y):
        return index
    selected, _ = train_test_split(index, train_size=size, stratify=y,
                                   random_state=seed)
    return np.sort(selected)


def row_loss(task, y, probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-15, 1 - 1e-15)
    if task.n_classes == 2:
        p = probability[:, 1]
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return -np.log(probability[np.arange(len(y)), y])


def permutation_matrix(model, X, y, task, repeats, seed):
    """(repeats, n_features) mean increase in row log-loss per permutation."""
    base = row_loss(task, y, model.predict_proba(X))
    rng = np.random.default_rng(seed)
    deltas = np.zeros((repeats, X.shape[1]))
    for r in range(repeats):
        for j in range(X.shape[1]):
            permuted = X.copy()
            permuted[:, j] = X[rng.permutation(len(X)), j]
            deltas[r, j] = float(np.mean(row_loss(task, y, model.predict_proba(permuted))
                                         - base))
    return deltas


def jaccard_top(values_a, values_b, k=TOP_K):
    top_a = set(np.argsort(-values_a)[:k])
    top_b = set(np.argsort(-values_b)[:k])
    return len(top_a & top_b) / len(top_a | top_b)


def spearman(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else np.nan


def collect(args):
    tasks = load_tasks(args.tasks)
    rows = []
    for task_name, task in tasks.items():
        X = task.frame[task.features].to_numpy(dtype=np.float64)
        for split_seed in args.split_seeds:
            train, test = load_locked(args.split_version, task_name, split_seed)
            sample = test[stratified_subset(task.y[test], args.test_samples, split_seed)]
            X_test = X[sample]
            y_test = task.y[sample]
            for model_name in args.models:
                for model_seed in args.model_seeds:
                    start = time.time()
                    model = make_model(model_name, task, model_seed)
                    model.fit(X[train], task.y[train])
                    deltas = permutation_matrix(
                        model, X_test, y_test, task, args.repeats,
                        seed=[split_seed, model_seed, args.repeats])
                    for repeat in range(args.repeats):
                        for j, feature in enumerate(task.features):
                            rows.append(dict(task=task_name, model=model_name,
                                             split_seed=split_seed,
                                             model_seed=model_seed, repeat=repeat,
                                             feature=feature,
                                             delta_log_loss=float(deltas[repeat, j])))
                    print(f'[stability] {task_name}/{model_name}/split {split_seed}/'
                          f'seed {model_seed} in {time.time() - start:.0f}s', flush=True)
    return pd.DataFrame(rows)


def run_level(raw, level='feature'):
    if level == 'group':
        raw = raw.assign(feature=raw.feature.str.split('.').str[0])
        raw = (raw.groupby(['task', 'model', 'split_seed', 'model_seed', 'repeat',
                            'feature'], as_index=False).delta_log_loss.sum())
    return (raw.groupby(['task', 'model', 'split_seed', 'model_seed', 'feature'])
            .delta_log_loss.mean().unstack('feature'))


def pair_table(raw):
    frames = []
    for level in ('feature', 'group'):
        matrix = run_level(raw, level)
        records = []
        for task, block in matrix.groupby(level=0):
            block = block.droplevel(0)
            labels = list(block.index)
            arrays = {label: block.loc[label].to_numpy() for label in labels}
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    model_a, split_a, seed_a = labels[i]
                    model_b, split_b, seed_b = labels[j]
                    if split_a == split_b and seed_a == seed_b and model_a != model_b:
                        source = 'model'
                    elif model_a == model_b and seed_a == seed_b and split_a != split_b:
                        source = 'split'
                    elif (model_a == model_b and split_a == split_b
                          and seed_a != seed_b):
                        source = 'model_seed'
                    else:
                        continue
                    records.append(dict(task=task, level=level, source=source,
                                        spearman=spearman(arrays[labels[i]],
                                                          arrays[labels[j]]),
                                        jaccard_top10=jaccard_top(arrays[labels[i]],
                                                                  arrays[labels[j]])))
        frames.append(pd.DataFrame(records))

    # Within-run permutation agreement is the noise floor, not an independent
    # sample: all pairs of repeats inside one fitted model.
    pivot = raw.pivot_table(index=['task', 'model', 'split_seed', 'model_seed',
                                   'repeat'],
                            columns='feature', values='delta_log_loss')
    records = []
    for (task, _, _, _), block in pivot.groupby(level=[0, 1, 2, 3]):
        values = block.droplevel([0, 1, 2, 3]).to_numpy()
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                records.append(dict(task=task, level='feature',
                                    source='permutation',
                                    spearman=spearman(values[i], values[j]),
                                    jaccard_top10=jaccard_top(values[i], values[j])))
    frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True)


def stability_table(raw, level, bootstrap, seed):
    matrix = run_level(raw, level)
    ranks = matrix.rank(axis=1, ascending=False)
    rng = np.random.default_rng(seed)
    records = []
    for task, block in matrix.groupby(level=0):
        rank_block = ranks.loc[block.index]
        for name in block.columns:
            values = block[name].to_numpy()
            run_ranks = rank_block[name].to_numpy()
            boots = [np.mean(rng.choice(run_ranks, len(run_ranks), replace=True))
                     for _ in range(bootstrap)]
            records.append(dict(task=task, name=name,
                                importance_mean=float(values.mean()),
                                importance_sd=float(values.std(ddof=1)),
                                rank_mean=float(run_ranks.mean()),
                                rank_sd=float(run_ranks.std(ddof=1)),
                                ci_lo=float(np.percentile(boots, 2.5)),
                                ci_hi=float(np.percentile(boots, 97.5))))
    return pd.DataFrame(records)


def summaries(raw, args):
    pairs = pair_table(raw)
    pairs.to_csv(OUT / 'stability_pairs.csv', index=False)
    summary = (pairs.groupby(['task', 'level', 'source'])
               .agg(n_pairs=('spearman', 'size'),
                    spearman_mean=('spearman', 'mean'),
                    spearman_sd=('spearman', 'std'),
                    jaccard_mean=('jaccard_top10', 'mean'))
               .reset_index())
    summary['source'] = pd.Categorical(summary.source, SOURCE_ORDER, ordered=True)
    summary = summary.sort_values(['task', 'level', 'source'])
    summary.to_csv(OUT / 'stability_summary.csv', index=False)

    features = (stability_table(raw, 'feature', args.bootstrap, args.bootstrap_seed)
                .rename(columns={'name': 'feature'}))
    groups = (stability_table(raw, 'group', args.bootstrap, args.bootstrap_seed)
              .rename(columns={'name': 'group'}))
    features.to_csv(OUT / 'stability_features.csv', index=False)
    groups.to_csv(OUT / 'stability_groups.csv', index=False)
    return summary, features, groups


def figures(groups):
    for task, block in groups.groupby('task'):
        block = block.sort_values('rank_mean')
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.barh(block['group'], block.rank_mean, xerr=block.rank_sd, color='#2c6fbb')
        ax.set(xlabel='Mean rank across runs (1 = most important), ± SD',
               title=f'Group importance rank stability — {task}')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'group_rank_stability_{task}.png', dpi=160)
        plt.close(fig)


def write_report(summary, features, groups, raw, args):
    lines = ['# Độ ổn định của XAI\n']
    lines.append('Sinh bởi `python/stability_analysis.py`. '
                 'Permutation importance (tăng log-loss khi hoán vị từng feature), '
                 'không dùng sampler.\n')
    lines.append('## 1. Thiết kế\n')
    lines.append(f'- Task: {", ".join(args.tasks)}; model: {", ".join(args.models)}; '
                 f'split seeds: {args.split_seeds}; model seeds: {args.model_seeds}; '
                 f'{args.repeats} lần hoán vị, {args.test_samples} hàng test mỗi run.')
    lines.append(f'- Chia dữ liệu: `{args.split_version}` (source-aware, khóa sẵn); '
                 'không có hàng test nào dùng để fit hay chọn tham số.')
    lines.append('- Bốn nguồn biến thiên được tách bằng cách chỉ so các run khác nhau '
                 'đúng một yếu tố: model seed, split, model family, và hoán vị trong cùng '
                 'một model (noise floor, **không** phải mẫu độc lập).\n')
    lines.append('## 2. Đồng thuận hạng theo nguồn\n')
    display = summary.copy()
    display.columns = ['task', 'cấp', 'nguồn', 'số cặp', 'Spearman TB', 'SD',
                       'Jaccard top-10']
    lines.append(table_text(display.round(4)) + '\n')
    lines.append('## 3. Hạng theo nhóm feature\n')
    display = groups[['task', 'group', 'importance_mean', 'importance_sd',
                      'rank_mean', 'rank_sd', 'ci_lo', 'ci_hi']].copy()
    display = display.sort_values(['task', 'rank_mean'])
    display.columns = ['task', 'nhóm', 'importance TB', 'SD importance',
                       'hạng TB', 'SD hạng', 'CI 2.5%', 'CI 97.5%']
    lines.append(table_text(display.round(4)) + '\n')
    for task in groups.task.unique():
        lines.append(f'![Hạng nhóm {task}](../results/stability/{args.out_name}/figures/'
                     f'group_rank_stability_{task}.png)\n')
    lines.append('## 4. Cách đọc\n')
    lines.append('- Nguồn `permutation` là sàn nhiễu: nếu đồng thuận giữa các nguồn khác '
                 'không cao hơn sàn này, khác biệt hạng giữa run là nhiễu hoán vị.')
    lines.append('- CI bootstrap lấy mẫu lại **run**, không lấy mẫu lại hàng hay feature; '
                 'các lần hoán vị trong cùng run không được coi là quan sát độc lập.')
    lines.append('- Ổn định hạng không đồng nghĩa với đúng nhân quả; đây là kiểm tra độ '
                 'lặp lại của giải thích, không phải kiểm định ý nghĩa.\n')
    (ROOT / 'report' / 'STABILITY_REPORT.md').write_text('\n'.join(lines))


def table_text(frame):
    from generate_report import table
    return table(frame)


def write_manifest(args):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/stability_analysis.py',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        tasks=args.tasks, models=args.models, split_seeds=args.split_seeds,
        model_seeds=args.model_seeds, repeats=args.repeats,
        test_samples=args.test_samples, split_version=args.split_version,
        importance='permutation importance (mean row log-loss increase)',
        top_k=TOP_K, bootstrap=args.bootstrap,
        note='Permutation repeats share model and test rows and are not '
             'independent samples; they only define the noise floor.')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def run(args):
    global OUT
    OUT = ROOT / 'results' / 'stability' / args.out_name
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    raw = collect(args)
    raw.to_csv(OUT / 'stability_runs.csv', index=False)
    summary, features, groups = summaries(raw, args)
    figures(groups)
    write_report(summary, features, groups, raw, args)
    write_manifest(args)
    print(summary.to_string(index=False), flush=True)
    print(f'Stability artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--split-seeds', default='42,43,44')
    parser.add_argument('--model-seeds', default='0,1,2')
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--test-samples', type=int, default=800)
    parser.add_argument('--split-version', default='source_aware_v1')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    parser.add_argument('--out-name', default='stability_v1')
    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    args.models = [m.strip() for m in args.models.split(',') if m.strip()]
    args.split_seeds = [int(s) for s in args.split_seeds.split(',') if s.strip()]
    args.model_seeds = [int(s) for s in args.model_seeds.split(',') if s.strip()]
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        parser.error(f'unknown tasks: {sorted(unknown)}')
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f'unknown models: {sorted(unknown)}')
    run(args)


if __name__ == '__main__':
    main()
