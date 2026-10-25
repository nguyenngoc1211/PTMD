"""Leave-one-group-out retraining ablation with matched random controls.

Scientific question
-------------------
Does removing an entire Volatility plugin group *hurt* the model more than
removing an equal number of arbitrary features? Only the first case is evidence
that the group carries information the model actually uses.

Design
------
For every (task, model, seed):

1. Fit the full model on all 55 features -> reference metrics.
2. For each of the nine groups: delete the group's columns, refit, rescore.
   The drop is expressed in the metric's own direction, so a positive value
   always means "the model got worse".
3. For each group: delete the same number of features drawn at random from the
   *complement* of that group (``--controls`` draws). The control never
   partially ablates the group under test, so it answers "would dropping k
   unrelated features hurt as much?".
4. Optional substitution test (``--substitution``): for the pair of groups with
   the strongest cross-group correlation, drop g, drop its partner p, and drop
   g+p. If dropping both is no worse than dropping the stronger one alone, the
   two groups are substitutable rather than additive.

Controls
--------
* Identical hyper-parameters for every condition: the tuning budget is
  structurally zero and equal, so no condition can win by being tuned harder.
* Same split, same rows, same preprocessing for all conditions of a
  (task, model, seed) triple.
* Binary F1 is saturated at 1.0 on this benchmark, so ``log_loss`` is the
  primary metric; F1/PR-AUC are reported for completeness.

Outputs (new tree, nothing existing is overwritten):
    results/group_experiments/ablation/runs.csv
    results/group_experiments/ablation/deltas.csv
    results/group_experiments/ablation/summary.csv
    results/group_experiments/ablation/rankings.csv
    results/group_experiments/ablation/group_correlation.csv
    results/group_experiments/ablation/substitution.csv        (optional)
    results/group_experiments/ablation/figures/*.png
    results/group_experiments/ablation/manifest.json
    report/group_ablation.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import CONFIG, ROOT
from generate_report import table as table_text
from group_protocol import (GROUPS, MODELS, TASKS, degradation, group_map,
                            load_tasks, make_model, random_feature_sets,
                            rank_values, score)

OUT = ROOT / 'results' / 'group_experiments' / 'ablation'
METRICS = ['log_loss', 'accuracy', 'f1', 'macro_f1', 'roc_auc', 'pr_auc',
           'balanced_accuracy']


def fit_score(task, X_all, train, test, columns, model_name, seed):
    """Fit one condition and return metrics plus wall-clock seconds."""
    index = [task.features.index(c) for c in columns]
    model = make_model(model_name, task, seed)
    start = time.time()
    model.fit(X_all[np.ix_(train, index)], task.y[train])
    metrics = score(model, X_all[np.ix_(test, index)], task.y[test], task)
    return metrics, time.time() - start


def primary_metric(task):
    """Log-loss for every task: the only metric not saturated at 1.0 here."""
    return 'log_loss'


def group_correlation(task):
    """Mean |Spearman| between members of each pair of groups."""
    X = task.frame[task.features]
    corr = X.corr(method='spearman').abs()
    mapping = group_map(task.features)
    rows = []
    for a, b in combinations(GROUPS, 2):
        block = corr.loc[mapping[a], mapping[b]].to_numpy()
        rows.append(dict(group_a=a, group_b=b, mean_abs_spearman=float(block.mean()),
                         max_abs_spearman=float(block.max())))
    return pd.DataFrame(rows).sort_values('mean_abs_spearman', ascending=False)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    tasks = load_tasks(args.tasks)
    rng_seed = CONFIG['seed']
    runs = []
    substitutions = []
    correlation = {}

    for task_name, task in tasks.items():
        mapping = group_map(task.features)
        X_all = task.frame[task.features].to_numpy(dtype=np.float64)
        correlation[task_name] = group_correlation(task)
        correlation[task_name].assign(task=task_name).to_csv(
            OUT / f'group_correlation_{task_name}.csv', index=False)
        if args.substitution and task_name in args.substitution_tasks:
            pairs = correlation[task_name].head(3)
        else:
            pairs = correlation[task_name].iloc[0:0]

        for seed in args.seeds:
            train, test = task.split(seed)
            for model_name in args.models:
                full_cols = list(task.features)
                full, seconds = fit_score(task, X_all, train, test, full_cols,
                                          model_name, seed)
                runs.append(dict(task=task_name, model=model_name, seed=seed,
                                 condition='full', group='__all__',
                                 control_index=-1, n_features=len(full_cols),
                                 fit_seconds=seconds, **full))
                print(f'[{task_name}/{model_name}/seed {seed}] full '
                      f'{primary_metric(task)}={full[primary_metric(task)]:.5f} '
                      f'({seconds:.1f}s)', flush=True)

                for group in GROUPS:
                    columns = task.complement_of(group)
                    metrics, seconds = fit_score(task, X_all, train, test,
                                                 columns, model_name, seed)
                    runs.append(dict(task=task_name, model=model_name, seed=seed,
                                     condition='drop_group', group=group,
                                     control_index=-1, n_features=len(columns),
                                     fit_seconds=seconds, **metrics))
                    print(f'  drop {group:<11} '
                          f'{primary_metric(task)}={metrics[primary_metric(task)]:.5f}',
                          flush=True)

                    for k, subset in enumerate(random_feature_sets(
                            task.features, len(mapping[group]), args.controls,
                            rng_seed + seed * 1000 + GROUPS.index(group) * 37,
                            exclude=mapping[group])):
                        kept = [f for f in task.features if f not in set(subset)]
                        metrics, seconds = fit_score(task, X_all, train, test,
                                                     kept, model_name, seed)
                        runs.append(dict(task=task_name, model=model_name,
                                         seed=seed, condition='random_drop',
                                         group=group, control_index=k,
                                         n_features=len(kept),
                                         fit_seconds=seconds, **metrics))

                for row in pairs.itertuples():
                    partner = row.group_b
                    alone = task.complement_of(partner)
                    both = [f for f in task.features
                            if f.split('.')[0] not in (row.group_a, partner)]
                    for label, columns in ((f'partner_{partner}', alone),
                                           (f'pair_{partner}', both)):
                        metrics, seconds = fit_score(task, X_all, train, test,
                                                     columns, model_name, seed)
                        substitutions.append(dict(
                            task=task_name, model=model_name, seed=seed,
                            group=row.group_a, partner=partner, condition=label,
                            mean_abs_spearman=row.mean_abs_spearman,
                            n_features=len(columns), fit_seconds=seconds,
                            **metrics))
                print(f'[{task_name}/{model_name}/seed {seed}] done', flush=True)

    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / 'runs.csv', index=False)
    deltas = build_deltas(runs)
    deltas.to_csv(OUT / 'deltas.csv', index=False)
    summary, rankings = summarise(deltas, args)
    summary.to_csv(OUT / 'summary.csv', index=False)
    rankings.to_csv(OUT / 'rankings.csv', index=False)
    if substitutions:
        pd.DataFrame(substitutions).to_csv(OUT / 'substitution.csv', index=False)
    figures(summary, deltas, tasks)
    write_report(summary, rankings, deltas, correlation, tasks, args)
    write_manifest(args)
    print(summary.to_string(index=False), flush=True)
    print(f'Ablation artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def build_deltas(runs):
    """Express every condition as a change relative to its own full model."""
    rows = []
    keys = ['task', 'model', 'seed']
    full = runs[runs.condition == 'full'].set_index(keys)
    for record in runs[runs.condition != 'full'].itertuples():
        reference = full.loc[(record.task, record.model, record.seed)]
        row = dict(task=record.task, model=record.model, seed=record.seed,
                   condition=record.condition, group=record.group,
                   control_index=record.control_index, n_features=record.n_features)
        for metric in METRICS:
            value = getattr(record, metric, np.nan)
            base = reference.get(metric, np.nan)
            if pd.isna(value) or pd.isna(base):
                row[f'delta_{metric}'] = np.nan
            else:
                row[f'delta_{metric}'] = degradation(
                    {metric: base}, {metric: value}, metric)
        row['full_log_loss'] = reference['log_loss']
        row['full_f1'] = reference.get('f1', np.nan)
        row['full_macro_f1'] = reference.get('macro_f1', np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(deltas, args):
    """Aggregate across seeds and compare each group against the random null.

    Seeds reuse the same dataset, so the spread across seeds is *not* a
    sampling distribution. It is reported as run-to-run variability only; the
    empirical p-value compares the group against matched random drops, and the
    per-seed ranking is what the stability analysis consumes.
    """
    summary_rows = []
    ranking_rows = []
    for (task, model), block in deltas.groupby(['task', 'model']):
        null = block[block.condition == 'random_drop']
        for group, observed in block[block.condition == 'drop_group'].groupby('group'):
            row = dict(task=task, model=model, group=group,
                       n_features=int(observed.n_features.iloc[0]),
                       n_seeds=int(observed.seed.nunique()))
            for metric in ('log_loss', 'f1', 'macro_f1'):
                column = f'delta_{metric}'
                row[f'{metric}_delta_mean'] = float(observed[column].mean())
                row[f'{metric}_delta_sd'] = float(observed[column].std(ddof=1)) \
                    if len(observed) > 1 else float('nan')
                row[f'{metric}_null_mean'] = float(null[column].mean()) if len(null) else np.nan
                row[f'{metric}_null_sd'] = float(null[column].std(ddof=1)) if len(null) > 1 else np.nan
                # Pooled matched null: fraction of random drops at least as harmful.
                values = null[column].dropna().to_numpy()
                effect = row[f'{metric}_delta_mean']
                row[f'{metric}_p_empirical'] = (
                    float((np.sum(values >= effect) + 1) / (len(values) + 1))
                    if len(values) else np.nan)
            summary_rows.append(row)
        # Per-seed ranking of the primary metric, used by the stability module.
        for seed, seed_block in block[block.condition == 'drop_group'].groupby('seed'):
            seed_block = seed_block.reset_index(drop=True)
            ranked = rank_values(seed_block['delta_log_loss'].to_numpy(),
                                 seed_block['group'])
            for group, rank in ranked.items():
                ranking_rows.append(dict(task=task, model=model, seed=int(seed),
                                         group=group, rank=float(rank),
                                         delta_log_loss=float(
                                             seed_block.loc[seed_block.group == group,
                                                            'delta_log_loss'].iloc[0])))
    summary = pd.DataFrame(summary_rows)
    summary['rank_log_loss'] = (summary.groupby(['task', 'model'])['log_loss_delta_mean']
                                .rank(ascending=False, method='min').astype(int))
    summary = summary.sort_values(['task', 'model', 'rank_log_loss'])
    return summary, pd.DataFrame(ranking_rows)


def figures(summary, deltas, tasks):
    for task in summary.task.unique():
        block = summary[summary.task == task].copy()
        models = block.model.unique()
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 5.0),
                                 squeeze=False, sharey=True)
        for ax, model in zip(axes[0], models):
            part = block[block.model == model].sort_values('log_loss_delta_mean')
            null_mean = part.log_loss_null_mean.mean()
            null_sd = part.log_loss_null_sd.mean()
            colours = ['#b5453a' if p < 0.05 else '#7f8c8d'
                       for p in part.log_loss_p_empirical]
            ax.barh(part.group, part.log_loss_delta_mean, color=colours)
            ax.axvline(0, color='black', lw=.8)
            ax.axvline(null_mean, color='#2c6fbb', ls='--', lw=1,
                       label='random-drop mean')
            ax.axvspan(null_mean - null_sd, null_mean + null_sd, color='#2c6fbb',
                       alpha=.15, label='±1 SD of null')
            ax.set_title(model)
            ax.set_xlabel('Δ log-loss when group removed')
            ax.legend(fontsize=7, loc='lower right')
        fig.suptitle(f'Group ablation — {task} (red: empirical p < 0.05 vs matched random drops)')
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'ablation_{task}.png', dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.0 * len(tasks), 4.4),
                             squeeze=False)
    for ax, (task, data) in zip(axes[0], tasks.items()):
        part = deltas[(deltas.task == task) & (deltas.condition == 'drop_group')]
        order = (part.groupby('group').delta_log_loss.mean()
                 .sort_values(ascending=False).index)
        data_by_group = [part[part.group == g].delta_log_loss.to_numpy() for g in order]
        ax.boxplot(data_by_group, tick_labels=order, vert=True)
        ax.axhline(0, color='black', lw=.8)
        ax.set_ylabel('Δ log-loss vs full model')
        ax.set_title(task)
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle('Per-seed spread of group ablation effects (all models pooled)')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'ablation_delta_distribution.png', dpi=160)
    plt.close(fig)


def write_report(summary, rankings, deltas, correlation, tasks, args):
    lines = ['# Group ablation: bỏ từng nhóm đặc trưng forensic\n']
    lines.append('Sinh bởi `python/run_group_ablation.py`. Đây là thực nghiệm **mới**, '
                 'độc lập với pipeline đã kiểm chứng; không ghi đè artifact cũ.\n')
    lines.append('## 1. Câu hỏi và thiết kế\n')
    lines.append('- Câu hỏi: bỏ toàn bộ một nhóm plugin có làm mô hình **tệ hơn** nhiều '
                 'hơn việc bỏ một số lượng đặc trưng ngẫu nhiên bằng đúng nhóm đó không?')
    lines.append('- Mỗi điều kiện: fit lại mô hình từ đầu trên cùng split, cùng seed, '
                 'cùng siêu tham số. Ngân sách tuning = 0 và **bằng nhau** cho mọi điều kiện.')
    lines.append('- Đối chứng: với mỗi nhóm k đặc trưng, bỏ k đặc trưng ngẫu nhiên lấy từ '
                 f'**phần bù** của nhóm, {args.controls} lần rút.')
    lines.append('- Metric chính: `log_loss`. F1 nhị phân đã bão hoà ở 1.0 nên không thể '
                 'xếp hạng các điều kiện; F1/macro-F1 vẫn được báo cáo.\n')
    lines.append('## 2. Phạm vi đã chạy\n')
    lines.append(f'- Task: {", ".join(args.tasks)}; model: {", ".join(args.models)}; '
                 f'seed: {", ".join(map(str, args.seeds))}.')
    for task_name, task in tasks.items():
        lines.append(f'- `{task_name}`: {len(task.frame):,} dòng, {task.n_classes} lớp'
                     + (f', bỏ {task.conflicts_dropped} dòng xung đột nhãn '
                        f'({task.conflict_groups} nhóm vector)'
                        if task.conflicts_dropped else '') + '.')
    lines.append('')
    lines.append('## 3. Kết quả theo task\n')
    for task in summary.task.unique():
        block = summary[summary.task == task]
        lines.append(f'### {task}\n')
        table = block[['model', 'group', 'n_features', 'log_loss_delta_mean',
                       'log_loss_delta_sd', 'log_loss_null_mean',
                       'log_loss_p_empirical', 'rank_log_loss']].copy()
        table.columns = ['model', 'nhóm', 'số feature', 'Δlog-loss TB', 'SD qua seed',
                         'null TB', 'p thực nghiệm', 'hạng']
        lines.append(table_text(table.round(5)) + '\n')
        significant = block[block.log_loss_p_empirical < 0.05]
        if len(significant):
            lines.append('- Nhóm vượt đối chứng ngẫu nhiên (p < 0.05): ' +
                         ', '.join(f'`{g}`' for g in
                                   significant.sort_values('rank_log_loss').group) + '.')
        else:
            lines.append('- **Không nhóm nào** vượt đối chứng ngẫu nhiên ở p < 0.05: '
                         'không có bằng chứng ablation cho lợi ích dự đoán của bất kỳ nhóm nào.')
        lines.append('')
    lines.append('![Group ablation](../results/group_experiments/ablation/figures/'
                 'ablation_binary.png)\n')
    lines.append('![Phân bố hiệu ứng](../results/group_experiments/ablation/figures/'
                 'ablation_delta_distribution.png)\n')
    lines.append('## 4. Tương quan giữa các nhóm (khả năng thay thế)\n')
    for task_name, corr in correlation.items():
        top = corr.head(3)
        lines.append(f'- `{task_name}`: cặp tương quan mạnh nhất — ' + '; '.join(
            f'`{r.group_a}`↔`{r.group_b}` (|ρ| TB {r.mean_abs_spearman:.3f})'
            for r in top.itertuples()) + '.')
    lines.append('')
    if (OUT / 'substitution.csv').exists():
        sub = pd.read_csv(OUT / 'substitution.csv')
        lines.append('### 4.1 Kiểm tra thay thế (drop đơn vs drop cặp)\n')
        lines.append('Nếu bỏ cả hai nhóm tương quan **không tệ hơn** bỏ nhóm mạnh hơn một '
                     'mình, hai nhóm chia sẻ thông tin thay vì bổ sung cho nhau.\n')
        lines.append(table_text(sub[['task', 'model', 'group', 'partner', 'condition',
                                     'log_loss', 'macro_f1']].round(5)) + '\n')
    lines.append('## 5. Cách đọc và giới hạn\n')
    lines.append('- Δ dương = bỏ nhóm làm log-loss tăng = nhóm có ích cho mô hình này.')
    lines.append('- p thực nghiệm so với phân bố random-drop **cùng cỡ**, không phải '
                 'p-value của một phép kiểm định tham số; độ phân giải bị giới hạn bởi '
                 f'số lần rút ({args.controls} × 9 nhóm × số seed).')
    lines.append('- SD qua seed là biến động do chia lại dữ liệu và khởi tạo mô hình; '
                 '**các seed dùng chung dataset nên không phải mẫu độc lập**.')
    lines.append('- Ablation chỉ đo phần thông tin mà mô hình *này* không thể bù đắp bằng '
                 'các nhóm còn lại; nhóm bị thay thế tốt sẽ có Δ≈0 dù bản thân nó có tín hiệu.')
    lines.append('- Chưa có split theo specimen; xem `report/source_aware_split.md`.\n')
    (ROOT / 'report').mkdir(exist_ok=True)
    (ROOT / 'report' / 'group_ablation.md').write_text('\n'.join(lines))
    print('Generated report/group_ablation.md', flush=True)


def write_manifest(args):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/run_group_ablation.py',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        config=CONFIG,
        tasks=args.tasks,
        models=args.models,
        seeds=args.seeds,
        controls_per_group=args.controls,
        primary_metric='log_loss',
        tuning_budget='none (identical hyper-parameters for every condition)',
        note='Random-drop controls are drawn from the complement of the ablated group.',
    )
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--seeds', default='42,43,44,45,46')
    parser.add_argument('--controls', type=int, default=3,
                        help='Matched random-drop draws per group')
    parser.add_argument('--substitution', action='store_true',
                        help='Also run drop-partner and drop-pair tests')
    parser.add_argument('--substitution-tasks', default='binary,type4')
    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    args.models = [m.strip() for m in args.models.split(',') if m.strip()]
    args.seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    args.substitution_tasks = [t.strip() for t in args.substitution_tasks.split(',')
                               if t.strip()]
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        parser.error(f'unknown tasks: {sorted(unknown)}')
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f'unknown models: {sorted(unknown)}')
    run(args)


if __name__ == '__main__':
    main()
