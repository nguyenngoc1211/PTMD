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
from split_manager import STRATEGIES
from group_protocol import (GROUPS, MODELS, TASKS, degradation, group_map,
                            load_tasks, make_model, random_feature_sets,
                            rank_values, score)

OUT = ROOT / 'results' / 'group_experiments' / 'ablation'
METRICS = ['log_loss', 'accuracy', 'f1', 'macro_f1', 'roc_auc', 'pr_auc',
           'balanced_accuracy']
# Below this full-model log-loss the task has no headroom for ablation deltas.
# Measured baseline on the old leaky random split was 7.0e-05.
SATURATION_LOG_LOSS = 1e-3


def set_out(name):
    """Point every writer at a versioned results directory."""
    global OUT
    OUT = ROOT / 'results' / 'group_experiments' / name


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


def blocks(task_name, correlation, args):
    """Correlated group pairs to ablate jointly (substitution test).

    Two groups that each look redundant alone may still be jointly necessary,
    because each substitutes for the other. Dropping the pair breaks the
    substitution and exposes the shared information. The pairs are the
    highest-mean |Spearman| cross-group links, ranked before any result is
    seen, so the selection cannot chase a favourable delta.
    """
    if not args.blocks:
        return []
    table = correlation[task_name].head(args.blocks)
    return [(row.group_a, row.group_b) for row in table.itertuples()]


def run(args):
    split = getattr(args, 'split', 'random')
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
            train, test = task.split(seed, strategy=split)
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
                t0 = time.time()

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
                        if (k + 1) % 25 == 0 or k + 1 == args.controls:
                            print(f'    {group} ctrl {k + 1}/{args.controls} '
                                  f'({time.time() - t0:.0f}s elapsed)',
                                  flush=True)

                for group_a, group_b in blocks(task_name, correlation, args):
                    kept = [f for f in task.features
                            if f.split('.')[0] not in (group_a, group_b)]
                    metrics, seconds = fit_score(task, X_all, train, test, kept,
                                                 model_name, seed)
                    runs.append(dict(task=task_name, model=model_name, seed=seed,
                                     condition='drop_block',
                                     group=f'{group_a}+{group_b}', control_index=-1,
                                     n_features=len(kept), fit_seconds=seconds,
                                     **metrics))
                    print(f'  drop BLOCK {group_a}+{group_b:<10} '
                          f'{primary_metric(task)}={metrics[primary_metric(task)]:.5f}',
                          flush=True)

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
    runs['split'] = split
    runs.to_csv(OUT / 'runs.csv', index=False)
    deltas = build_deltas(runs)
    deltas.to_csv(OUT / 'deltas.csv', index=False)
    recall_deltas(runs).to_csv(OUT / 'per_class_recall_deltas.csv', index=False)
    summary, rankings = summarise(deltas, args)
    summary.to_csv(OUT / 'summary.csv', index=False)
    rankings.to_csv(OUT / 'rankings.csv', index=False)
    block_table = summarise_blocks(deltas, summary)
    if len(block_table):
        block_table.to_csv(OUT / 'blocks.csv', index=False)
    if substitutions:
        pd.DataFrame(substitutions).to_csv(OUT / 'substitution.csv', index=False)
    figures(summary, deltas, tasks)
    write_report(summary, rankings, deltas, correlation, tasks, args, block_table)
    write_manifest(args)
    print(summary.to_string(index=False), flush=True)
    print(f'Ablation artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def summarise_blocks(deltas, summary):
    """Pairwise block ablation versus the sum of its single-group drops.

    If dropping A alone costs little and dropping B alone costs little, but
    dropping both costs a lot, the pair carries information neither group can
    supply on its own. ``synergy`` is that gap; a positive value means the two
    groups are *not* interchangeable even though each looks redundant alone.
    """
    rows = []
    blocks = deltas[deltas.condition == 'drop_block']
    singles = deltas[deltas.condition == 'drop_group']
    for record in blocks.itertuples():
        first, second = record.group.split('+')
        own = singles[(singles.task == record.task) & (singles.model == record.model)
                      & (singles.seed == record.seed)
                      & (singles.group.isin([first, second]))]
        if len(own) != 2:
            continue
        row = dict(task=record.task, model=record.model, seed=int(record.seed),
                   block=record.group, n_features=int(record.n_features),
                   block_log_loss_delta=float(record.delta_log_loss),
                   single_max_log_loss_delta=float(own.delta_log_loss.max()),
                   single_sum_log_loss_delta=float(own.delta_log_loss.sum()))
        row['synergy'] = float(record.delta_log_loss) - float(own.delta_log_loss.sum())
        if 'delta_macro_f1' in record._fields:
            row['block_macro_f1_delta'] = float(record.delta_macro_f1)
        rows.append(row)
    return pd.DataFrame(rows)


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


def recall_deltas(runs):
    """Per-class recall change (condition minus full) for every condition."""
    rows = []
    full = runs[runs.condition == 'full'].set_index(['task', 'model', 'seed'])
    for record in runs[runs.condition != 'full'].itertuples():
        base = json.loads(full.loc[(record.task, record.model, record.seed),
                                   'per_class_recall'])
        current = json.loads(record.per_class_recall)
        for label, value in current.items():
            rows.append(dict(task=record.task, model=record.model, seed=record.seed,
                             condition=record.condition, group=record.group,
                             control_index=record.control_index, label=label,
                             delta_recall=float(value) - float(base[label])))
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
            # Controls are matched by group: same number of features and drawn
            # from the complement of that group. Pooling across groups would mix
            # different removal sizes and invalidate the "same-size" comparison.
            matched = null[null.group == group]
            row = dict(task=task, model=model, group=group,
                       n_features=int(observed.n_features.iloc[0]),
                       n_seeds=int(observed.seed.nunique()),
                       n_controls=int(matched.control_index.nunique()))
            for metric in METRICS:
                column = f'delta_{metric}'
                row[f'{metric}_delta_mean'] = float(observed[column].mean())
                row[f'{metric}_delta_sd'] = float(observed[column].std(ddof=1)) \
                    if len(observed) > 1 else float('nan')
                row[f'{metric}_null_mean'] = (float(matched[column].mean())
                                              if len(matched) else np.nan)
                row[f'{metric}_null_sd'] = (float(matched[column].std(ddof=1))
                                            if len(matched) > 1 else np.nan)
                # Fraction of matched random drops at least as harmful.
                values = matched[column].dropna().to_numpy()
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
            colours = ['#b5453a' if p < 0.05 else '#7f8c8d'
                       for p in part.log_loss_p_empirical]
            bars = ax.barh(part.group, part.log_loss_delta_mean, color=colours)
            ax.axvline(0, color='black', lw=.8)
            for bar, (_, row) in zip(bars, part.iterrows()):
                centre = bar.get_y() + bar.get_height() / 2
                ax.plot([row.log_loss_null_mean] * 2, [centre - .32, centre + .32],
                        color='#2c6fbb', lw=2)
                if np.isfinite(row.log_loss_null_sd):
                    ax.plot([row.log_loss_null_mean - row.log_loss_null_sd,
                             row.log_loss_null_mean + row.log_loss_null_sd],
                            [centre, centre], color='#2c6fbb', lw=1, alpha=.6)
            ax.plot([], [], color='#2c6fbb', lw=2, label='random-drop mean (matched)')
            ax.plot([], [], color='#2c6fbb', lw=1, alpha=.6, label='±1 SD')
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


def saturation_note(deltas, task_name):
    """Warn when a task is so well solved that ablation cannot resolve groups.

    A group's delta is bounded above by the headroom the full model leaves. On
    a task with log-loss ~7e-05 and F1 = 1.0 there is no headroom, so deltas of
    ~1e-5 are noise and "no group matters" would be a measurement artefact, not
    a finding. This must be stated before any ranking is read.
    """
    block = deltas[deltas.task == task_name]
    headroom = float(block['full_log_loss'].min())
    ceiling = block['full_f1'].dropna().max() if block['full_f1'].notna().any() else np.nan
    ceiling_macro = (block['full_macro_f1'].dropna().max()
                     if block['full_macro_f1'].notna().any() else np.nan)
    observed = float(block['delta_log_loss'].abs().max())
    saturated = headroom < SATURATION_LOG_LOSS
    return dict(task=task_name, full_log_loss_min=headroom, full_f1_max=ceiling,
                full_macro_f1_max=ceiling_macro, max_abs_delta=observed,
                headroom_ratio=(observed / headroom if headroom > 0 else np.nan),
                saturated=saturated)


def write_report(summary, rankings, deltas, correlation, tasks, args, block_table=None):
    split = getattr(args, 'split', 'random')
    lines = ['# Group ablation: bỏ từng nhóm đặc trưng forensic\n']
    lines.append('Sinh bởi `python/run_group_ablation.py`. Không ghi đè artifact cũ.\n')
    lines.append('## 1. Câu hỏi và thiết kế\n')
    lines.append('- Câu hỏi: bỏ toàn bộ một nhóm plugin có làm mô hình **tệ hơn** nhiều '
                 'hơn việc bỏ một số lượng đặc trưng ngẫu nhiên bằng đúng nhóm đó không?')
    lines.append('- Mỗi điều kiện: fit lại mô hình từ đầu trên cùng split, cùng seed, '
                 'cùng siêu tham số. Ngân sách tuning = 0 và **bằng nhau** cho mọi điều kiện.')
    lines.append(f'- Split: `{split}`'
                 + (' — mọi snapshot của cùng một specimen nằm về MỘT phía '
                    '(khắc phục 99,8% specimen giao train/test của split ngẫu nhiên cũ).'
                    if split == 'source_aware' else
                    ' — chia dòng ngẫu nhiên, **có specimen leakage** (xem TECHNICAL_AUDIT §3).'))
    lines.append('- Đối chứng: với mỗi nhóm k đặc trưng, bỏ k đặc trưng ngẫu nhiên lấy từ '
                 f'**phần bù** của nhóm, {args.controls} lần rút.')
    if getattr(args, 'blocks', 0):
        lines.append(f'- Block-out: {args.blocks} cặp nhóm có |ρ|cross cao nhất bị bỏ '
                     '**đồng thời**, để phân biệt "nhóm vô dụng" với "hai nhóm thay thế '
                     'nhau".\n')
    lines.append('- Metric chính: `log_loss`.\n')

    notes = pd.DataFrame([saturation_note(deltas, t) for t in summary.task.unique()])
    lines.append('## 2. Kiểm tra bão hoà trước khi đọc hạng\n')
    display = notes.copy()
    display.columns = ['task', 'log-loss full nhỏ nhất', 'F1 full lớn nhất',
                       'macro-F1 full lớn nhất', '|Δ| lớn nhất quan sát',
                       'tỷ lệ Δ/headroom', 'bão hoà?']
    lines.append(table_text(display.round(6)) + '\n')
    for row in notes.itertuples():
        if row.saturated:
            lines.append(f'- **`{row.task}` BÃO HOÀ** (log-loss full {row.full_log_loss_min:.2e} '
                         f'< {SATURATION_LOG_LOSS:.0e}). Mọi Δ trên task này nằm trong phần '
                         'cuối của thang đo và **không được dùng để kết luận nhóm nào vô '
                         'quan trọng**; chỉ dùng làm đối chứng phương pháp.')
    if not notes.saturated.any():
        lines.append('- Không task nào bão hoà; Δ có headroom để giải thích.')
    lines.append('')

    lines.append('## 3. Phạm vi đã chạy\n')
    lines.append(f'- Task: {", ".join(args.tasks)}; model: {", ".join(args.models)}; '
                 f'seed: {", ".join(map(str, args.seeds))}; split: `{split}`.')
    for task_name, task in tasks.items():
        lines.append(f'- `{task_name}`: {len(task.frame):,} dòng, {task.n_classes} lớp'
                     + (f', bỏ {task.conflicts_dropped} dòng xung đột nhãn '
                        f'({task.conflict_groups} nhóm vector)'
                        if task.conflicts_dropped else '') + '.')
    total_fits = int((deltas.condition != 'full').sum() + 1)
    lines.append(f'- Tổng số lần fit mô hình được ghi nhận: ~{total_fits:,}.\n')

    lines.append('## 4. Kết quả theo task\n')
    for task in summary.task.unique():
        block = summary[summary.task == task]
        saturated = bool(notes[notes.task == task].saturated.iloc[0])
        lines.append(f'### {task}\n')
        columns = [c for c in ['model', 'group', 'n_features', 'log_loss_delta_mean',
                               'log_loss_delta_sd', 'log_loss_null_mean', 'log_loss_null_sd',
                               'log_loss_p_empirical', 'accuracy_delta_mean',
                               'macro_f1_delta_mean', 'roc_auc_delta_mean',
                               'pr_auc_delta_mean', 'rank_log_loss']
                   if c in block.columns]
        friendly = {'model': 'model', 'group': 'nhóm', 'n_features': 'số feature giữ lại',
                    'log_loss_delta_mean': 'Δlog-loss TB', 'log_loss_delta_sd': 'SD qua seed',
                    'log_loss_null_mean': 'null TB', 'log_loss_null_sd': 'null SD',
                    'log_loss_p_empirical': 'p thực nghiệm',
                    'accuracy_delta_mean': 'Δaccuracy TB',
                    'macro_f1_delta_mean': 'Δmacro-F1 TB',
                    'roc_auc_delta_mean': 'ΔROC-AUC TB',
                    'pr_auc_delta_mean': 'ΔPR-AUC TB',
                    'rank_log_loss': 'hạng'}
        table = block[columns].copy()
        table.columns = [friendly.get(c, c) for c in columns]
        lines.append(table_text(table.round(5)) + '\n')
        significant = block[block.log_loss_p_empirical < 0.05]
        if saturated:
            lines.append('- Task bão hoà: **không diễn giải bảng này thành kết luận về '
                         'tầm quan trọng của nhóm**.')
        elif len(significant):
            lines.append('- Nhóm vượt đối chứng ngẫu nhiên (p < 0.05): ' +
                         ', '.join(f'`{g}`' for g in
                                   significant.sort_values('rank_log_loss').group) + '.')
        else:
            lines.append('- Không nhóm nào vượt đối chứng ngẫu nhiên ở p < 0.05 **trên task '
                         'này và split này**. Đây là phát ngôn có phạm vi, không phải phát '
                         'ngôn chung về giá trị forensic của các nhóm.')
        lines.append('')
    recall_path = OUT / 'per_class_recall_deltas.csv'
    if recall_path.exists():
        record = pd.read_csv(recall_path)
        record = record[record.condition == 'drop_group']
        if len(record):
            lines.append('### 4.1 Thay đổi recall theo từng lớp khi bỏ nhóm\n')
            lines.append('Δrecall = recall(sau khi bỏ nhóm) − recall(full); âm = lớp đó '
                         'bị mất recall. Bảng chỉ dùng drop group, không trộn đối chứng.\n')
            pivot = record.pivot_table(index='group', columns='label',
                                       values='delta_recall', aggfunc='mean')
            lines.append(table_text(pivot.round(4)) + '\n')
    for task in summary.task.unique():
        lines.append(f'![Group ablation — {task}](../{OUT.relative_to(ROOT)}/figures/'
                     f'ablation_{task}.png)\n')
    lines.append(f'![Phân bố hiệu ứng](../{OUT.relative_to(ROOT)}/figures/'
                 'ablation_delta_distribution.png)\n')

    if block_table is not None and len(block_table):
        lines.append('## 5. Block-out: bổ sung hay thay thế?\n')
        lines.append('`synergy` = Δ(cả hai nhóm) − Δ(nhóm A) − Δ(nhóm B). Dương có nghĩa '
                     'hai nhóm **cùng nhau** mang thông tin mà từng nhóm đơn lẻ không '
                     'cung cấp được — tức là không thay thế được nhau, dù Δ đơn lẻ nhỏ.\n')
        display = block_table.copy()
        display.columns = [c.replace('_', ' ') for c in display.columns]
        lines.append(table_text(display.round(6)) + '\n')

    lines.append('## 6. Tương quan giữa các nhóm (khả năng thay thế)\n')
    for task_name, corr in correlation.items():
        top = corr.head(3)
        lines.append(f'- `{task_name}`: cặp tương quan mạnh nhất — ' + '; '.join(
            f'`{r.group_a}`↔`{r.group_b}` (|ρ| TB {r.mean_abs_spearman:.3f})'
            for r in top.itertuples()) + '.')
    lines.append('')
    if (OUT / 'substitution.csv').exists():
        sub = pd.read_csv(OUT / 'substitution.csv')
        lines.append('### 6.1 Kiểm tra thay thế (drop đơn vs drop cặp)\n')
        lines.append('Nếu bỏ cả hai nhóm tương quan **không tệ hơn** bỏ nhóm mạnh hơn một '
                     'mình, hai nhóm chia sẻ thông tin thay vì bổ sung cho nhau.\n')
        lines.append(table_text(sub[['task', 'model', 'group', 'partner', 'condition',
                                     'log_loss', 'macro_f1']].round(5)) + '\n')
    lines.append('## 7. Cách đọc và giới hạn\n')
    lines.append('- Δ dương = bỏ nhóm làm log-loss tăng = nhóm có ích cho mô hình này.')
    lines.append('- p thực nghiệm so với phân bố random-drop **cùng cỡ**, không phải '
                 'p-value của một phép kiểm định tham số; độ phân giải bị giới hạn bởi '
                 f'số lần rút ({args.controls}).')
    lines.append('- SD qua seed là biến động do chia lại dữ liệu và khởi tạo mô hình; '
                 '**các seed dùng chung dataset nên không phải mẫu độc lập**.')
    lines.append('- Ablation chỉ đo phần thông tin mà mô hình *này* không thể bù đắp bằng '
                 'các nhóm còn lại; nhóm bị thay thế tốt sẽ có Δ≈0 dù bản thân nó có tín hiệu.')
    lines.append('- Δ≈0 **không** được viết thành "nhóm vô ích"; phải viết "thông tin trong '
                 'nhóm có thể thay thế bởi nhóm khác", kèm bằng chứng block-out ở mục 5.\n')
    (ROOT / 'report').mkdir(exist_ok=True)
    name = getattr(args, 'report_name', None) or 'group_ablation.md'
    (ROOT / 'report' / name).write_text('\n'.join(lines))
    print(f'Generated report/{name}', flush=True)


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
        split=getattr(args, 'split', 'random'),
        blocks=getattr(args, 'blocks', 0),
        controls_per_group=args.controls,
        primary_metric='log_loss',
        tuning_budget='none (identical hyper-parameters for every condition)',
        note='Random-drop controls are drawn from the complement of the ablated group.',
        saturation_threshold_log_loss=SATURATION_LOG_LOSS,
    )
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--seeds', default='42,43,44,45,46')
    parser.add_argument('--controls', type=int, default=3,
                        help='Matched random-drop draws per group')
    parser.add_argument('--split', default='random', choices=list(STRATEGIES),
                        help='random = the verified (leaky) rule; '
                             'source_aware = no specimen crosses train/test')
    parser.add_argument('--blocks', type=int, default=0,
                        help='Jointly ablate this many top-correlated group pairs')
    parser.add_argument('--out-name', default='ablation',
                        help='Results subdirectory under results/group_experiments')
    parser.add_argument('--report-name', default=None,
                        help='Report file name under report/')
    parser.add_argument('--force', action='store_true',
                        help='Allow writing into a results directory that already has runs.csv')
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
    if args.out_name != 'ablation':
        set_out(args.out_name)
    if (OUT / 'runs.csv').exists() and not args.force:
        parser.error(f'{OUT.relative_to(ROOT)}/runs.csv already exists; '
                     'choose another --out-name or pass --force (never overwrite '
                     'a previous experiment silently)')
    run(args)


if __name__ == '__main__':
    main()
