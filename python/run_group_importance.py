"""Direct group-level attribution: joint permutation and exact group Shapley.

Two rules govern this module.

1. A group measure must be computed *at the group level*. Summing feature-wise
   SHAP or feature-wise CFI and calling the result "group importance" is not a
   group measure: it attributes interaction effects to individual members and
   then adds them up. Nothing here does that.
2. Permutation and Shapley answer different questions and are both reported.
   Permutation asks "how much does *this fitted model* lose when the group's
   association with the rest of the row is destroyed?". Shapley asks "how much
   of the model output is attributable to the group, averaged over all
   coalitions?".

Methods
-------
``permutation`` (direct group permutation)
    The whole column block of a group is permuted jointly with one row
    permutation. Within-group dependence is preserved exactly (the rows are
    real rows); only the group's relation to the other features is broken.
    Matched random blocks of the same width, drawn from the complement, give a
    null that says how much damage an arbitrary block of that width causes.

``shapley`` (exact group Shapley, interventional value function)
    With nine players there are 2^9 = 512 coalitions, so the Shapley values are
    computed exactly, by definition, from the model itself:

        v(S) = E_background[ f(x_S, B_{S^c}) ]

    ``f`` is logit P(malware) for the binary task and log P(true class) for the
    multiclass tasks, so the nine group values sum exactly to
    ``f(x) - v(empty)`` for every explained row. This is a direct group
    decomposition, not a post-hoc sum of feature attributions.

    The interventional value function breaks feature dependence across the
    coalition boundary, which is the standard limitation of marginal Shapley.
    It is stated in the report and is *not* repaired by pretending the result
    is conditional. A conditional group Shapley would need one conditional
    sampler per coalition (512 samplers); that is out of budget here and is
    recorded as an open item rather than approximated silently.

Outputs (new tree):
    results/group_experiments/importance/permutation_runs.csv
    results/group_experiments/importance/permutation_summary.csv
    results/group_experiments/importance/permutation_rows.npz
    results/group_experiments/importance/shapley_groups.csv
    results/group_experiments/importance/shapley_checks.json
    results/group_experiments/importance/rankings.csv
    results/group_experiments/importance/figures/*.png
    results/group_experiments/importance/manifest.json
    report/group_importance.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import CONFIG, ROOT
from generate_report import table as table_text
from group_protocol import (GROUPS, MODELS, TASKS, group_map, load_tasks,
                            make_model, random_feature_sets, rank_values, score)
from split_manager import load_locked

OUT = ROOT / 'results' / 'group_experiments' / 'importance'


def fit(task, train, columns, model_name, seed):
    index = [task.features.index(c) for c in columns]
    X = task.frame[task.features].to_numpy(dtype=np.float64)
    model = make_model(model_name, task, seed)
    model.fit(X[np.ix_(train, index)], task.y[train])
    return model, index


def _row_logloss(probability, y):
    """Per-row log-loss for binary probabilities or a full class matrix."""
    probability = np.asarray(probability, dtype=np.float64)
    y = np.asarray(y)
    if probability.ndim == 1:
        p = np.clip(probability, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    p = np.clip(probability[np.arange(len(y)), y], 1e-15, 1.0)
    return -np.log(p)


def permutation(task, model, index, test, groups, repeats, seed, controls):
    """Jointly permute each group block; return per-row and aggregate effects."""
    X = task.frame[task.features].to_numpy(dtype=np.float64)
    X_test = X[np.ix_(test, index)]
    y_test = task.y[test]
    base_loss = _row_logloss(model.predict_proba(X_test), y_test)
    base = score(model, X_test, y_test, task)
    rng = np.random.default_rng(seed)
    rows = {}
    runs = []
    for group, members in groups.items():
        local = [index.index(task.features.index(c)) for c in members]
        effects = []
        for _ in range(repeats):
            permuted = X_test.copy()
            order = rng.permutation(len(X_test))
            permuted[:, local] = X_test[np.ix_(order, local)]
            loss = _row_logloss(model.predict_proba(permuted), y_test)
            effects.append(loss - base_loss)
        mean_effect = np.mean(effects, axis=0)
        rows[group] = mean_effect
        runs.append(dict(condition='permute_group', group=group, control_index=-1,
                         delta_log_loss=float(mean_effect.mean()),
                         delta_log_loss_sd=float(np.std(np.mean(effects, axis=1), ddof=1))
                         if repeats > 1 else float('nan')))
        for k, subset in enumerate(random_feature_sets(
                task.features, len(members), controls,
                seed * 7919 + GROUPS.index(group), exclude=members)):
            local_random = [index.index(task.features.index(c)) for c in subset]
            effects = []
            for _ in range(repeats):
                permuted = X_test.copy()
                order = rng.permutation(len(X_test))
                permuted[:, local_random] = X_test[np.ix_(order, local_random)]
                loss = _row_logloss(model.predict_proba(permuted), y_test)
                effects.append(loss - base_loss)
            mean_effect = np.mean(effects, axis=0)
            runs.append(dict(condition='permute_random', group=group, control_index=k,
                             delta_log_loss=float(mean_effect.mean()),
                             delta_log_loss_sd=float(np.std(np.mean(effects, axis=1), ddof=1))
                             if repeats > 1 else float('nan'),
                             delta_macro_f1=float('nan')))
    return pd.DataFrame(runs), pd.DataFrame(rows), base


def shapley(task, model, index, test, groups, n_explain, n_background, seed):
    """Exact Shapley over the nine groups with an interventional value function."""
    X = task.frame[task.features].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    explain = np.sort(rng.choice(test, size=min(n_explain, len(test)), replace=False))
    train_pool = np.setdiff1d(np.arange(len(X)), test)
    background = np.sort(rng.choice(train_pool, size=min(n_background, len(train_pool)),
                                    replace=False))
    X_explain = X[np.ix_(explain, index)]
    X_background = X[np.ix_(background, index)]
    y_explain = task.y[explain]
    players = list(groups)
    n_players = len(players)
    columns_of = {g: [index.index(task.features.index(c)) for c in members]
                  for g, members in groups.items()}
    all_columns = set(range(len(index)))

    coalitions = []
    for size in range(n_players + 1):
        coalitions.extend(combinations(range(n_players), size))
    position = {coalition: i for i, coalition in enumerate(coalitions)}

    values = np.empty((len(coalitions), len(explain)))
    n_rows, n_bg = len(explain), len(background)
    for i, coalition in enumerate(coalitions):
        inside = sorted({c for player in coalition for c in columns_of[players[player]]})
        outside = sorted(all_columns - set(inside))
        block = np.empty((n_rows * n_bg, len(index)))
        if inside:
            block[:, inside] = np.repeat(X_explain[:, inside], n_bg, axis=0)
        if outside:
            block[:, outside] = np.tile(X_background[:, outside], (n_rows, 1))
        values[i] = _output(model, block, task, np.repeat(y_explain, n_bg)
                            ).reshape(n_rows, n_bg).mean(axis=1)

    factorial = [math.factorial(i) for i in range(n_players + 1)]
    phi = np.zeros((n_rows, n_players))
    for player in range(n_players):
        others = [p for p in range(n_players) if p != player]
        for size in range(n_players):
            weight = factorial[size] * factorial[n_players - size - 1] / factorial[n_players]
            for subset in combinations(others, size):
                with_player = tuple(sorted(subset + (player,)))
                without = tuple(sorted(subset))
                phi[:, player] += weight * (values[position[with_player]]
                                            - values[position[without]])
    # Additivity check: the nine values must reconstruct f(x) - v(empty).
    reconstructed = phi.sum(axis=1) + values[position[()]]
    target = _output(model, X_explain, task, y_explain)
    error = float(np.max(np.abs(reconstructed - target)))
    summary = pd.DataFrame({
        'group': players,
        'mean_phi': phi.mean(axis=0),
        'mean_abs_phi': np.abs(phi).mean(axis=0),
        'sd_phi': phi.std(axis=0, ddof=1),
        'n_explained': n_rows,
        'n_background': n_bg,
    })
    checks = dict(n_coalitions=len(coalitions), n_explained=n_rows,
                  n_background=n_bg, additivity_max_abs_error=error,
                  explained_rows=int(len(explain)), seed=int(seed))
    return summary, checks, phi, explain


def _output(model, X, task, y):
    """Scalar model output whose group decomposition is additive."""
    probability = model.predict_proba(X)
    if task.n_classes == 2:
        p = np.clip(probability[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p) - np.log1p(-p)
    p = np.clip(probability[np.arange(len(X)), y], 1e-12, 1.0)
    return np.log(p)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    tasks = load_tasks(args.tasks)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    permutation_rows, permutation_runs, shapley_rows = [], [], []
    checks = []

    for task_name, task in tasks.items():
        groups = group_map(task.features)
        for seed in args.seeds:
            if args.split == 'source_aware':
                train, test = load_locked(args.version, task_name, seed)
            else:
                train, test = task.split(seed)
            for model_name in args.models:
                model, index = fit(task, train, task.features, model_name, seed)
                if 'permutation' in methods:
                    start = time.time()
                    runs, rows, base = permutation(task, model, index, test, groups,
                                                   args.permutation_repeats, seed,
                                                   args.controls)
                    runs.insert(0, 'task', task_name)
                    runs.insert(1, 'model', model_name)
                    runs.insert(2, 'seed', seed)
                    permutation_runs.append(runs)
                    for group, values in rows.items():
                        permutation_rows.append(dict(task=task_name, model=model_name,
                                                     seed=seed, group=group,
                                                     mean_row_effect=float(values.mean()),
                                                     median_row_effect=float(np.median(values))))
                    rows.assign(task=task_name, model=model_name, seed=seed,
                                test_position=test).to_csv(
                        OUT / f'permutation_rows_{task_name}_{model_name}_{seed}.csv',
                        index=False)
                    print(f'[perm {task_name}/{model_name}/seed {seed}] '
                          f'{time.time() - start:.1f}s', flush=True)
                if 'shapley' in methods and model_name in args.shapley_models:
                    start = time.time()
                    summary, check, phi, explain = shapley(
                        task, model, index, test, groups, args.shapley_explain,
                        args.shapley_background, seed)
                    summary.insert(0, 'task', task_name)
                    summary.insert(1, 'model', model_name)
                    summary.insert(2, 'seed', seed)
                    shapley_rows.append(summary)
                    check.update(task=task_name, model=model_name)
                    checks.append(check)
                    np.savez_compressed(
                        OUT / f'shapley_values_{task_name}_{model_name}_{seed}.npz',
                        phi=phi, groups=np.array(list(groups)), explain=explain)
                    print(f'[shap {task_name}/{model_name}/seed {seed}] '
                          f'{time.time() - start:.1f}s '
                          f'additivity_err={check["additivity_max_abs_error"]:.2e}',
                          flush=True)

    if permutation_runs:
        runs = pd.concat(permutation_runs, ignore_index=True)
        runs.to_csv(OUT / 'permutation_runs.csv', index=False)
        summary = summarise_permutation(runs)
        summary.to_csv(OUT / 'permutation_summary.csv', index=False)
        pd.DataFrame(permutation_rows).to_csv(OUT / 'permutation_group_means.csv',
                                              index=False)
        figures_permutation(summary)
    if shapley_rows:
        shapley_table = pd.concat(shapley_rows, ignore_index=True)
        shapley_table['rank'] = (shapley_table.groupby(['task', 'model', 'seed'])
                                 ['mean_abs_phi']
                                 .rank(ascending=False, method='min').astype(int))
        shapley_table.to_csv(OUT / 'shapley_groups.csv', index=False)
        (OUT / 'shapley_checks.json').write_text(json.dumps(checks, indent=2))
        figures_shapley(shapley_table)
    build_rankings(args.ablation_name)
    write_report(args)
    write_manifest(args)
    print(f'Group importance artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def summarise_permutation(runs):
    rows = []
    for (task, model), block in runs.groupby(['task', 'model']):
        null = block[block.condition == 'permute_random']
        for group, observed in block[block.condition == 'permute_group'].groupby('group'):
            values = null.delta_log_loss.to_numpy()
            effect = float(observed.delta_log_loss.mean())
            rows.append(dict(
                task=task, model=model, group=group,
                n_seeds=int(observed.seed.nunique()),
                delta_log_loss_mean=effect,
                delta_log_loss_sd=float(observed.delta_log_loss.std(ddof=1))
                if len(observed) > 1 else float('nan'),
                null_mean=float(values.mean()) if len(values) else np.nan,
                null_sd=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                p_empirical=float((np.sum(values >= effect) + 1) / (len(values) + 1))
                if len(values) else np.nan,
            ))
    summary = pd.DataFrame(rows)
    summary['rank'] = (summary.groupby(['task', 'model'])['delta_log_loss_mean']
                       .rank(ascending=False, method='min').astype(int))
    return summary.sort_values(['task', 'model', 'rank'])


def build_rankings(ablation_name='ablation'):
    """Long table of per-run group rankings from every method that has run."""
    frames = []
    path = OUT / 'permutation_runs.csv'
    if path.exists():
        runs = pd.read_csv(path)
        observed = runs[runs.condition == 'permute_group']
        for (task, model, seed), block in observed.groupby(['task', 'model', 'seed']):
            ranked = rank_values(block.delta_log_loss.to_numpy(), block.group)
            for group, rank in ranked.items():
                frames.append(dict(method='group_permutation', task=task, model=model,
                                   seed=int(seed), group=group, rank=float(rank),
                                   value=float(block.loc[block.group == group,
                                                         'delta_log_loss'].iloc[0])))
    path = OUT / 'shapley_groups.csv'
    if path.exists():
        table = pd.read_csv(path)
        for record in table.itertuples():
            frames.append(dict(method='group_shapley', task=record.task,
                               model=record.model, seed=int(record.seed),
                               group=record.group, rank=float(record.rank),
                               value=float(record.mean_abs_phi)))
    path = ROOT / 'results' / 'group_experiments' / ablation_name / 'rankings.csv'
    if path.exists():
        table = pd.read_csv(path)
        for record in table.itertuples():
            frames.append(dict(method='retrain_ablation', task=record.task,
                               model=record.model, seed=int(record.seed),
                               group=record.group, rank=float(record.rank),
                               value=float(record.delta_log_loss)))
    if frames:
        pd.DataFrame(frames).to_csv(OUT / 'rankings.csv', index=False)


def figures_permutation(summary):
    for task in summary.task.unique():
        block = summary[summary.task == task]
        models = block.model.unique()
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.8),
                                 squeeze=False, sharey=True)
        for ax, model in zip(axes[0], models):
            part = block[block.model == model].sort_values('delta_log_loss_mean')
            colours = ['#b5453a' if p < 0.05 else '#7f8c8d'
                       for p in part.p_empirical]
            ax.barh(part.group, part.delta_log_loss_mean, color=colours)
            ax.axvline(0, color='black', lw=.8)
            if part.null_mean.notna().any():
                ax.axvline(part.null_mean.mean(), color='#2c6fbb', ls='--', lw=1)
            ax.set_title(model)
            ax.set_xlabel('Δ log-loss, group permuted jointly')
        fig.suptitle(f'Direct group permutation — {task}')
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'permutation_{task}.png', dpi=160)
        plt.close(fig)


def figures_shapley(table):
    for task in table.task.unique():
        block = table[table.task == task]
        models = block.model.unique()
        fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.8),
                                 squeeze=False, sharey=True)
        for ax, model in zip(axes[0], models):
            part = (block[block.model == model].groupby('group')['mean_abs_phi']
                    .mean().sort_values())
            ax.barh(part.index, part.values, color='#3b6ea5')
            ax.set_title(model)
            ax.set_xlabel('mean |φ| over explained rows')
        fig.suptitle(f'Exact group Shapley (interventional) — {task}')
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'shapley_{task}.png', dpi=160)
        plt.close(fig)


def write_report(args):
    lines = ['# Giải thích trực tiếp ở mức nhóm đặc trưng\n']
    lines.append('Sinh bởi `python/run_group_importance.py`. Thực nghiệm **mới**; '
                 'không ghi đè artifact cũ.\n')
    lines.append('## 1. Nguyên tắc\n')
    lines.append('- Mọi phép đo ở đây tính **trực tiếp trên nhóm**. Không lấy tổng SHAP/CFI '
                 'theo từng feature rồi gọi là mức nhóm.')
    lines.append('- Group permutation: hoán vị đồng thời toàn bộ khối cột của nhóm bằng một '
                 'hoán vị hàng duy nhất → giữ nguyên phân phối đồng thời trong nhóm, chỉ phá '
                 'quan hệ giữa nhóm và phần còn lại.')
    lines.append('- Group Shapley: tính **chính xác** trên 2^9 = 512 liên minh, hàm giá trị '
                 'interventional `v(S) = E_background[f(x_S, B_S^c)]`. Chín giá trị cộng lại '
                 'đúng bằng `f(x) − v(∅)` (có kiểm tra cộng tính).')
    lines.append(f'- Chia dữ liệu: `{args.split}`'
                 + (f' (khóa `{args.version}`)' if args.split == 'source_aware' else '')
                 + f'; {args.permutation_repeats} hoán vị/nhóm, {args.controls} khối đối '
                 'chứng cùng độ rộng.\n')
    lines.append('## 2. Kết quả permutation\n')
    path = OUT / 'permutation_summary.csv'
    if path.exists():
        summary = pd.read_csv(path)
        for task in summary.task.unique():
            lines.append(f'### {task}\n')
            block = summary[summary.task == task]
            display = block[['model', 'group', 'delta_log_loss_mean',
                             'delta_log_loss_sd', 'null_mean', 'p_empirical', 'rank']].copy()
            display.columns = ['model', 'nhóm', 'Δlog-loss TB', 'SD qua seed',
                               'null TB', 'p thực nghiệm', 'hạng']
            lines.append(table_text(display.round(5)) + '\n')
            lines.append('![Group permutation]'
                         f'(../{OUT.relative_to(ROOT)}/figures/'
                         f'permutation_{task}.png)\n')
    lines.append('## 3. Kết quả group Shapley\n')
    path = OUT / 'shapley_groups.csv'
    if path.exists():
        table = pd.read_csv(path)
        summary = (table.groupby(['task', 'model', 'group'])
                   .agg(mean_abs_phi=('mean_abs_phi', 'mean'),
                        sd_abs_phi=('mean_abs_phi', 'std'),
                        rank_mean=('rank', 'mean'))
                   .reset_index())
        for task in summary.task.unique():
            lines.append(f'### {task}\n')
            block = summary[summary.task == task].sort_values(
                ['model', 'mean_abs_phi'], ascending=[True, False])
            lines.append(table_text(block.round(5)) + '\n')
            lines.append('![Group Shapley]'
                         f'(../{OUT.relative_to(ROOT)}/figures/'
                         f'shapley_{task}.png)\n')
        checks = json.loads((OUT / 'shapley_checks.json').read_text())
        worst = max(c['additivity_max_error'] for c in checks)
        lines.append(f'- Kiểm tra cộng tính: sai số tuyệt đối lớn nhất trên mọi lượt chạy là '
                     f'**{worst:.3e}**. Giá trị nhỏ xác nhận phép phân rã đúng là Shapley '
                     'trên đúng hàm giá trị đã khai báo.\n')
    lines.append('## 4. Giới hạn\n')
    lines.append('- Hàm giá trị interventional phá phụ thuộc giữa các feature ở hai bên ranh '
                 'giới liên minh. Đây là hạn chế chuẩn của marginal Shapley, không phải lỗi '
                 'triển khai; group Shapley **có điều kiện** cần 512 bộ sinh riêng và chưa '
                 'được chạy — ghi nhận là việc mở, không xấp xỉ ngầm.')
    lines.append('- Group permutation đo mức phụ thuộc của **mô hình đã fit**, không đo thông '
                 'tin duy nhất trong dữ liệu. Nhóm bị thay thế tốt sẽ có hiệu ứng nhỏ.')
    lines.append('- Các seed dùng chung dataset nên SD qua seed là biến động chạy lại, không '
                 'phải khoảng tin cậy.\n')
    (ROOT / 'report').mkdir(exist_ok=True)
    suffix = args.out_name.removeprefix('importance').lstrip('_')
    name = f'group_importance_{suffix}.md' if suffix else 'group_importance.md'
    (ROOT / 'report' / name).write_text('\n'.join(lines))
    print(f'Generated report/{name}', flush=True)


def write_manifest(args):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/run_group_importance.py',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        config=CONFIG, methods=args.methods, tasks=args.tasks,
        models=args.models, seeds=args.seeds,
        split=args.split, split_version=args.version,
        ablation_name=args.ablation_name,
        permutation_repeats=args.permutation_repeats,
        permutation_controls=args.controls,
        shapley_models=args.shapley_models,
        shapley_explain=args.shapley_explain,
        shapley_background=args.shapley_background,
        shapley_value_function='logit P(malware) [binary]; log P(true class) [multiclass]',
        group_measure='direct (block permutation / exact Shapley over 9 players)',
    )
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--methods', default='permutation,shapley')
    parser.add_argument('--permutation-repeats', type=int, default=10)
    parser.add_argument('--controls', type=int, default=3)
    parser.add_argument('--shapley-models', default='xgboost',
                        help='Models to run exact group Shapley for')
    parser.add_argument('--shapley-explain', type=int, default=400)
    parser.add_argument('--shapley-background', type=int, default=64)
    parser.add_argument('--split', default='source_aware')
    parser.add_argument('--version', default='source_aware_v1')
    parser.add_argument('--out-name', default='importance',
                        help='results subdirectory under group_experiments')
    parser.add_argument('--ablation-name', default='ablation',
                        help='ablation results directory used for cross-method '
                             'rankings (e.g. ablation_v2)')
    args = parser.parse_args()
    global OUT
    OUT = ROOT / 'results' / 'group_experiments' / args.out_name
    args.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    args.models = [m.strip() for m in args.models.split(',') if m.strip()]
    args.seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    args.shapley_models = [m.strip() for m in args.shapley_models.split(',') if m.strip()]
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        parser.error(f'unknown tasks: {sorted(unknown)}')
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f'unknown models: {sorted(unknown)}')
    run(args)


if __name__ == '__main__':
    main()
