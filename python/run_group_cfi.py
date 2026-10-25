"""Direct group conditional feature importance (group CFI) with sampler gate.

What makes this a *group* measure
----------------------------------
For group g we condition on every feature outside g and draw the whole group
jointly from an ARF/FORDE density estimated on training rows only:

    x'_g ~ p(x_g | x_{-g}),   x'_{-g} = x_{-g}

The replacement block is therefore a joint draw, not k independent per-feature
draws. A group CFI obtained by summing feature-wise CFI would ignore exactly the
dependence this construction preserves, so it is not used here.

The sampler gate
----------------
A conditional importance is only interpretable if the conditional sampler
produces plausible rows. Every group is screened before its CFI may be used as
evidence:

* ``coverage``          fraction of test rows with a complete finite draw;
* ``negative_rate``     any negative count/average is physically invalid;
* ``out_of_range_rate`` synthetic values outside the training support,
                        compared with the same rate for real test rows;
* ``ks``                marginal distribution distance, versus the
                        half-sample distance of real data (the noise floor);
* ``corr_error``        within-group Spearman structure destroyed by the
                        sampler, versus the same statistic on two halves of
                        real data.

A group whose sampler fails is still reported, but flagged ``gate_pass = False``
and excluded from the conclusions. Its CFI is never presented as confirmed.

Outputs (new tree):
    results/group_experiments/cfi/input/...
    results/group_experiments/cfi/draws/...
    results/group_experiments/cfi/group_cfi.csv
    results/group_experiments/cfi/diagnostics_summary.csv
    results/group_experiments/cfi/feature_diagnostics.csv
    results/group_experiments/cfi/summary.csv
    results/group_experiments/cfi/figures/*.png
    results/group_experiments/cfi/manifest.json
    report/group_cfi.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from common import CONFIG, ROOT, adjust_bh, inference
from generate_report import table as table_text
from group_protocol import GROUPS, MODELS, TASKS, group_map, load_tasks, make_model

OUT = ROOT / 'results' / 'group_experiments' / 'cfi'
DRAWS = OUT / 'draws'
INPUT = OUT / 'input'

# Gate thresholds, fixed before looking at any result.
MIN_COVERAGE = 0.99
MAX_NEGATIVE_RATE = 0.0
OUT_OF_RANGE_SLACK = 0.02
CORR_ERROR_FACTOR = 2.0
CORR_ERROR_FLOOR = 0.15


def stratified_subset(y, size, seed):
    """Same helper as export_for_carfi.subset: deterministic, stratified."""
    index = np.arange(len(y))
    if size >= len(y):
        return index
    selected, _ = train_test_split(index, train_size=size, stratify=y,
                                   random_state=seed)
    return np.sort(selected)


def align_group(frame, columns, indices, repeats):
    """Return draws as (rows, repeats, k), enforcing the R/Python contract."""
    expected = pd.MultiIndex.from_product([indices, range(1, repeats + 1)],
                                          names=['test_position', 'replicate'])
    actual = pd.MultiIndex.from_frame(frame[['test_position', 'replicate']])
    if actual.has_duplicates:
        raise ValueError('duplicate (test_position, replicate) in draws')
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('draws do not cover every test position and replicate')
    ordered = frame.set_index(['test_position', 'replicate'])[columns].reindex(expected)
    return ordered.to_numpy(dtype=np.float64).reshape(len(indices), repeats, len(columns))


def row_loss(y, probability, n_classes):
    if n_classes == 2:
        p = np.clip(probability[:, 1], 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    p = np.clip(probability[np.arange(len(y)), y], 1e-15, 1.0)
    return -np.log(p)


def run_r(task_name, seed, task, train_rows, test_rows, repeats):
    """Write the plan, run the R sampler, return the draw directory."""
    INPUT.mkdir(parents=True, exist_ok=True)
    DRAWS.mkdir(parents=True, exist_ok=True)
    features = task.features
    train_path = INPUT / f'{task_name}_{seed}_train.csv'
    test_path = INPUT / f'{task_name}_{seed}_test.csv'
    task.frame.iloc[train_rows][features].to_csv(train_path, index=False)
    task.frame.iloc[test_rows][features].to_csv(test_path, index=False)
    plan = dict(root=str(ROOT), task=task_name, seed=seed, repeats=repeats,
                features=features, groups=group_map(features),
                train_csv=str(train_path), test_csv=str(test_path),
                out_dir=str(DRAWS), arf_num_trees=CONFIG['arf_num_trees'],
                arf_min_node_size=CONFIG['arf_min_node_size'])
    plan_path = INPUT / f'plan_{task_name}_{seed}.json'
    plan_path.write_text(json.dumps(plan, indent=2))
    start = time.time()
    result = subprocess.run(
        ['Rscript', str(ROOT / 'r' / 'group_conditional_sampling.R'), str(plan_path)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'R sampler failed:\n{result.stdout}\n{result.stderr}')
    print(result.stdout.strip(), flush=True)
    print(f'[cfi] R sampler for {task_name}/seed {seed} took {time.time() - start:.0f}s',
          flush=True)


def group_importance(task, model, test_rows, repeats, seed):
    """Compute per-row group CFI and sampler diagnostics for one seed."""
    features = task.features
    X = task.frame[features].to_numpy(dtype=np.float64)
    X_test = X[test_rows]
    y_test = task.y[test_rows]
    train_rows = np.setdiff1d(np.arange(len(X)), test_rows)
    baseline = row_loss(y_test, model.predict_proba(X_test), task.n_classes)

    rows, diagnostics = [], []
    for group in GROUPS:
        members = task.features_of(group)
        local = [features.index(c) for c in members]
        draws = pd.read_csv(DRAWS / f'draws_{task.name}_{seed}_{group}.csv')
        block = align_group(draws, members, np.arange(len(test_rows)), repeats)
        valid = np.isfinite(block).all(axis=(1, 2))
        differences = np.full(len(y_test), np.nan)
        if valid.any():
            repeated = np.repeat(X_test[valid], repeats, axis=0)
            repeated[:, local] = block[valid].reshape(-1, len(members))
            loss = row_loss(np.repeat(y_test[valid], repeats),
                            model.predict_proba(repeated), task.n_classes)
            differences[valid] = (loss.reshape(valid.sum(), repeats).mean(axis=1)
                                  - baseline[valid])
        stats = inference(differences, CONFIG['alpha'])
        stats['conditional_importance'] = stats.pop('importance')
        stats['conditional_importance_se'] = stats.pop('se')
        rows.append(dict(task=task.name, seed=seed, group=group,
                         n_features=len(members),
                         coverage=float(valid.mean()), **stats))
        diagnostics.append(sampler_diagnostics(
            task, group, members, block, valid, X_test, train_rows, X, seed))
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def sampler_diagnostics(task, group, members, block, valid, X_test, train_rows,
                        X, seed):
    """Screen the conditional sampler for one group."""
    train_block = X[np.ix_(train_rows, [task.features.index(c) for c in members])]
    low = train_block.min(axis=0)
    high = train_block.max(axis=0)

    synthetic_matrix = block.reshape(-1, len(members))
    real_matrix = X_test[:, [task.features.index(c) for c in members]]
    synthetic_finite = np.isfinite(synthetic_matrix).all(axis=1)
    synthetic_matrix = synthetic_matrix[synthetic_finite]
    real_matrix = real_matrix[synthetic_finite]

    negative = float((synthetic_matrix < 0).mean()) if synthetic_matrix.size else np.nan
    synth_out = float(((synthetic_matrix < low) | (synthetic_matrix > high)).mean()) \
        if synthetic_matrix.size else np.nan
    real_out = float(((real_matrix < low) | (real_matrix > high)).mean()) \
        if real_matrix.size else np.nan

    ks_synth = float(np.mean([ks_2samp(synthetic_matrix[:, j], real_matrix[:, j]).statistic
                              for j in range(len(members))]))
    half = len(real_matrix) // 2
    ks_null = float(np.mean([ks_2samp(real_matrix[:half, j], real_matrix[half:, j]).statistic
                             for j in range(len(members))]))

    if len(members) > 1 and synthetic_matrix.shape[0] > 10:
        corr_synth = pd.DataFrame(synthetic_matrix).corr(method='spearman').to_numpy()
        corr_real = pd.DataFrame(real_matrix).corr(method='spearman').to_numpy()
        corr_error = float(np.linalg.norm(np.nan_to_num(corr_synth - corr_real))
                           / np.sqrt(len(members) * (len(members) - 1)))
        corr_null = float(np.linalg.norm(np.nan_to_num(
            pd.DataFrame(real_matrix[:half]).corr(method='spearman').to_numpy()
            - pd.DataFrame(real_matrix[half:]).corr(method='spearman').to_numpy()))
            / np.sqrt(len(members) * (len(members) - 1)))
    else:
        corr_error = corr_null = 0.0

    reasons = []
    if float(valid.mean()) < MIN_COVERAGE:
        reasons.append('coverage')
    if not np.isnan(negative) and negative > MAX_NEGATIVE_RATE:
        reasons.append('negative values')
    if not np.isnan(synth_out) and synth_out > real_out + OUT_OF_RANGE_SLACK:
        reasons.append('outside train support')
    if corr_error > max(CORR_ERROR_FACTOR * corr_null, CORR_ERROR_FLOOR):
        reasons.append('within-group structure')

    return dict(task=task.name, seed=seed, group=group, n_features=len(members),
                coverage=float(valid.mean()), negative_rate=negative,
                out_of_range_synth=synth_out, out_of_range_real=real_out,
                ks_synth=ks_synth, ks_null=ks_null,
                corr_error=corr_error, corr_null=corr_null,
                gate_pass=not reasons, gate_reasons=';'.join(reasons))


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    tasks = load_tasks(args.tasks)
    cfi_rows, diagnostic_rows, meta_rows = [], [], []
    for task_name, task in tasks.items():
        for seed in args.seeds:
            train, test = task.split(seed)
            arf_train = stratified_subset(task.y[train], args.arf_train, seed)
            importance_test = stratified_subset(task.y[test], args.test_samples, seed)
            run_r(task_name, seed, task, train[arf_train], test[importance_test],
                  args.repeats)
            model = make_model(args.model, task, seed)
            model.fit(task.frame.iloc[train][task.features].to_numpy(dtype=np.float64),
                      task.y[train])
            importance, diagnostics = group_importance(
                task, model, test[importance_test], args.repeats, seed)
            cfi_rows.append(importance)
            diagnostic_rows.append(diagnostics)
            meta = json.loads((DRAWS / f'arf_meta_{task_name}_{seed}.json').read_text())
            meta_rows.append(meta)
            print(f'[cfi] {task_name}/seed {seed} done', flush=True)

    cfi = pd.concat(cfi_rows, ignore_index=True)
    cfi['q_value'] = np.nan
    for _, index in cfi.groupby(['task', 'seed']).groups.items():
        cfi.loc[index, 'q_value'] = adjust_bh(cfi.loc[index, 'p_value'])
    cfi['rank'] = (cfi.groupby(['task', 'seed'])['conditional_importance']
                   .rank(ascending=False, method='min').astype(int))
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    cfi = cfi.merge(diagnostics[['task', 'seed', 'group', 'gate_pass', 'gate_reasons']],
                    on=['task', 'seed', 'group'], how='left')
    cfi.to_csv(OUT / 'group_cfi.csv', index=False)
    diagnostics.to_csv(OUT / 'diagnostics_summary.csv', index=False)
    collect_feature_diagnostics()
    summary = summarise(cfi, diagnostics)
    summary.to_csv(OUT / 'summary.csv', index=False)
    (OUT / 'arf_meta.json').write_text(json.dumps(meta_rows, indent=2))
    figures(cfi, diagnostics, summary)
    write_report(cfi, diagnostics, summary, args)
    write_manifest(args)
    print(summary.to_string(index=False), flush=True)
    print(f'Group CFI artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def collect_feature_diagnostics():
    frames = [pd.read_csv(path) for path in sorted(DRAWS.glob('diagnostics_*.csv'))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            OUT / 'feature_diagnostics.csv', index=False)


def summarise(cfi, diagnostics):
    rows = []
    for (task, group), block in cfi.groupby(['task', 'group']):
        gate = diagnostics[(diagnostics.task == task) & (diagnostics.group == group)]
        rows.append(dict(
            task=task, group=group, n_features=int(block.n_features.iloc[0]),
            n_seeds=int(block.seed.nunique()),
            cfi_mean=float(block.conditional_importance.mean()),
            cfi_sd=float(block.conditional_importance.std(ddof=1))
            if len(block) > 1 else float('nan'),
            cfi_median=float(block.conditional_importance.median()),
            coverage_min=float(block.coverage.min()),
            n_significant_bh=int(((block.q_value < CONFIG['alpha'])
                                  & (block.ci_lo > 0)).sum()),
            n_seeds_significant=int(
                block.assign(sig=(block.q_value < CONFIG['alpha']) & (block.ci_lo > 0))
                .groupby('seed').sig.any().sum()),
            gate_pass=bool(gate.gate_pass.all()) if len(gate) else False,
            gate_reasons=';'.join(sorted(set(r for r in gate.gate_reasons.fillna('')
                                             if r))),
            negative_rate_max=float(gate.negative_rate.max()) if len(gate) else np.nan,
            out_of_range_synth_max=float(gate.out_of_range_synth.max())
            if len(gate) else np.nan,
            out_of_range_real_max=float(gate.out_of_range_real.max())
            if len(gate) else np.nan,
            corr_error_max=float(gate.corr_error.max()) if len(gate) else np.nan,
            corr_null_max=float(gate.corr_null.max()) if len(gate) else np.nan,
        ))
    summary = pd.DataFrame(rows)
    summary['rank'] = (summary.groupby('task')['cfi_mean']
                       .rank(ascending=False, method='min').astype(int))
    return summary.sort_values(['task', 'rank'])


def figures(cfi, diagnostics, summary):
    for task in summary.task.unique():
        block = summary[summary.task == task].sort_values('cfi_mean')
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colours = ['#2c6fbb' if p else '#b0b0b0' for p in block.gate_pass]
        ax.barh(block.group, block.cfi_mean, xerr=block.cfi_sd, color=colours)
        ax.axvline(0, color='black', lw=.8)
        ax.set(xlabel='Group CFI (Δ log-loss, mean ± SD over seeds)',
               title=f'Direct group CFI — {task}\n(grey = sampler gate failed)')
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'group_cfi_{task}.png', dpi=160)
        plt.close(fig)

        part = diagnostics[diagnostics.task == task]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].scatter(part.out_of_range_real, part.out_of_range_synth, c='#3b6ea5')
        axes[0].plot([0, 1], [0, 1], 'r--', lw=1)
        axes[0].set(xlabel='real test out-of-support rate',
                    ylabel='synthetic out-of-support rate', title='Support')
        axes[1].scatter(part.ks_null, part.ks_synth, c='#3b6ea5')
        axes[1].plot([0, 1], [0, 1], 'r--', lw=1)
        axes[1].set(xlabel='KS noise floor (real halves)',
                    ylabel='KS synthetic vs real', title='Marginals')
        axes[2].scatter(part.corr_null, part.corr_error, c='#3b6ea5')
        axes[2].plot([0, 1], [0, 1], 'r--', lw=1)
        axes[2].set(xlabel='correlation noise floor',
                    ylabel='within-group structure error', title='Dependence')
        fig.suptitle(f'Sampler diagnostics — {task} (points above the red line are worse)')
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'sampler_diagnostics_{task}.png', dpi=160)
        plt.close(fig)


def write_report(cfi, diagnostics, summary, args):
    lines = ['# Group CFI trực tiếp và chẩn đoán bộ sinh ARF/FORDE\n']
    lines.append('Sinh bởi `python/run_group_cfi.py` + `r/group_conditional_sampling.R`. '
                 'Thực nghiệm **mới**; không ghi đè artifact cũ.\n')
    lines.append('## 1. Thiết kế\n')
    lines.append('- Với mỗi nhóm g: điều kiện trên **toàn bộ** feature ngoài nhóm, lấy mẫu '
                 '**đồng thời cả nhóm** từ ARF/FORDE đã fit chỉ trên train.')
    lines.append('- ARF chỉ thấy train; test chỉ xuất hiện dưới dạng evidence. Không có '
                 'unconditional fallback: hàng không có support bị bỏ và ghi vào coverage.')
    lines.append(f'- {args.repeats} draw/hàng, {args.arf_train} hàng ARF train, '
                 f'{args.test_samples} hàng test, model `{args.model}`.')
    lines.append('- Ngưỡng gate **đặt trước**: coverage ≥ '
                 f'{MIN_COVERAGE}, không có giá trị âm, tỉ lệ ngoài support không vượt '
                 f'quá {OUT_OF_RANGE_SLACK} so với dữ liệu thật, sai lệch cấu trúc trong '
                 f'nhóm ≤ max({CORR_ERROR_FACTOR}× null, {CORR_ERROR_FLOOR}).\n')
    lines.append('## 2. Chẩn đoán bộ sinh\n')
    display = diagnostics[['task', 'seed', 'group', 'coverage', 'negative_rate',
                           'out_of_range_synth', 'out_of_range_real', 'ks_synth',
                           'ks_null', 'corr_error', 'corr_null', 'gate_pass',
                           'gate_reasons']].copy()
    display.columns = ['task', 'seed', 'nhóm', 'coverage', 'tỉ lệ âm',
                       'ngoài support (synth)', 'ngoài support (real)',
                       'KS synth', 'KS null', 'lỗi cấu trúc', 'null cấu trúc',
                       'gate', 'lý do']
    lines.append(table_text(display.round(5)) + '\n')
    lines.append('![Chẩn đoán bộ sinh](../results/group_experiments/cfi/figures/'
                 'sampler_diagnostics_binary.png)\n')
    lines.append('## 3. Group CFI\n')
    display = summary[['task', 'group', 'n_features', 'cfi_mean', 'cfi_sd',
                       'coverage_min', 'n_seeds_significant', 'gate_pass',
                       'gate_reasons', 'rank']].copy()
    display.columns = ['task', 'nhóm', 'số feature', 'CFI TB', 'SD qua seed',
                       'coverage min', 'số seed có q<0.05 & ci_lo>0', 'gate',
                       'lý do fail', 'hạng']
    lines.append(table_text(display.round(6)) + '\n')
    lines.append('![Group CFI](../results/group_experiments/cfi/figures/'
                 'group_cfi_binary.png)\n')
    lines.append('## 4. Kết luận được phép và không được phép\n')
    failed = summary[~summary.gate_pass]
    if len(failed):
        lines.append('- Nhóm **fail gate** (không dùng CFI làm bằng chứng): ' +
                     ', '.join(f'`{r.task}/{r.group}` ({r.gate_reasons})'
                               for r in failed.itertuples()) + '.')
    else:
        lines.append('- Mọi nhóm đều qua gate ở mọi seed đã chạy.')
    passed = summary[summary.gate_pass & (summary.n_seeds_significant > 0)]
    if len(passed):
        lines.append('- Nhóm qua gate **và** có hiệu ứng dương nhất quán: ' +
                     ', '.join(f'`{r.task}/{r.group}`' for r in passed.itertuples()) + '.')
    lines.append('- Bất kỳ nhóm fail gate nào cũng chỉ được báo cáo như **hạn chế của bộ '
                 'sinh**, không được trích như bằng chứng XAI đã xác nhận.\n')
    lines.append('## 5. Giới hạn\n')
    lines.append('- CFI đo mức tăng log-loss khi thay nhóm bằng mẫu có điều kiện; nó phụ '
                 'thuộc chất lượng ARF/FORDE và vào support của train.')
    lines.append('- FORGE làm tròn theo số chữ số thập phân quan sát trong train, nên giá '
                 'trị synthetic không phải là bản sao liên tục của phân phối thật.')
    lines.append('- ARF fit trên một tập con train; kết quả có thể thay đổi nếu tăng '
                 '`arf_train_samples`.')
    lines.append('- Các seed dùng chung dataset, SD qua seed không phải khoảng tin cậy.\n')
    (ROOT / 'report').mkdir(exist_ok=True)
    (ROOT / 'report' / 'group_cfi.md').write_text('\n'.join(lines))
    print('Generated report/group_cfi.md', flush=True)


def write_manifest(args):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/run_group_cfi.py',
        sampler='r/group_conditional_sampling.R',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        config=CONFIG, tasks=args.tasks, seeds=args.seeds, model=args.model,
        repeats=args.repeats, arf_train=args.arf_train,
        test_samples=args.test_samples,
        conditioning='all features outside the group, group sampled jointly',
        gate=dict(min_coverage=MIN_COVERAGE,
                  max_negative_rate=MAX_NEGATIVE_RATE,
                  out_of_range_slack=OUT_OF_RANGE_SLACK,
                  corr_error_factor=CORR_ERROR_FACTOR,
                  corr_error_floor=CORR_ERROR_FLOOR),
    )
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--model', default='xgboost')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--arf-train', type=int, default=3000)
    parser.add_argument('--test-samples', type=int, default=1500)
    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    args.seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if args.model not in MODELS:
        parser.error(f'unknown model: {args.model}')
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        parser.error(f'unknown tasks: {sorted(unknown)}')
    run(args)


if __name__ == '__main__':
    main()
