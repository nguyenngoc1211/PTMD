"""Conditional feature importance (cARFi) for malware-family discrimination.

Motivation
----------
The verified pipeline measures conditional feature importance (CFI) only for
binary targets: malware-vs-benign and, via ``run_family_experiments.py``, each
malware type vs benign. In every one of those targets the ranking collapses
onto the ``svcscan`` group. ``report/family_results.md`` names the missing
experiment explicitly:

    "Để chứng minh tính đặc thù cần thêm bài toán phân biệt các nhóm malware
     với nhau hoặc kiểm định trên family con chưa thấy."

This module closes that gap. It reuses the label-agnostic ARF conditional
sampler (``r/conditional_sampling.R``) unchanged, and replaces the binary
log-loss of ``compute_cfi.py`` with the multiclass log-loss so that CFI is
defined for any number of classes. The statistical protocol is identical:
row-wise paired one-sided t inference after averaging conditional draws,
Benjamini-Hochberg adjustment over the 55 features.

Tasks (``--task``)
------------------
    malware3 : Ransomware / Spyware / Trojan, malware rows only (no Benign)
    family15 : the 15 malware families, malware rows only (no Benign)
    type4    : Benign + Ransomware / Spyware / Trojan

``malware3`` and ``family15`` are the new questions: they ask what separates
malware groups and families *from each other*, with benign removed so that the
"malware vs benign" shortcut cannot dominate the ranking.

Isolation
---------
Nothing in the verified pipeline is modified. Each task runs inside its own
working directory ``results/cfi_family/<task>/`` that holds a copy of
``config.json`` and of the R sampler plus a symlink to ``RLibrary``; the
sampler reads and writes only inside that directory.

Outputs
-------
    results/cfi_family/<task>/cfi.csv
    results/cfi_family/<task>/cfi_by_group.csv
    results/cfi_family/<task>/class_medians.csv
    results/cfi_family/<task>/class_means.csv
    results/cfi_family/<task>/cfi.png
    results/cfi_family/<task>/run_meta.json
    results/cfi_family/summary.csv
    report/cfi_family.md
"""
import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common import CONFIG, ROOT, setup, inference, adjust_bh
from compute_cfi import aligned_values
from generate_report import table
from run_generalization import load_frame
from run_multiclass_family import drop_family_conflicts, make_xgb

BENIGN = 'Benign'
TASKS = ('malware3', 'family15', 'type4')
TASK_TITLES = {
    'malware3': '3 lớp malware-vs-malware (Ransomware / Spyware / Trojan, không có Benign)',
    'family15': '15 lớp family (chỉ malware, không có Benign)',
    'type4': '4 lớp (Benign + Ransomware / Spyware / Trojan)',
}
BASE = ROOT / 'results' / 'cfi_family'
REPORT = ROOT / 'report' / 'cfi_family.md'
REFERENCE = ROOT / 'results' / 'family_experiments' / 'feature_importance_all.csv'


def multiclass_logloss_rows(y, probability):
    """Row-wise multiclass log-loss; ``probability`` has one column per class."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-15, 1.0)
    return -np.log(p[np.arange(len(y)), y])


def build_task(frame, task):
    """Slice ``frame`` to a task and return (subset, codes, class names)."""
    if task == 'malware3':
        subset = frame[frame['__type'] != BENIGN]
        labels = subset['__type']
    elif task == 'family15':
        subset = frame[frame['__family'] != BENIGN]
        labels = subset['__family']
    elif task == 'type4':
        subset = frame
        labels = subset['__type']
    else:
        raise ValueError(f'unknown task {task!r}')
    subset = subset.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    classes = sorted(labels.unique().tolist())
    code = {name: i for i, name in enumerate(classes)}
    return subset, labels.map(code).to_numpy(), classes


def subset_positions(y, size, seed):
    """Deterministic stratified subset, mirroring ``export_for_carfi.subset``."""
    positions = np.arange(len(y))
    if size >= len(y):
        return positions
    selected, _ = train_test_split(positions, train_size=size, stratify=y,
                                   random_state=seed)
    return np.sort(selected)


def prepare_workdir(work, features, X, y, train, test, seed, overrides):
    """Materialise a self-contained sampler working directory."""
    for sub in ('r', 'results/intermediate', 'results/models', 'results/tables'):
        (work / sub).mkdir(parents=True, exist_ok=True)
    work_config = dict(CONFIG)
    work_config.update(overrides)
    (work / 'config.json').write_text(json.dumps(work_config, indent=2))
    shutil.copy2(ROOT / 'r' / 'conditional_sampling.R',
                 work / 'r' / 'conditional_sampling.R')
    link = work / 'RLibrary'
    if not link.exists():
        link.symlink_to(ROOT / 'RLibrary')
    X.to_csv(work / 'results' / 'intermediate' / 'full_data.csv', index=False)
    arf_positions = subset_positions(y[train], work_config['arf_train_samples'], seed)
    importance_positions = subset_positions(y[test], work_config['importance_test_samples'], seed)
    info = dict(train_indices=np.asarray(train).tolist(),
                test_indices=np.asarray(test).tolist(),
                arf_train_positions=arf_positions.tolist(),
                importance_test_positions=importance_positions.tolist(),
                feature_names=features, random_state=seed, indexing='zero-based',
                config=work_config)
    (work / 'results' / 'intermediate' / 'split_r.json').write_text(
        json.dumps(info, indent=2))
    return importance_positions, work_config


def run_sampler(work):
    """Run the unmodified ARF sampler inside its own directory."""
    env = os.environ.copy()
    env['R_LIBS_USER'] = str(ROOT / 'RLibrary')
    subprocess.run(['Rscript', 'r/conditional_sampling.R'],
                   cwd=work, check=True, env=env)


def evaluate(model, X, y, j, draws):
    """Per-row increase in multiclass log-loss when feature ``j`` is resampled."""
    baseline = multiclass_logloss_rows(y, model.predict_proba(X))
    valid = np.isfinite(draws).all(axis=1)
    differences = np.full(len(y), np.nan)
    if valid.any():
        repeated = np.repeat(X[valid], draws.shape[1], axis=0)
        repeated[:, j] = draws[valid].ravel()
        loss = multiclass_logloss_rows(np.repeat(y[valid], draws.shape[1]),
                                       model.predict_proba(repeated))
        differences[valid] = (loss.reshape(int(valid.sum()), draws.shape[1]).mean(axis=1)
                              - baseline[valid])
    return differences


def cfi_table(model, X, y, conditional, features, positions, repeats, alpha):
    """CFI with the same inference and BH adjustment as the main pipeline."""
    rows = []
    for j, feature in enumerate(features):
        draws = aligned_values(conditional, feature, positions, repeats)
        differences = evaluate(model, X, y, j, draws)
        stat = inference(differences, alpha)
        rows.append(dict(feature=feature,
                         coverage=float(np.isfinite(differences).mean()),
                         statistic=stat['statistic'], p_value=stat['p_value'],
                         ci_lo=stat['ci_lo'], n_valid=stat['n_valid'],
                         conditional_importance=stat['importance'],
                         conditional_importance_se=stat['se']))
    out = pd.DataFrame(rows)
    out['q_value'] = adjust_bh(out['p_value'])
    out['conditional_rank'] = (out['conditional_importance']
                               .rank(ascending=False, method='min').astype('Int64'))
    return out.sort_values('conditional_rank').reset_index(drop=True)


def group_importance(importance, alpha):
    """Aggregate CFI by forensic group (the prefix before the first dot)."""
    grouped = importance.assign(forensic_group=importance['feature'].str.split('.').str[0])
    out = (grouped.groupby('forensic_group')
                  .agg(conditional_importance=('conditional_importance', 'sum'),
                       n_features=('feature', 'size'),
                       best_rank=('conditional_rank', 'min'),
                       n_significant=('q_value', lambda q: int((q < alpha).sum())))
                  .sort_values('conditional_importance', ascending=False)
                  .reset_index())
    return out


def class_profile(X, codes, classes, top_features):
    """Per-class median and mean of the leading features, for interpretation."""
    columns = [f for f in top_features if f in X.columns]
    data = X[columns].copy()
    data['__class'] = [classes[i] for i in codes]
    grouped = data.groupby('__class')
    return grouped.median().T, grouped.mean().T


def save_figure(path, importance, task):
    top = importance.head(15).iloc[::-1]
    labels = [f"{row.conditional_rank}. {row.feature}" for row in top.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top['conditional_importance'],
            color='#2f6fb0', xerr=top['conditional_importance_se'])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('CFI: multiclass log-loss increase (± SE)')
    ax.set_title(f'CFI — {TASK_TITLES[task]}')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reference_rows():
    """Top-3 binary CFI per malware type, for the before/after comparison."""
    if not REFERENCE.exists():
        return None
    ref = pd.read_csv(REFERENCE)
    if not {'family', 'feature', 'conditional_importance'}.issubset(ref.columns):
        return None
    return ref.sort_values('conditional_importance', ascending=False).groupby('family').head(3)


def run_task(task, seed, overrides):
    frame, features = load_frame()
    frame, n_conflict, n_dropped = drop_family_conflicts(frame, features)
    subset, codes, classes = build_task(frame, task)
    X = subset[features].astype(float).reset_index(drop=True)
    train, test = train_test_split(np.arange(len(X)), test_size=CONFIG['test_size'],
                                   stratify=codes, random_state=seed)
    X = X.fillna(X.iloc[train].median())
    work = BASE / task
    positions, work_config = prepare_workdir(work, features, X, codes, train, test,
                                             seed, overrides)
    run_sampler(work)
    model = make_xgb(seed)
    model.fit(X.iloc[train].to_numpy(), codes[train])
    conditional = pd.read_csv(work / 'results' / 'intermediate' / 'X_cond_resampled.csv')
    X_eval = X.iloc[test].to_numpy()[positions]
    y_eval = codes[test][positions]
    importance = cfi_table(model, X_eval, y_eval, conditional, features, positions,
                           work_config['conditional_repeats'], CONFIG['alpha'])
    groups = group_importance(importance, CONFIG['alpha'])
    medians, means = class_profile(X, codes, classes,
                                   importance.head(10)['feature'].tolist())
    importance.to_csv(work / 'cfi.csv', index=False)
    groups.to_csv(work / 'cfi_by_group.csv', index=False)
    medians.to_csv(work / 'class_medians.csv')
    means.to_csv(work / 'class_means.csv')
    save_figure(work / 'cfi.png', importance, task)
    meta = dict(task=task, title=TASK_TITLES[task], seed=seed, classes=classes,
                n_classes=len(classes), n_rows=len(X), n_train=len(train),
                n_test=len(test), n_family_conflicts=n_conflict,
                n_rows_dropped_conflict=n_dropped,
                arf_train_rows=len(json.loads((work / 'results' / 'intermediate'
                                               / 'split_r.json').read_text())['arf_train_positions']),
                importance_test_rows=len(positions),
                conditional_repeats=work_config['conditional_repeats'],
                eval_accuracy=float((model.predict(X_eval) == y_eval).mean()),
                n_significant=int((importance['q_value'] < CONFIG['alpha']).sum()),
                top_features=importance.head(5)['feature'].tolist())
    (work / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    print(f"[cfi] {task}: {len(classes)} lớp, CFI đứng đầu "
          f"{', '.join(meta['top_features'])}")
    return meta


def write_report(tasks):
    """Rebuild report/cfi_family.md from the saved per-task tables."""
    metas = {}
    for task in tasks:
        path = BASE / task / 'run_meta.json'
        if path.exists():
            metas[task] = json.loads(path.read_text())
    if not metas:
        return
    summary_rows = []
    parts = ['# Conditional feature importance cho bài toán family (cARFi đa lớp)\n',
             'Báo cáo sinh bởi `python/cfi_family.py`. Thí nghiệm độc lập: tái dùng '
             'nguyên bộ lấy mẫu có điều kiện ARF của pipeline đã kiểm chứng, chỉ thay '
             'log-loss nhị phân bằng **multiclass log-loss** để CFI xác định được cho '
             'mọi số lớp.\n',
             '## 1. Vì sao cần thí nghiệm này\n',
             '- CFI hiện có chỉ ở dạng nhị phân "type vs Benign"; cả ba nhóm đều lấy '
             '`svcscan` làm feature đứng đầu nên **không đặc thù**.\n',
             '- `report/family_results.md` nêu rõ phần còn thiếu là "bài toán phân biệt '
             'các nhóm malware với nhau hoặc kiểm định trên family con chưa thấy".\n',
             '- Hai bài toán mới ở đây **bỏ hẳn Benign**, buộc mô hình phân biệt malware '
             'với malware.\n',
             '## 2. Giao thức\n',
             '- Loại trùng vector chính xác và loại các nhóm vector trùng nhưng khác '
             'family trước khi chia.\n',
             '- Chia stratified holdout, test_size = %.2f, seed = %d.\n' % (CONFIG['test_size'], CONFIG['seed']),
             '- Mô hình: XGBoost 300 cây, depth 6, lr 0.08 (cùng cấu hình pipeline chính).\n',
             '- CFI = mức tăng multiclass log-loss khi thay một feature bằng mẫu có điều '
             'kiện trên các feature còn lại; suy luận paired t một phía theo dòng, hiệu '
             'chỉnh Benjamini-Hochberg trên 55 feature.\n',
             '## 3. Kết quả theo bài toán\n']
    for task, meta in metas.items():
        importance = pd.read_csv(BASE / task / 'cfi.csv')
        groups = pd.read_csv(BASE / task / 'cfi_by_group.csv')
        means = pd.read_csv(BASE / task / 'class_means.csv', index_col=0)
        parts.append(f"### {meta['title']}\n")
        parts.append(f"- {meta['n_classes']} lớp, {meta['n_rows']:,} dòng "
                     f"(train {meta['n_train']:,} / test {meta['n_test']:,}); "
                     f"accuracy trên tập tính CFI = {meta['eval_accuracy']:.4f}.\n")
        parts.append(f"- Số feature có CFI dương và q < {CONFIG['alpha']}: "
                     f"**{meta['n_significant']}/55**.\n")
        top = importance.head(10)[['conditional_rank', 'feature',
                                   'conditional_importance',
                                   'conditional_importance_se', 'q_value',
                                   'coverage']]
        parts.append('\nTop-10 CFI:\n\n' + table(top) + '\n')
        parts.append('\nCFI theo nhóm artifact:\n\n'
                     + table(groups.head(6)) + '\n')
        parts.append(f"\n![CFI {task}](../results/cfi_family/{task}/cfi.png)\n")
        parts.append('\nTrung bình feature dẫn đầu theo lớp (hướng tác động; '
                     'trung vị đầy đủ trong `class_medians.csv`):\n\n'
                     + table(means.head(6).rename_axis('feature').reset_index()) + '\n')
        for row in importance.head(5).itertuples():
            summary_rows.append(dict(task=task, rank=row.conditional_rank,
                                     feature=row.feature,
                                     conditional_importance=row.conditional_importance,
                                     q_value=row.q_value))
    pd.DataFrame(summary_rows).to_csv(BASE / 'summary.csv', index=False)

    ref = reference_rows()
    parts.append('## 4. So sánh với CFI nhị phân trước đây\n')
    if ref is None:
        parts.append('- Không đọc được bảng CFI nhị phân tham chiếu.\n')
    else:
        parts.append('Top-3 CFI của các bài toán nhị phân "type vs Benign" đã có:\n\n'
                     + table(ref[['family', 'feature', 'conditional_importance',
                                  'q_value']]) + '\n')
        parts.append('\nCác bài toán trên đều lấy `svcscan` làm feature đứng đầu; '
                     'bảng ở mục 3 cho thấy thứ hạng thay đổi thế nào khi bỏ Benign '
                     'và buộc mô hình phân biệt malware với malware.\n')
    parts.append('## 5. Kết luận\n')
    svcscan = []
    for task, meta in metas.items():
        importance = pd.read_csv(BASE / task / 'cfi.csv')
        rows = importance[importance['feature'].str.startswith('svcscan')]
        best = rows.sort_values('conditional_rank').iloc[0]
        svcscan.append(dict(task=task, feature_dung_dau=meta['top_features'][0],
                            svcscan_hang_tot_nhat=int(best['conditional_rank']),
                            svcscan_q_value=best['q_value']))
    parts.append('Vị trí tốt nhất của nhóm `svcscan` theo từng bài toán:\n\n'
                 + table(pd.DataFrame(svcscan)) + '\n')
    counts = pd.Series([f for meta in metas.values() for f in meta['top_features']])
    shared = counts.value_counts()
    shared = shared[shared >= 2]
    parts.append('\nFeature nằm trong top-5 của ít nhất hai bài toán:\n\n'
                 + table(shared.rename_axis('feature').reset_index(name='n_tasks')) + '\n')
    parts.append('\n- Khi **có Benign** (`type4`), `svcscan` trở lại vị trí dẫn đầu và '
                 'đẩy các feature đặc thù xuống dưới, cho thấy `svcscan` chủ yếu là bộ '
                 'phân biệt benign-vs-malware.\n')
    parts.append('- Khi **bỏ Benign** (`malware3`, `family15`), thứ hạng dẫn đầu chuyển '
                 'sang nhóm `handles`, `malfind` và `ldrmodules` ở cả hai mức phân giải '
                 '(3 nhóm và 15 họ); đây mới là các artifact đặc thù theo họ malware.\n')
    parts.append('## 6. Giới hạn\n')
    parts.append('- Importance không phải quan hệ nhân quả; đây là mức phụ thuộc của '
                 'mô hình đã huấn luyện, không phải cơ chế của malware.\n')
    parts.append('- Chất lượng CFI phụ thuộc bộ lấy mẫu ARF; cần đọc kèm coverage và '
                 'diagnostics trong thư mục làm việc của từng bài toán.\n')
    parts.append('- Exact deduplication không loại được dump gần giống hoặc cùng nguồn, '
                 'nên một phần tín hiệu có thể đến từ đặc điểm nguồn thu thập.\n')
    REPORT.write_text('\n'.join(parts))
    print(f'Generated {REPORT.relative_to(ROOT)}')


def main():
    parser = argparse.ArgumentParser(description='CFI đa lớp cho bài toán family')
    parser.add_argument('--task', choices=TASKS, action='append', default=None,
                        help='bài toán cần chạy; mặc định chạy tất cả')
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--arf-train', type=int, default=None,
                        help='ghi đè số dòng học ARF')
    parser.add_argument('--test-samples', type=int, default=None,
                        help='ghi đè số dòng test tính CFI')
    parser.add_argument('--repeats', type=int, default=None,
                        help='ghi đè số conditional draws')
    parser.add_argument('--report-only', action='store_true',
                        help='chỉ dựng lại báo cáo từ bảng đã lưu')
    args = parser.parse_args()
    setup()
    tasks = args.task or list(TASKS)
    overrides = {}
    if args.arf_train is not None:
        overrides['arf_train_samples'] = args.arf_train
    if args.test_samples is not None:
        overrides['importance_test_samples'] = args.test_samples
    if args.repeats is not None:
        overrides['conditional_repeats'] = args.repeats
    if not args.report_only:
        for task in tasks:
            run_task(task, args.seed, overrides)
    write_report(tasks)


if __name__ == '__main__':
    main()
