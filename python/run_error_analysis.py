"""Multiclass error analysis on the locked source-aware splits.

``run_split_impact`` reports scalar metrics per split; this script keeps the
confusion structure. For each task it evaluates two closed-set questions:

* ``with_benign`` — all classes of the task (16 families or 4 types);
* ``malware_only`` — the same locked split, restricted to malware rows, which
  removes the easy Benign-vs-malware decision and exposes family/type confusions.

Closed-set family identification is *not* the same question as leave-one-family
-out binary detection (``run_generalization.py``): the closed-set model has seen
every family during training, while LOFO asks whether a never-seen family can be
flagged as malware. The report states this separation explicitly.

No tuning is performed; every model uses ``group_protocol.make_model`` and the
split is the locked ``source_aware_v1`` file, so rows from one specimen never
appear on both sides.

Outputs:
    results/error_analysis/<out-name>/runs.csv
    results/error_analysis/<out-name>/per_class.csv
    results/error_analysis/<out-name>/summary.csv
    results/error_analysis/<out-name>/top_confusions.csv
    results/error_analysis/<out-name>/confusion_{counts,normalized}_*.csv
    results/error_analysis/<out-name>/figures/*.png
    report/error_analysis_<suffix>.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support)

from common import CONFIG, ROOT
from generate_report import table as table_text
from group_protocol import BENIGN, build_task, load_base_frame, make_model, score
from split_manager import load_locked

OUT = ROOT / 'results' / 'error_analysis' / 'error_v1'
VARIANTS = ('with_benign', 'malware_only')
PAIRS = (('Transponder', 'CWS'), ('Transponder', 'Gator'), ('Pysa', 'Zeus'))


def malware_only(task):
    """Same task restricted to malware rows, with labels renumbered from 0."""
    keep = np.array([task.classes[k] != BENIGN for k in task.y])
    classes = [c for c in task.classes if c != BENIGN]
    remap = {task.classes.index(c): i for i, c in enumerate(classes)}
    y = np.array([remap[k] for k in task.y[keep]], dtype=int)
    frame = task.frame.loc[keep].reset_index(drop=True)
    return replace(task, frame=frame, y=y, classes=classes), keep


def variant_arrays(task, train, test, variant):
    """Feature matrix, labels, and split masks for one variant."""
    X = task.frame[task.features].to_numpy(dtype=np.float64)
    train_mask = np.zeros(len(task.y), dtype=bool)
    train_mask[train] = True
    test_mask = np.zeros(len(task.y), dtype=bool)
    test_mask[test] = True
    if variant == 'with_benign':
        return task, X, task.y, train_mask, test_mask
    if variant == 'malware_only':
        reduced, keep = malware_only(task)
        return (reduced, X[keep], reduced.y, train_mask[keep], test_mask[keep])
    raise ValueError(f'unknown variant {variant!r}')


def confusion_frames(matrix, classes):
    """Counts and row-normalized confusion matrices with class labels."""
    counts = pd.DataFrame(matrix, index=classes, columns=classes)
    counts.index.name = 'true'
    totals = counts.sum(axis=1).replace(0, np.nan)
    normalized = counts.div(totals, axis=0).fillna(0.0)
    return counts, normalized


def top_confusions(matrix, classes):
    """Off-diagonal cell with the largest share of each true class."""
    rows = []
    totals = matrix.sum(axis=1)
    for i, label in enumerate(classes):
        if totals[i] == 0:
            continue
        scores = matrix[i].astype(float).copy()
        scores[i] = -1.0
        j = int(np.argmax(scores))
        if scores[j] <= 0:
            continue
        rows.append(dict(true_label=label, predicted_label=classes[j],
                         count=int(matrix[i, j]),
                         rate=float(matrix[i, j] / totals[i]),
                         n_true=int(totals[i])))
    return pd.DataFrame(rows).sort_values(['rate', 'count'],
                                          ascending=False).reset_index(drop=True)


def pair_checks(matrix, classes, pairs=PAIRS):
    """Both directions of prespecified label pairs; absent labels are skipped."""
    labels = list(classes)
    rows = []
    for first, second in pairs:
        if first not in labels or second not in labels:
            continue
        for true_label, predicted_label in ((first, second), (second, first)):
            i, j = labels.index(true_label), labels.index(predicted_label)
            total = int(matrix[i].sum())
            rows.append(dict(true_label=true_label, predicted_label=predicted_label,
                             count=int(matrix[i, j]),
                             rate=float(matrix[i, j] / total) if total else np.nan,
                             n_true=total))
    return pd.DataFrame(rows)


def evaluate(task, model_name, seed, train_mask, test_mask, X, y):
    model = make_model(model_name, task, seed)
    start = time.time()
    model.fit(X[train_mask], y[train_mask])
    seconds = time.time() - start
    probability = model.predict_proba(X[test_mask])
    prediction = probability.argmax(axis=1)
    metrics = score(model, X[test_mask], y[test_mask], task)
    labels = list(range(task.n_classes))
    matrix = confusion_matrix(y[test_mask], prediction, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y[test_mask], prediction, labels=labels, zero_division=0)
    per_class = pd.DataFrame(dict(label=task.classes, precision=precision,
                                  recall=recall, f1=f1, support=support))
    row = dict(task=task.name, variant=VARIANTS[0], model=model_name, seed=seed,
               n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
               n_classes=task.n_classes, fit_seconds=round(seconds, 2),
               per_class_recall=metrics.pop('per_class_recall'), **metrics)
    if BENIGN in task.classes:
        index = task.classes.index(BENIGN)
        true = y[test_mask]
        row['benign_false_negative_rate'] = float(
            (prediction[true == index] != index).mean()) if np.any(true == index) else np.nan
        row['benign_false_positive_rate'] = float(
            (prediction[true != index] == index).mean()) if np.any(true != index) else np.nan
    return row, matrix, per_class


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    frame, features = load_base_frame()
    rows, matrices, per_class_rows = [], {}, []
    for task_name in args.tasks:
        task = build_task(frame, features, task_name)
        for seed in args.seeds:
            train, test = load_locked(args.split_version, task_name, seed)
            for variant in args.variants:
                reduced, X, y, train_mask, test_mask = variant_arrays(
                    task, train, test, variant)
                for model_name in args.models:
                    row, matrix, per_class = evaluate(
                        reduced, model_name, seed, train_mask, test_mask, X, y)
                    row['variant'] = variant
                    rows.append(row)
                    key = (task_name, variant, model_name)
                    matrices.setdefault(key, []).append(matrix)
                    per_class.insert(0, 'seed', seed)
                    per_class.insert(0, 'model', model_name)
                    per_class.insert(0, 'variant', variant)
                    per_class.insert(0, 'task', task_name)
                    per_class_rows.append(per_class)
                    shown = ' '.join(f'{m}={row[m]:.4f}' for m in
                                     ('macro_f1', 'accuracy', 'log_loss')
                                     if m in row)
                    print(f'[error] {task_name}/{variant}/{model_name}/seed {seed}: '
                          f'{shown}', flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / 'runs.csv', index=False)
    per_class = pd.concat(per_class_rows, ignore_index=True)
    per_class.to_csv(OUT / 'per_class.csv', index=False)

    tops, pairs = [], []
    for (task_name, variant, model_name), parts in matrices.items():
        task = build_task(frame, features, task_name)
        classes = ([c for c in task.classes if c != BENIGN]
                   if variant == 'malware_only' else task.classes)
        total = np.sum(parts, axis=0)
        counts, normalized = confusion_frames(total, classes)
        stem = f'confusion_{{}}_{task_name}_{variant}_{model_name}'
        counts.to_csv(OUT / (stem.format('counts') + '.csv'))
        normalized.to_csv(OUT / (stem.format('normalized') + '.csv'))
        block = top_confusions(total, classes)
        block.insert(0, 'model', model_name)
        block.insert(0, 'variant', variant)
        block.insert(0, 'task', task_name)
        tops.append(block)
        checked = pair_checks(total, classes)
        if len(checked):
            checked.insert(0, 'model', model_name)
            checked.insert(0, 'variant', variant)
            checked.insert(0, 'task', task_name)
            pairs.append(checked)
        figure_matrix(normalized, f'{task_name} / {variant} / {model_name}',
                      OUT / 'figures' / f'confusion_{task_name}_{variant}_{model_name}.png')
    top = pd.concat(tops, ignore_index=True)
    top.to_csv(OUT / 'top_confusions.csv', index=False)
    pair_table = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    pair_table.to_csv(OUT / 'pair_checks.csv', index=False)

    present = [m for m in ('macro_f1', 'accuracy', 'balanced_accuracy', 'log_loss',
                           'roc_auc', 'pr_auc',
                           'benign_false_negative_rate', 'benign_false_positive_rate')
               if runs[m].notna().any()]
    summary = (runs.groupby(['task', 'variant', 'model'])[present]
                   .agg(['mean', 'std']))
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    write_report(runs, summary, top, args)
    write_manifest(args, runs)
    print(summary.to_string(index=False), flush=True)
    print(f'Error-analysis artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def figure_matrix(normalized, title, path):
    size = max(6.0, 0.55 * len(normalized))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(normalized.to_numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(normalized.columns)))
    ax.set_yticks(range(len(normalized.index)))
    ax.set_xticklabels(normalized.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(normalized.index, fontsize=7)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(runs, summary, top, args):
    lines = ['# Báo cáo lỗi phân loại đa lớp (source-aware)', '',
             f'**Phạm vi:** task `{", ".join(args.tasks)}`, model `{", ".join(args.models)}`, '
             f'seed `{", ".join(map(str, args.seeds))}`, split `{args.split_version}`. '
             'Không tuning; mọi mô hình dùng siêu tham số cố định.', '',
             '> Hai câu hỏi khác nhau: (a) **nhận dạng họ/type trong tập đóng** — mô hình '
             'đã thấy mọi lớp khi huấn luyện; (b) **phát hiện malware với họ chưa thấy** '
             '(LOFO nhị phân, `run_generalization.py`). Báo cáo này chỉ trả lời (a). '
             'Không được suy ra (b) từ các bảng dưới đây.', '',
             '## 1. Chỉ số tổng hợp', '']
    display = summary.copy()
    for column in display.columns:
        if display[column].dtype.kind == 'f':
            display[column] = display[column].round(4)
    lines.append(table_text(display) + '\n')
    lines.append('`with_benign` gồm cả lớp Benign; `malware_only` dùng cùng split nhưng chỉ '
                 'giữ các dòng malware, nên loại bỏ quyết định dễ Benign-vs-malware và phơi '
                 'ra nhầm lẫn giữa họ/type.\n')
    benign = runs.dropna(subset=['benign_false_negative_rate'])
    if len(benign):
        lines.extend(['## 2. Lỗi Benign', ''])
        columns = ['task', 'model', 'seed', 'benign_false_negative_rate',
                   'benign_false_positive_rate']
        lines.append(table_text(benign[columns].round(4)) + '\n')
        lines.append('`benign_false_negative_rate` = tỉ lệ dòng Benign bị dự đoán là '
                     'malware (bỏ sót Benign); `benign_false_positive_rate` = tỉ lệ dòng '
                     'malware bị dự đoán là Benign. Hai lỗi này không xuất hiện trong '
                     '`malware_only` vì lớp Benign đã bị loại.\n')
    lines.extend(['## 3. Cặp nhầm lẫn mạnh nhất', ''])
    lines.append(table_text(top.head(20).round(4)) + '\n')
    lines.append('Cột `rate` = phần trăm của lớp thật bị dự đoán sang lớp kia (cộng dồn '
                 'qua các seed). Đây là mô tả lỗi, không phải bằng chứng nhân quả.\n')
    lines.extend(['## 4. Giới hạn', ''])
    lines.append('- Một dataset CIC-MalMem-2022; split source-aware chỉ ngăn specimen '
                 'leakage, không phải bằng chứng ngoài phân phối.')
    lines.append('- Các seed dùng chung dữ liệu; SD qua seed là biến động chạy lại, không '
                 'phải khoảng tin cậy độc lập.')
    lines.append('- Kết luận chỉ trong tập đóng; không dùng để tuyên bố khả năng nhận dạng '
                 'họ chưa thấy.')
    (ROOT / 'report').mkdir(exist_ok=True)
    suffix = args.out_name.removeprefix('error').lstrip('_')
    name = f'error_analysis_{suffix}.md' if suffix else 'error_analysis.md'
    (ROOT / 'report' / name).write_text('\n'.join(lines))
    print(f'Generated report/{name}', flush=True)


def write_manifest(args, runs):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/run_error_analysis.py', created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        tasks=args.tasks, variants=args.variants, models=args.models,
        seeds=args.seeds, split_version=args.split_version,
        n_runs=len(runs), no_tuning=True,
        scope='closed-set multiclass error analysis; not external validation',
    )
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default='family16,type4')
    parser.add_argument('--models', default='xgboost')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--split-version', default='source_aware_v1')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--out-name', default='error_v1')
    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    args.models = [m.strip() for m in args.models.split(',') if m.strip()]
    args.seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    args.variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        parser.error(f'unknown variants: {sorted(unknown)}')
    global OUT
    OUT = ROOT / 'results' / 'error_analysis' / args.out_name
    run(args)


if __name__ == '__main__':
    main()
