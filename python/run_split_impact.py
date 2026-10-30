"""Measure what the source-aware split does to model performance.

The audit found that the random 80/20 split puts 99.8% of test specimens in
train as well (``TECHNICAL_AUDIT.md`` §3), so any metric measured on it is
inflated by near-duplicate snapshots. This script quantifies the inflation: the
same model, same seeds and same features, fitted and scored on the random split
and on the locked source-aware split.

It is deliberately the smallest possible experiment that isolates the split.
Hyper-parameters are fixed, there is no tuning, and nothing is selected on test.

    python/run_split_impact.py --seeds 42,43,44

Outputs (never overwriting the verified pipeline):
    results/splits/split_impact_xgboost.csv   per (task, strategy, seed)
    results/splits/split_impact_summary.csv   aggregated per (task, strategy)
    report/split_impact.md
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import ROOT
from generate_report import table as table_text
from group_protocol import TASKS, build_task, load_base_frame, make_model, score

OUT = ROOT / 'results' / 'splits'
SCALAR = ['log_loss', 'accuracy', 'f1', 'macro_f1', 'roc_auc', 'pr_auc',
          'balanced_accuracy']


def run(seeds, tasks, model_name):
    frame, features = load_base_frame()
    rows = []
    for task_name in tasks:
        task = build_task(frame, features, task_name)
        X = task.frame[task.features].to_numpy(dtype=np.float64)
        for strategy in ('random', 'source_aware'):
            for seed in seeds:
                train, test = task.split(seed, strategy=strategy)
                model = make_model(model_name, task, seed)
                start = time.time()
                model.fit(X[train], task.y[train])
                fit_seconds = time.time() - start
                metrics = score(model, X[test], task.y[test], task)
                rows.append(dict(task=task_name, strategy=strategy, seed=seed,
                                 model=model_name, n_train=len(train),
                                 n_test=len(test), fit_seconds=round(fit_seconds, 2),
                                 per_class_recall=metrics.pop('per_class_recall'),
                                 **metrics))
                shown = ' '.join(f'{k}={metrics[k]:.5f}' for k in SCALAR
                                 if k in metrics)
                print(f'{task_name:9s} {strategy:12s} seed{seed}: {shown}', flush=True)
    return pd.DataFrame(rows)


def summarise(runs):
    present = [m for m in SCALAR if runs[m].notna().any()]
    wide = (runs.groupby(['task', 'strategy', 'model'])[present]
                .agg(['mean', 'std']))
    wide.columns = [f'{metric}_{stat}' for metric, stat in wide.columns]
    summary = wide.reset_index()
    # Inflation = how much better the leaky split looks than the clean one.
    gaps = []
    for (task, model_name), block in summary.groupby(['task', 'model']):
        clean = block[block.strategy == 'source_aware']
        leaky = block[block.strategy == 'random']
        if len(clean) == 0 or len(leaky) == 0:
            continue
        row = dict(task=task, model=model_name)
        for metric in present:
            row[f'{metric}_gap_random_minus_clean'] = (
                float(leaky[f'{metric}_mean'].iloc[0])
                - float(clean[f'{metric}_mean'].iloc[0]))
        gaps.append(row)
    return summary, pd.DataFrame(gaps)


def write_report(runs, summary, gaps, seeds):
    lines = ['# Tác động của source-aware split lên hiệu năng\n']
    lines.append('Sinh bởi `python/run_split_impact.py`. Cùng mô hình, cùng seed, '
                 'cùng feature; chỉ **cách chia** thay đổi.\n')
    lines.append('## Thiết kế\n')
    lines.append(f'- Seed: {", ".join(map(str, seeds))}; model: '
                 f'{", ".join(runs.model.unique())}.')
    lines.append('- `random`: chia dòng stratified như pipeline đã kiểm chứng — '
                 '**99,8% specimen test cũng có mặt ở train**.')
    lines.append('- `source_aware`: mọi snapshot của một specimen nằm về một phía — '
                 '**0% giao nhau** (xác minh `split_manager --verify`).')
    lines.append('- Không tuning, không chọn feature trên test.\n')
    lines.append('## Khoảng cách giữa hai cách chia\n')
    display = gaps.copy()
    for metric in ('accuracy', 'f1', 'macro_f1', 'roc_auc', 'pr_auc', 'log_loss'):
        column = f'{metric}_gap_random_minus_clean'
        if column in display.columns:
            display[column] = display[column].round(5)
    display.columns = [c.replace('_gap_random_minus_clean', ' (random − clean)')
                        .replace('_', ' ') for c in display.columns]
    lines.append(table_text(display) + '\n')
    lines.append('Giá trị dương ở accuracy/F1/AUC = split ngẫu nhiên **thổi phồng**; '
                 'giá trị dương ở log-loss = split ngẫu nhiên **làm đẹp** sai số.\n')
    lines.append('## Bảng đầy đủ\n')
    columns = ['task', 'strategy', 'seed', 'log_loss', 'accuracy', 'f1',
               'macro_f1', 'roc_auc', 'pr_auc']
    table = runs[[c for c in columns if c in runs.columns]].copy()
    lines.append(table_text(table.round(5)) + '\n')
    lines.append('## Cách đọc\n')
    lines.append('- Nếu khoảng cách ≈0, kết luận cũ không phụ thuộc leakage.')
    lines.append('- Nếu khoảng cách lớn (>0,01 ở metric chính), **mọi con số của split '
                 'ngẫu nhiên phải được coi là cận trên lạc quan**, và báo cáo nộp phải '
                 'dùng số source-aware.')
    lines.append('- Binary F1 có thể vẫn cao sau khi làm sạch: LOFO (giữ trọn một họ, '
                 'do đó cũng không có specimen giao nhau) đã cho F1 = 0,9988. Nghĩa là '
                 'phát hiện malware-vs-benign vốn đã dễ; **độ khó thật nằm ở gán nhãn họ**.')
    lines.append('- Đây là thí nghiệm trên **một model, không tuning**. Không được dùng '
                 'để tuyên bố về model khác.\n')
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'report' / 'split_impact.md').write_text('\n'.join(lines))
    print('Generated report/split_impact.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--model', default='xgboost')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    runs = run(seeds, tasks, args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / 'split_impact_xgboost.csv', index=False)
    summary, gaps = summarise(runs)
    summary.to_csv(OUT / 'split_impact_summary.csv', index=False)
    gaps.to_csv(OUT / 'split_impact_gaps.csv', index=False)
    write_report(runs, summary, gaps, seeds)
    print()
    print(gaps.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
