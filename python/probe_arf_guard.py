#!/usr/bin/env python
"""Đếm số cột kích hoạt guard ARF theo (seed, arf_train).

Guard ở run_group_cfi.group_importance chặn một nhóm khi stratified subset làm
cột đó thành hằng số trong subset nhưng vẫn biến thiên trên toàn bộ train: khi đó
ARF không còn phân phối điều kiện để draw. Hàm này tái hiện đúng phép thử của
guard mà không cần chạy R, nên chạy được trên máy thiếu RAM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from group_protocol import load_tasks  # noqa: E402
from run_group_cfi import stratified_subset  # noqa: E402
from split_manager import load_locked  # noqa: E402


def probe(task_name: str, split_version: str, seeds, sizes):
    task = load_tasks([task_name])[task_name]
    rows = []
    for seed in seeds:
        train, _ = load_locked(split_version, task_name, seed)
        full = task.frame.iloc[train][task.features]
        varies_full = full.min() != full.max()
        for size in sizes:
            subset = full.iloc[stratified_subset(task.y[train], size, seed)]
            hit = (subset.min() == subset.max()) & varies_full
            rows.append(dict(task=task_name, seed=seed, arf_train=size,
                             split_version=split_version,
                             n_guard_columns=int(hit.sum()),
                             guard_columns=';'.join(sorted(hit[hit].index)) or '-'))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='family16')
    parser.add_argument('--split-version', default='source_aware_v1')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--sizes', default='3000,5000,8000,10000')
    parser.add_argument('--out', default=str(ROOT / 'results/group_experiments/'
                                             'arf_constant_guard_probe.csv'))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(',')]
    sizes = [int(value) for value in args.sizes.split(',')]
    frame = probe(args.task, args.split_version, seeds, sizes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
