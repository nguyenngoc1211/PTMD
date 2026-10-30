"""Check whether MemMal-D2024 shares rows with the local CIC-MalMem-2022 data.

The only admissible use of a second dataset as external validation is if its
rows are independent of the local benchmark. This script compares the public
MemMal-D2024 CSV against the local CIC-MalMem CSV on the 55 shared numeric
features plus ``Category`` and reports how many rows match exactly. A row is a
match only when every shared numeric value (after identical CSV parsing) is
equal at 12 decimals and the Category string is equal.

Output is a JSON audit under ``results/external_benchmarks/memmal_d2024_independence``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_keys(frame, features):
    keys = set()
    columns = ['Category'] + features
    for row in frame[columns].itertuples(index=False, name=None):
        values = np.round(np.asarray(row[1:], dtype=np.float64), 12).tolist()
        keys.add((row[0],) + tuple(values))
    return keys


def compare(memmal, local):
    features = [c for c in local.columns if c not in ('Category', 'Class')]
    missing = [f for f in features if f not in memmal.columns]
    if missing:
        raise SystemExit(f'MemMal CSV is missing shared features: {missing}')
    local_malware = local[local['Class'] == 'Malware']
    local_keys = row_keys(local_malware, features)
    memmal_keys = row_keys(memmal, features)
    matched = len(memmal_keys & local_keys)
    return dict(
        memmal_rows=int(len(memmal)),
        memmal_unique_rows=int(len(memmal_keys)),
        local_rows=int(len(local)),
        local_malware_rows=int(len(local_malware)),
        local_malware_unique_rows=int(len(local_keys)),
        shared_features=int(len(features)),
        matched_unique_rows=int(matched),
        unmatched_unique_rows=int(len(memmal_keys) - matched),
        share_of_memmal_matched=round(matched / len(memmal_keys), 6),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memmal', default='data/external/memmal_d2024/'
                                            'All_Malware_Samples-Combined2006-2021.csv')
    parser.add_argument('--local', default=CONFIG['data'])
    parser.add_argument('--out-name', default='memmal_d2024_independence')
    args = parser.parse_args()

    memmal_path = ROOT / args.memmal
    local_path = ROOT / args.local
    started = time.time()
    memmal = pd.read_csv(memmal_path)
    local = pd.read_csv(local_path)
    result = compare(memmal, local)
    result.update(
        script='python/verify_memmal_independence.py',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        elapsed_seconds=round(time.time() - started, 2),
        memmal_path=str(args.memmal),
        memmal_sha256=sha256(memmal_path),
        local_path=str(args.local),
        local_sha256=sha256(local_path),
        scope='Malware subset only; exact equality of pandas-parsed numeric values '
              'plus Category, not raw dump comparison.',
        verdict=('NOT INDEPENDENT: every unique MemMal-D2024 row occurs in CIC-MalMem'
                 if result['unmatched_unique_rows'] == 0 else
                 'PARTIALLY INDEPENDENT: some MemMal rows do not occur in CIC-MalMem'),
    )
    out = ROOT / 'results' / 'external_benchmarks' / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'overlap.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f'Audit written to {out.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
