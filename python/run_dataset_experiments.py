"""Run binary malware experiments from explicit dataset manifests in isolated snapshots."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from common import CONFIG, ROOT
from preprocess import prepare


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def normalize_dataset(path, spec):
    """Select only declared features; fail on unknown labels or malformed numeric data."""
    name = spec.get('dataset_id', '')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name):
        raise ValueError('dataset_id must use lowercase letters, numbers, _ or -')
    for field in ('source', 'collection_context'):
        if not isinstance(spec.get(field), str) or not spec[field].strip():
            raise ValueError(f'Manifest requires {field}')
    features = spec.get('feature_columns', [])
    if (not isinstance(features, list) or len(features) < 2
            or not all(isinstance(f, str) and f.strip() for f in features)
            or len(set(features)) != len(features)):
        raise ValueError('Declare at least two distinct numeric feature_columns')
    label = spec['label_column']
    category = spec.get('category_column')
    reserved = {label, category, 'Class', 'Category', 'label', '__label',
                'test_position', 'replicate'}
    if any(f in reserved or any(c in f for c in '|[]<\n\r') for f in features):
        raise ValueError('Feature list contains a label, reserved name or unsupported character')
    cases = spec.get('case_features', features[:7])
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 7
            or len(set(cases)) != len(cases) or not set(cases) <= set(features)):
        raise ValueError('case_features must contain 1 to 7 distinct declared features')
    mapping = spec.get('label_map', {})
    if not mapping or set(mapping.values()) != {'Benign', 'Malware'}:
        raise ValueError('label_map must map raw strings to both Benign and Malware')
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError('CSV has duplicate column names')
    frame = pd.read_csv(path, dtype={label: 'string'}, encoding='utf-8-sig')
    required = set(features) | {label} | ({category} if category else set())
    if not required <= set(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    # No implicit case-folding, multiclass collapsing or removal of unknown labels.
    labels = frame[label].str.strip().map(mapping)
    if labels.isna().any():
        raise ValueError('Unmapped/missing labels; declare every raw label in label_map')
    if set(labels) != {'Benign', 'Malware'}:
        raise ValueError('CSV must contain both Benign and Malware')
    result = frame[features].apply(pd.to_numeric, errors='raise')
    result = result.replace([np.inf, -np.inf], np.nan)
    result['Category'] = frame[category].fillna('Unknown') if category else 'Unknown'
    result['Class'] = labels
    return result, cases, sorted(set(frame.columns) - required)


def git_revision():
    result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def snapshot(work, frame, spec, config, provenance):
    work.mkdir(parents=True, exist_ok=False)
    for directory in ('python', 'r', 'data', 'tests'):
        (work / directory).mkdir()
    for pattern in ('python/*.py', 'r/*.R', 'tests/*.py', 'run_all.sh', 'requirements*.txt'):
        for source in ROOT.glob(pattern):
            shutil.copy2(source, work / source.relative_to(ROOT))
    (work / 'RLibrary').symlink_to(ROOT / 'RLibrary', target_is_directory=True)
    frame.to_csv(work / config['data'], index=False)
    for filename, value in [('config.json', config), ('dataset_manifest.json', spec),
                            ('provenance.json', provenance)]:
        (work / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def summarize(base, works):
    rows = []
    for work in works:
        receipt = json.loads((work / 'results/intermediate/verification.json').read_text())
        if receipt['status'] != 'passed':
            raise ValueError(f'Unverified run: {work}')
        config = json.loads((work / 'config.json').read_text())
        row = pd.read_csv(work / 'results/tables/model_metrics.csv').iloc[0].to_dict()
        row.update(seed=config['seed'], coverage_min=receipt['coverage_min'],
                   coverage_max=receipt['coverage_max'], run=str(work.relative_to(base)))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(base / 'metrics_by_seed.csv', index=False)
    metrics[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'coverage_min']].agg(
        ['mean', 'std', 'min', 'max']).to_csv(base / 'metrics_summary.csv', index_label='statistic')
    print(metrics.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Source CSV (relative to current directory)')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrites')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--mode', choices=['full', 'smoke'], default='full')
    parser.add_argument('--prepare-only', action='store_true', help='Validate and snapshot without fitting models')
    parser.add_argument('--threads', type=int, default=CONFIG['threads'])
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**31-10000 for s in seeds):
        parser.error('Seeds must be distinct nonnegative integers below 2**31-10000')
    if args.threads < 1:
        parser.error('--threads must be positive')
    source, base = args.data.resolve(), args.output.resolve()
    if base.exists():
        parser.error(f'Output already exists; choose a new directory: {base}')
    spec = json.loads(args.manifest.read_text())
    source_hash = digest(source)
    if spec.get('expected_sha256') and source_hash != spec['expected_sha256']:
        raise ValueError('Source CSV SHA-256 differs from the manifest')
    frame, cases, ignored = normalize_dataset(source, spec)
    if digest(source) != source_hash:
        raise ValueError('Source CSV changed while being read')
    config = dict(CONFIG, data='data/input.csv', threads=args.threads,
                  case_features=cases, dataset_description=f"{spec['dataset_id']} — {spec['source']}")
    if args.mode == 'smoke':
        config.update(arf_train_samples=200, importance_test_samples=50,
                      conditional_repeats=2, permutation_repeats=2,
                      arf_num_trees=5, arf_min_node_size=10)
    # Validate all planned splits before launching expensive work or creating artifacts.
    for seed in seeds:
        split, _, _ = prepare(frame, seed=seed, test_size=config['test_size'])
        if min(np.bincount(split['y_train']), default=0) < 2:
            raise ValueError('Training split needs at least two observations per class')
    provenance = dict(dataset_id=spec['dataset_id'], source_file=source.name,
                      source_sha256=source_hash, source_rows=len(frame),
                      row_order='normalized CSV row i corresponds to source CSV data row i (zero-based)',
                      ignored_columns=ignored, git_commit=git_revision(), mode=args.mode,
                      created_utc=datetime.now(timezone.utc).isoformat(),
                      protocol='independent within-dataset stratified holdout; retrain for every seed',
                      note='Source fingerprints in run_manifest.json identify the exact snapshot, including local edits.')
    if not args.prepare_only and not shutil.which('Rscript'):
        raise RuntimeError('Install R and run Rscript r/install_dependencies.R first')
    base.mkdir(parents=True, exist_ok=False)
    works = []
    for seed in seeds:
        work = base / f'seed_{seed}'
        snapshot(work, frame, spec, dict(config, seed=seed), provenance)
        works.append(work)
    status = dict(status='prepared', mode=args.mode, seeds=seeds, dataset_id=spec['dataset_id'])
    status_path = base / 'status.json'
    status_path.write_text(json.dumps(status, indent=2))
    if args.prepare_only:
        print(f'Validated snapshots: {base}. No models fitted; choose a NEW output for a full run.')
        return
    start = time.monotonic()
    try:
        for work in works:
            print(f'Running {work.name}; log: {work / "pipeline.log"}', flush=True)
            status.update(status='running', current_seed=work.name)
            status_path.write_text(json.dumps(status, indent=2))
            env = dict(os.environ, PYTHON=sys.executable)
            with (work / 'pipeline.log').open('w') as log:
                subprocess.run(['bash', str(work / 'run_all.sh')], cwd=work, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        summarize(base, works)
        status.update(status='passed', elapsed_seconds=time.monotonic() - start)
    except BaseException:
        status.update(status='failed', elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))
    print(f'{args.mode} run verified: {base}', flush=True)


if __name__ == '__main__':
    main()
