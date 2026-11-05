"""Build every number and figure the workshop paper quotes, straight from artifacts.

The workshop paper covers the data audit and the split-locking procedure only, so this
script deliberately reads the dataset, the locked-split manifest and the leakage /
split-impact CSVs, and never touches results/group_experiments/. Anything it prints
into the paper is first written to results/conference/ so the claim has an artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))
from group_protocol import SPECIMEN_PATTERN, load_base_frame, parse_category  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'Obfuscated-MalMem2022.csv'
SPLITS = ROOT / 'results' / 'splits'
OUT = ROOT / 'results' / 'conference'
FIGS = ROOT / 'latex_conference' / 'figures'


def unit_profile() -> pd.DataFrame:
    """Row counts per source unit, before and after exact-vector deduplication."""
    frame = pd.read_csv(DATA, usecols=['Class', 'Category'])
    parsed = frame['Category'].map(parse_category)
    info = pd.DataFrame({
        'type': [p[0] for p in parsed],
        'family': [p[1] for p in parsed],
        'specimen': [p[2] for p in parsed],
        'run': [p[3] for p in parsed],
        'label': frame['Class'].str.strip().str.lower(),
    })
    vectors = pd.read_csv(DATA).drop(columns=['Class', 'Category'])
    first_of_vector = ~vectors.duplicated()
    info['kept_after_dedup'] = first_of_vector

    benign = info['type'].eq('Benign')
    unparsable = ~benign & info['specimen'].isna()
    rows = [
        ('rows_raw', len(info)),
        ('rows_benign', int(benign.sum())),
        ('rows_malware', int((~benign).sum())),
        ('rows_unique_vectors', int(first_of_vector.sum())),
        ('rows_duplicate_vectors', int((~first_of_vector).sum())),
        ('specimens_raw', info.loc[info['specimen'].notna(), 'specimen'].nunique()),
        ('specimens_unparsable_id_rows', int(unparsable.sum())),
        ('rows_per_specimen_min', int(info['specimen'].value_counts().min())),
        ('rows_per_specimen_median', float(info['specimen'].value_counts().median())),
        ('rows_per_specimen_max', int(info['specimen'].value_counts().max())),
        ('rows_per_specimen_after_dedup_min',
         int(info.loc[info['specimen'].notna() & info['kept_after_dedup'], 'specimen']
             .value_counts().min())),
        ('rows_per_specimen_after_dedup_median',
         float(info.loc[info['specimen'].notna() & info['kept_after_dedup'], 'specimen']
               .value_counts().median())),
        ('rows_per_specimen_after_dedup_max',
         int(info.loc[info['specimen'].notna() & info['kept_after_dedup'], 'specimen']
             .value_counts().max())),
        ('families_including_benign', int(info['family'].dropna().nunique())),
        ('types_including_benign', int(info['type'].dropna().nunique())),
    ]
    # A family label that flips between snapshots of one specimen cannot be a
    # multiclass target; count those vectors, since the task definitions differ by it.
    spec = info[info['specimen'].notna()]
    per_spec_family = spec.groupby('specimen')['family'].nunique()
    rows.append(('specimens_with_conflicting_family', int((per_spec_family > 1).sum())))
    # Grouping happens on the deduplicated frame, so the unit count that the split
    # manifest reports is this one, not specimens_raw plus rows_benign.
    dedup = info[info['kept_after_dedup']]
    mal = dedup[dedup['specimen'].notna()]
    rows.append(('units_after_dedup', int(mal['specimen'].nunique() +
                                          dedup['label'].eq('benign').sum())))
    rows.append(('specimen_units_after_dedup', int(mal['specimen'].nunique())))
    rows.append(('singleton_benign_units_after_dedup', int(dedup['label'].eq('benign').sum())))
    # The specimen id repeats the same run index twice on some ids, so those "specimens"
    # are really two samples of one source. Grouping by id still merges them, which is
    # conservative for splitting, but the id is not a unique source key.
    per_spec_rows = spec.groupby('specimen').size()
    per_spec_runs = spec.groupby('specimen')['run'].nunique()
    rows.append(('specimens_with_duplicate_run_index',
                 int((per_spec_rows > per_spec_runs).sum())))
    dist = per_spec_rows.value_counts().sort_index().rename_axis('rows_per_specimen')
    dist.rename('n_specimens').to_csv(OUT / 'rows_per_specimen.csv',
                                      header=True)
    return pd.DataFrame(rows, columns=['quantity', 'value'])


def label_conflict_counts() -> pd.DataFrame:
    """Which vectors the family-conflict rule flags, and whether binary survives it.

    Uses the pipeline's own rule rather than re-deriving it: ``load_base_frame``
    replaces +-inf with NaN before hashing, and that normalization is what moves the
    count from 9 to 19. Two code paths computing "the" conflict count differently is
    exactly the failure this paper is supposed to be immune to.
    """
    from group_protocol import load_base_frame
    frame, features = load_base_frame(DATA)
    raw = pd.read_csv(DATA)
    label = raw['Class'].str.strip().str.lower()
    family = raw['Category'].map(lambda p: p[1])
    key = pd.util.hash_pandas_object(
        raw[features].replace([np.inf, -np.inf], np.nan), index=False)
    grouped = pd.DataFrame({'key': key, 'family': family, 'label': label})
    n_lab = grouped.groupby('key')['label'].nunique()
    n_fam = grouped.groupby('key')['family'].nunique()
    conflicting = set(n_fam[n_fam > 1].index)
    on_conflicting = n_lab.loc[n_lab.index.isin(conflicting)]
    unique_vectors = int(len(frame))
    return pd.DataFrame([
        ('vectors_with_conflicting_family', frame.attrs['conflicting_vectors']),
        ('vectors_flagged_by_pipeline', int(frame['__conflict'].sum())),
        ('raw_rows_on_those_vectors', int(grouped['key'].isin(conflicting).sum())),
        ('vectors_with_conflicting_binary_label', int((n_lab > 1).sum())),
        ('family_conflicting_vectors_binary_all_agree',
         bool((on_conflicting == 1).all())),
        ('rows_unique_vectors', unique_vectors),
        ('rows_removed_by_dedup', int(len(raw) - unique_vectors)),
        ('rows_family16_after_dedup_and_conflict',
         int(unique_vectors - frame.attrs['conflicting_vectors'])),
        ('rows_binary_after_dedup', unique_vectors),
    ], columns=['quantity', 'value'])


def redundancy() -> pd.DataFrame:
    """Near-duplicate feature pairs on the deduplicated frame, split by plugin."""
    frame, features = _dedup_features()
    ranks = frame[features].rank()
    corr = ranks.corr().to_numpy()
    np.fill_diagonal(corr, 0.0)
    iu = np.triu_indices(len(features), k=1)
    out = pd.DataFrame({
        'feature_a': [features[i] for i in iu[0]],
        'feature_b': [features[j] for j in iu[1]],
        'spearman': corr[iu],
    })
    out['plugin_a'] = out.feature_a.str.split('.').str[0]
    out['plugin_b'] = out.feature_b.str.split('.').str[0]
    out['same_plugin'] = out.plugin_a == out.plugin_b
    out['abs_spearman'] = out.spearman.abs()
    out.to_csv(OUT / 'redundancy_pairs.csv', index=False)
    rows = []
    for cut in (0.98, 0.95, 0.90, 0.80):
        sel = out[out.abs_spearman >= cut]
        rows.append(dict(threshold=cut, pairs=int(len(sel)),
                         same_plugin=int(sel.same_plugin.sum()),
                         cross_plugin=int((~sel.same_plugin).sum())))
    summary = pd.DataFrame(rows)
    constants = [f for f in features if frame[f].nunique() == 1]
    summary.attrs['constants'] = constants
    summary.attrs['pairs_total'] = len(out)
    summary.attrs['n_features'] = len(features)
    out.sort_values('abs_spearman', ascending=False).head(12)[
        ['feature_a', 'feature_b', 'spearman', 'same_plugin']].to_csv(
        OUT / 'redundancy_top.csv', index=False)
    return summary


def _dedup_features():
    from group_protocol import load_base_frame
    frame, features = load_base_frame(DATA)
    return frame, features


def tidy_leakage() -> pd.DataFrame:
    leak = pd.read_csv(SPLITS / 'leakage_comparison.csv', float_precision='round_trip')
    leak['multi_row_units_pct'] = leak.pct_units_leaked
    return leak


def tidy_manifest() -> pd.DataFrame:
    manifest = json.loads((SPLITS / 'source_aware_v1' / 'split_manifest.json').read_text())
    rows = [{k: s[k] for k in ('task', 'seed', 'strategy', 'n_train', 'n_test',
                               'n_units_train', 'n_units_test', 'units_crossing')}
            for s in manifest['splits']]
    return pd.DataFrame(rows)


def tidy_impact() -> pd.DataFrame:
    imp = pd.read_csv(SPLITS / 'split_impact_summary.csv', float_precision='round_trip')
    wide = imp.melt(id_vars=['task', 'strategy', 'model'],
                    value_vars=['accuracy_mean', 'macro_f1_mean', 'log_loss_mean',
                                'roc_auc_mean', 'balanced_accuracy_mean'],
                    var_name='metric', value_name='value')
    wide['metric'] = wide.metric.str.replace('_mean', '', regex=False)
    out = wide.pivot_table(index=['task', 'model', 'metric'], columns='strategy',
                           values='value').reset_index()
    out.columns = [c if isinstance(c, str) else c for c in out.columns]
    out = out.rename(columns={'random': 'value_random', 'source_aware': 'value_locked'})
    out['gap_random_minus_locked'] = out.value_random - out.value_locked
    return out


def fig_leakage(leak: pd.DataFrame, path: Path):
    tasks = ['binary', 'type4', 'family16']
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    width, step = 0.12, 0.16
    for i, task in enumerate(tasks):
        sub = leak[(leak.task == task) & (leak.strategy == 'random')].sort_values('seed')
        for j, (col, hatch) in enumerate([('pct_units_leaked', ''), ('pct_rows_leaked', '///')]):
            x = np.arange(len(tasks)) + (i - 1) * step + j * width
            ax.bar(x, sub[col], width, hatch=hatch, edgecolor='black', linewidth=0.5,
                   color='white' if j else '0.85',
                   label=f'{task}: ' + ('đơn vị nguồn' if j == 0 else 'dòng test')
                   if i == 1 else None)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='0.85', edgecolor='black')]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black',
                              hatch='///')]
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% của test')
    ax.set_xlabel('task')
    ax.legend(handles, ['đơn vị nguồn (specimen) rò rỉ', 'dòng test đến từ đơn vị rò rỉ'],
              fontsize=7, loc='lower right', ncol=1, framealpha=1)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_impact(impact: pd.DataFrame, path: Path):
    keep = impact[impact.metric.isin(['macro_f1', 'accuracy', 'log_loss'])]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    labels = [f'{r.task}\n{r.metric}' for r in keep.itertuples()]
    x = np.arange(len(keep))
    ax.bar(x - 0.2, keep.value_random, 0.4, color='0.8', edgecolor='black', linewidth=0.5,
           label='split ngẫu nhiên')
    ax.bar(x + 0.2, keep.value_locked, 0.4, color='0.35', edgecolor='black', linewidth=0.5,
           label='split khóa theo nguồn')
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel('XGBoost, trung bình 3 seed')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def task_construction() -> pd.DataFrame:
    """Rows, classes and unit strata per task, as the locked pipeline builds them."""
    from split_manager import MIXED, unit_labels
    from group_protocol import TASKS, build_task
    frame, features = load_base_frame()
    rows, mixed_ids = [], set()
    for name in TASKS:
        task = build_task(frame, features, name)
        units = task.frame['__unit'].astype(str).to_numpy()
        ids, strata, sizes = unit_labels(units, task.y)
        mixed = strata == MIXED
        if mixed.any() and not mixed_ids:
            mixed_ids = set(ids[mixed])
        rows.append(dict(task=name, rows=len(task.frame), classes=task.n_classes,
                         units=len(ids), mixed_units=int(mixed.sum()),
                         mixed_rows=int(sizes[mixed].sum()),
                         singleton_units=int(pd.Series(ids).str.startswith('__row_').sum())))
    table = pd.DataFrame(rows)
    # A mixed unit must be an id collision (>10 snapshots, exactly two families and
    # two types under one specimen hash), not a label that drifted across snapshots.
    info = pd.read_csv(DATA, usecols=['Class', 'Category'])
    parsed = info['Category'].map(parse_category)
    spec = pd.DataFrame({'specimen': [p[2] for p in parsed],
                         'family': [p[1] for p in parsed],
                         'type': [p[0] for p in parsed]}).dropna(subset=['specimen'])
    per_spec = spec.groupby('specimen').agg(rows=('family', 'size'),
                                            families=('family', 'nunique'),
                                            types=('type', 'nunique'))
    wide = per_spec[per_spec['rows'] > 10]
    collision = set(wide.index[(wide['families'] == 2) & (wide['types'] == 2)])
    table.attrs['ids_over_10_rows'] = int(len(wide))
    table.attrs['collision_ids_two_families_two_types'] = int(len(collision))
    table.attrs['mixed_units_covered_by_collisions'] = int(len(mixed_ids & collision))
    table.attrs['mixed_units_total'] = int(len(mixed_ids))
    return table


def fig_unit_size(path: Path):
    """How many snapshots each parsed source id carries."""
    dist = pd.read_csv(OUT / 'rows_per_specimen.csv')
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.bar(dist.rows_per_specimen, dist.n_specimens, width=0.8,
           color='0.8', edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('số dòng snapshot mang cùng một định danh specimen')
    ax.set_ylabel('số specimen (trục log)')
    ax.set_xticks(range(1, 21))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    unit_profile().to_csv(OUT / 'unit_profile.csv', index=False)
    label_conflict_counts().to_csv(OUT / 'label_conflicts.csv', index=False)
    tasks = task_construction()
    tasks.to_csv(OUT / 'task_construction.csv', index=False)
    red = redundancy()
    red.to_csv(OUT / 'redundancy_summary.csv', index=False)
    tidy_leakage().to_csv(OUT / 'leakage_tidy.csv', index=False)
    tidy_manifest().to_csv(OUT / 'split_manifest_tidy.csv', index=False)
    impact = tidy_impact()
    impact.to_csv(OUT / 'impact_tidy.csv', index=False)
    overlap = pd.DataFrame([json.loads((ROOT / 'results' / 'external_benchmarks' /
                                        'memmal_d2024_independence' / 'overlap.json')
                                       .read_text())])
    overlap.to_csv(OUT / 'cross_dataset_tidy.csv', index=False)
    fig_leakage(tidy_leakage(), FIGS / 'fig_leakage.pdf')
    fig_impact(impact, FIGS / 'fig_impact.pdf')
    fig_unit_size(FIGS / 'fig_unit_size.pdf')
    print(f'wrote {OUT} and {FIGS}')
    print(unit_profile().to_string(index=False))
    print(label_conflict_counts().to_string(index=False))
    print(red.to_string(index=False))
    print(tasks.to_string(index=False))
    print('mixed units covered by id collisions:',
          tasks.attrs['mixed_units_covered_by_collisions'], '/',
          tasks.attrs['mixed_units_total'], '| ids >10 rows:',
          tasks.attrs['ids_over_10_rows'])
    print('constants:', red.attrs['constants'], 'features:', red.attrs['n_features'],
          'pairs:', red.attrs['pairs_total'])


if __name__ == '__main__':
    main()
