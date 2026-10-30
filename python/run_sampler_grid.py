"""Pre-registered grid evaluation of the ARF/FORDE conditional sampler.

What this measures
------------------
For each setting in a small grid (``finite_bounds`` x epsilon x ARF training
size x tree count x min-node size) this script draws a fixed list of
representative features from ``p(x_j | x_-j)`` on a fixed evidence sample,
using ``r/sampler_grid.R``, and scores every draw against the domain table
from ``feature_domains.py``:

* ``invalid_rate``      values outside their semantic domain (negative counts,
                        ratios above 1, non-integer counts, non-finite);
* ``out_of_range``      values outside the training support, compared with the
                        same rate for real test rows;
* ``ks_synth``/``ks_null``  marginal fidelity, against the half-sample noise
                        floor of real data;
* ``constraint_excess`` violation of discovered cross-feature inequalities
                        minus the same violation rate on real test rows;
* ``cond_rank_corr``    conditional fidelity: Spearman between the per-row mean
                        of the draws and the real value the draw is conditioned
                        on (reported, not gated: a feature can be legitimately
                        independent of the rest);
* ``discriminator_accuracy``  ARF's own real-vs-synthetic classifier accuracy
                        (reported, not gated: its null depends on sample size).

Gate, fixed before any grid result was inspected
------------------------------------------------
A setting passes only if, for every feature:
    support_rate >= 0.98            (evidence inside the ARF-train range)
    leaf_support_rate >= 0.98       (FORDE found a matching leaf)
    coverage_conditionable >= 0.99  (draws for rows the model can condition on)
    invalid_rate == 0
    out_of_range_synth <= out_of_range_real + 0.02
    enforced constraint excess <= 0.02
    ks_synth <= max(0.10, 2 * ks_null)
``coverage`` overall is reported next to ``coverage_conditionable``. A row
whose evidence lies outside the range of the ARF training subset cannot be
conditioned on by any density estimated from those rows; that is a property of
the evidence sample, recorded as ``support_rate``, not a sampler failure.
Separately, a row for which no FORDE leaf matches (returned as all-NaN by
``forge(nomatch='na')``) is a model-support limit recorded as
``leaf_support_rate``. Coverage is measured only on rows that pass both.
Among passing settings, the recommendation is the one with the lowest
``max_ks_synth``, ties broken by lower fit time.

Constraint enforcement and boundary snapping
--------------------------------------------
Domains and constraints are derived from the training rows of the locked split
only, never from test rows. ARF/FORDE does not learn arithmetic inequalities,
so draws are proposed in a pool of ``repeats * pool`` per row and the first
``repeats`` proposals satisfying every *enforced* constraint (discovery
hold rate >= ``CONSTRAINT_HOLD``) are kept. Constraints below that hold rate
are reported as ``max_relaxed_constraint_excess`` but not gated: they are
near-deterministic relations, not hard ones. Rejection conditions the draw on
the near-deterministic region; it does not clip values.

``finite_bounds=local`` widens each leaf range by ``epsilon/2 * range`` on
both sides, so a ratio whose leaf minimum is 0 can produce values like
-2.5e-07. Values in ``[-tol, 0)`` with ``tol = epsilon * train_range / 2`` are
snapped to 0 and counted in ``n_snapped`` with their maximum magnitude
(``snap_max``) so the distributional impact is reported, not hidden.

Outputs (``results/sampler_v2/``): ``grid_features.csv``, ``grid_summary.csv``,
``draws/``, ``grid_manifest.json``, plus train-only ``input/domains_*.csv`` and
``input/constraints_*.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.model_selection import train_test_split

from common import CONFIG, ROOT
from feature_domains import classify, discover_constraints, validate
from group_protocol import load_tasks
from split_manager import load_locked

OUT = ROOT / 'results' / 'sampler_v2'
DRAWS = OUT / 'draws'
INPUT = OUT / 'input'

MIN_COVERAGE = 0.99
MIN_SUPPORT = 0.98
MAX_OUT_OF_RANGE_EXCESS = 0.02
MAX_CONSTRAINT_EXCESS = 0.02
CONSTRAINT_HOLD = 0.999
KS_FLOOR = 0.10

# One representative per domain type per group, plus the features the audit
# found were worst under the old sampler (psxview.*_false_avg).
DEFAULT_FEATURES = [
    'pslist.nproc',                          # count
    'pslist.avg_threads',                    # continuous
    'dlllist.ndlls',                         # count
    'handles.nhandles',                      # count, max ~1e6
    'ldrmodules.not_in_load_avg',            # bounded_ratio
    'malfind.commitCharge',                  # count, heavy tail
    'psxview.not_in_pslist',                 # count, small
    'psxview.not_in_pspcid_list_false_avg',  # bounded_ratio, old negatives
    'psxview.not_in_ethread_pool_false_avg', # bounded_ratio, old negatives
    'svcscan.nservices',                     # count
    'callbacks.ncallbacks',                  # count
    'modules.nmodules',                      # count, 3 distinct values
]


def stratified_subset(y, size, seed):
    index = np.arange(len(y))
    if size >= len(y):
        return index
    selected, _ = train_test_split(index, train_size=size, stratify=y,
                                   random_state=seed)
    return np.sort(selected)


def parse_list(text, convert=str):
    return [convert(item) for item in text.split(',') if item.strip()]


def parse_bounds(items, default_epsilon):
    """Parse ``bound[:epsilon]`` entries.

    Bounded settings need a small epsilon: with epsilon=0 the FORDE lower bound
    equals the empirical minimum, and ``cforde`` assigns zero density to any
    evidence value at that bound (``val != min.y`` in arf 0.2.5), so every
    evidence row containing a minimum value matches no leaf and forge returns
    NA. epsilon=1e-6 keeps draws inside the empirical support (measured, see
    the probe in SAMPLER_DIAGNOSTICS) while avoiding the zero-density rule.
    """
    out = []
    for item in items:
        if ':' in item:
            bound, eps = item.split(':')
            out.append((bound.strip(), float(eps)))
        else:
            out.append((item.strip(), default_epsilon))
    return out


def align_draws(draws, feature, n_rows, repeats):
    expected = pd.MultiIndex.from_product([range(n_rows), range(1, repeats + 1)],
                                          names=['test_position', 'replicate'])
    part = draws[draws.feature == feature]
    actual = pd.MultiIndex.from_frame(part[['test_position', 'replicate']])
    if actual.has_duplicates:
        raise ValueError(f'duplicate draws for {feature}')
    ordered = part.set_index(['test_position', 'replicate'])['value'].reindex(expected)
    return ordered.to_numpy(dtype=np.float64).reshape(n_rows, repeats)


def row_means(block):
    finite = np.isfinite(block)
    counts = finite.sum(axis=1)
    totals = np.where(finite, block, 0.0).sum(axis=1)
    means = np.full(len(block), np.nan)
    ok = counts > 0
    means[ok] = totals[ok] / counts[ok]
    return means


def snap_boundary(block, tol):
    """Snap values in ``[-tol, 0)`` to 0; return block, count, max magnitude.

    ``finite_bounds=local`` widens a leaf range by ``epsilon/2`` on each side,
    so a non-negative feature can produce an epsilon-scale negative. These are
    numerical-boundary artifacts, not draws from negative support; every
    affected value is counted and its magnitude reported so the correction is
    auditable. Values below ``-tol`` are left untouched and still fail the
    domain gate.
    """
    if tol <= 0:
        return block, 0, 0.0
    hit = (block < 0) & (block >= -tol)
    count = int(hit.sum())
    worst = float(-block[hit].min()) if count else 0.0
    return np.where(hit, 0.0, block), count, worst


def constraint_keep(block, feature, constraints, real_frame, repeats):
    """Boolean (n, pool) mask of proposals satisfying every constraint.

    ``constraints`` involve ``feature`` on one side; the other side is real
    evidence and is broadcast over proposals.
    """
    keep = np.isfinite(block)
    for row in constraints.itertuples():
        if row.lhs == feature:
            left = block
            right = np.repeat(real_frame[row.rhs].to_numpy(dtype=np.float64)[:, None],
                              repeats, axis=1)
        elif row.rhs == feature:
            right = block
            left = np.repeat(real_frame[row.lhs].to_numpy(dtype=np.float64)[:, None],
                             repeats, axis=1)
        else:
            continue
        ok = np.isfinite(left) & np.isfinite(right)
        satisfied = np.where(ok, (left <= right) if row.op == '<=' else (left >= right),
                             False)
        keep &= satisfied
    return keep


def select_constrained(block, keep, repeats):
    """Keep the first ``repeats`` accepted proposals per row (NaN padded).

    Proposals are i.i.d. given the evidence, so the first accepted proposal of
    a pool is distributed as a draw from the proposal density conditioned on
    the accepted region.
    """
    order = np.argsort(~keep, axis=1, kind='stable')
    take = order[:, :repeats]
    values = np.take_along_axis(block, take, axis=1)
    accepted = np.take_along_axis(keep, take, axis=1)
    values[~accepted] = np.nan
    acceptance = float(keep.sum() / max(int(np.isfinite(block).sum()), 1))
    return values, acceptance


def score_feature(feature, block, real_values, domain_row, train_low, train_high,
                  enforced, relaxed, real_frame, repeats, supported,
                  support_rate, leaf_support_rate):
    """All diagnostics for one feature under one sampler setting.

    ``support_rate`` is the share of test rows inside the global ARF-train
    range; ``leaf_support_rate`` the share for which FORDE found matching
    leaves. They are distinct estimands: the first is a data-support limit,
    the second a model-support limit. Coverage is only evaluated on rows that
    pass both, so a row the sampler cannot condition on never masquerades as
    a rejection/coverage failure.
    """
    n_rows = len(real_values)
    finite = np.isfinite(block)
    coverage = float(finite.any(axis=1).mean())
    conditionable = float(finite[supported].any(axis=1).mean()) if supported.any() else np.nan
    synth = block[finite]
    row_has_draw = finite.any(axis=1)
    real = real_values[row_has_draw]

    validation = validate(synth.reshape(-1, 1), domain_row, [feature]).iloc[0]
    ks_synth = float(ks_2samp(synth, real).statistic) if len(synth) and len(real) else np.nan
    # Test rows follow the dataset's class order, so the two halves are not
    # exchangeable unless shuffled; an unshuffled half-split would compare
    # classes, not samples, and inflate the null beyond any real KS.
    shuffled = np.random.default_rng(0).permutation(real)
    half = len(shuffled) // 2
    ks_null = (float(ks_2samp(shuffled[:half], shuffled[half:]).statistic)
               if half >= 10 else np.nan)
    real_out = float(((real < train_low) | (real > train_high)).mean()) if len(real) else np.nan
    synth_out = (float(((synth < train_low) | (synth > train_high)).mean())
                 if len(synth) else np.nan)

    means = row_means(block)
    ok = np.isfinite(means) & np.isfinite(real_values) & (coverage > 0)
    cond_rho = cond_nrmse = np.nan
    if ok.sum() >= 10 and np.std(means[ok]) > 0 and np.std(real_values[ok]) > 0:
        cond_rho = float(spearmanr(means[ok], real_values[ok]).statistic)
        rmse = float(np.sqrt(np.mean((means[ok] - real_values[ok]) ** 2)))
        sd = float(np.std(real_values[ok]))
        cond_nrmse = rmse / sd if sd > 0 else np.nan

    enforced_excess = constraint_excess(feature, block, enforced, real_frame,
                                        real_values, repeats)
    relaxed_excess = constraint_excess(feature, block, relaxed, real_frame,
                                       real_values, repeats)

    return dict(feature=feature, domain=domain_row.domain.iloc[0],
                n_draws=int(finite.sum()), coverage=coverage,
                support_rate=support_rate,
                leaf_support_rate=leaf_support_rate,
                coverage_conditionable=conditionable,
                n_invalid=int(validation.n_invalid),
                invalid_rate=float(validation.invalid_rate),
                negative_rate=float(validation.n_negative / max(len(synth), 1)),
                above_one_rate=float(validation.n_above_one / max(len(synth), 1)),
                non_integer_rate=float(validation.n_non_integer / max(len(synth), 1)),
                ks_synth=ks_synth, ks_null=ks_null,
                out_of_range_synth=synth_out, out_of_range_real=real_out,
                n_constraints=enforced_excess[0],
                max_constraint_excess=enforced_excess[1],
                n_relaxed_constraints=relaxed_excess[0],
                max_relaxed_constraint_excess=relaxed_excess[1],
                cond_rank_corr=cond_rho, cond_nrmse=cond_nrmse)


def constraint_excess(feature, block, constraints, real_frame, real_values, repeats):
    """Max violation-rate excess of ``constraints`` on selected draws."""
    excesses = []
    row_mask = np.isfinite(block).any(axis=1)
    for row in constraints.itertuples():
        if row.lhs == feature:
            left = block[row_mask]
            other = real_frame[row.rhs].to_numpy(dtype=np.float64)[row_mask]
            right = np.repeat(other[:, None], repeats, axis=1)
            real_left, real_right = real_values[row_mask], other
        elif row.rhs == feature:
            left = np.repeat(real_frame[row.lhs].to_numpy(dtype=np.float64)[row_mask, None],
                             repeats, axis=1)
            right = block[row_mask]
            real_left, real_right = left[:, 0], real_values[row_mask]
        else:
            continue
        pair = np.isfinite(left) & np.isfinite(right)
        if not pair.any():
            continue
        if row.op == '<=':
            synth_rate = float((left[pair] > right[pair]).mean())
            real_rate = float((real_left > real_right).mean())
        else:
            synth_rate = float((left[pair] < right[pair]).mean())
            real_rate = float((real_left < real_right).mean())
        excesses.append(synth_rate - real_rate)
    return len(excesses), (float(max(excesses)) if excesses else 0.0)


def run_r(task_name, seed, train_csv, test_csv, features, repeats, combos, tag,
          pool=1):
    INPUT.mkdir(parents=True, exist_ok=True)
    DRAWS.mkdir(parents=True, exist_ok=True)
    plan = dict(root=str(ROOT), task=task_name, seed=seed, repeats=repeats,
                pool=pool, features=features, train_csv=str(train_csv),
                test_csv=str(test_csv), out_dir=str(DRAWS), combos=combos)
    plan_path = INPUT / f'grid_plan_{task_name}_{seed}_{tag}.json'
    plan_path.write_text(json.dumps(plan, indent=2))
    start = time.time()
    result = subprocess.run(
        ['Rscript', str(ROOT / 'r' / 'sampler_grid.R'), str(plan_path)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'R grid sampler failed:\n{result.stdout}\n{result.stderr}')
    print(result.stdout.strip(), flush=True)
    print(f'[grid] R sampler {tag} took {time.time() - start:.0f}s', flush=True)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    DRAWS.mkdir(parents=True, exist_ok=True)
    features = args.features
    tasks = load_tasks([args.task])
    task = tasks[args.task]
    missing = set(features) - set(task.features)
    if missing:
        raise SystemExit(f'features not in task: {sorted(missing)}')

    train_idx, test_idx = load_locked(args.version, args.task, args.seed, args.split)
    test_rows = test_idx[stratified_subset(task.y[test_idx], args.test_samples, args.seed)]
    test_frame_all = task.frame.iloc[test_rows][task.features].reset_index(drop=True)
    test_frame = test_frame_all[features]
    train_frame = task.frame.iloc[train_idx][task.features]
    domains_all = classify(train_frame, task.features)
    constraint_table = discover_constraints(train_frame, task.features)
    domains_all.to_csv(INPUT / f'domains_{args.task}_{args.seed}.csv', index=False)
    constraint_table.to_csv(INPUT / f'constraints_{args.task}_{args.seed}.csv', index=False)
    domains = domains_all[domains_all.feature.isin(features)].reset_index(drop=True)
    involved = constraint_table[(constraint_table.lhs.isin(features))
                                | (constraint_table.rhs.isin(features))]
    enforced = involved[involved.hold_rate >= CONSTRAINT_HOLD]
    relaxed = involved[involved.hold_rate < CONSTRAINT_HOLD]
    print(f'[grid] {len(test_frame)} evidence rows, {len(features)} features, '
          f'{len(involved)} constraints involve grid features '
          f'({len(enforced)} enforced at hold >= {CONSTRAINT_HOLD})', flush=True)

    test_csv = INPUT / f'grid_test_{args.task}_{args.seed}_{args.test_samples}.csv'
    test_frame_all.to_csv(test_csv, index=False)

    all_rows, summaries = [], []
    for size in args.arf_train:
        train_rows = train_idx[stratified_subset(task.y[train_idx], size, args.seed)]
        train_csv = INPUT / f'grid_train_{args.task}_{args.seed}_{len(train_rows)}.csv'
        task.frame.iloc[train_rows][task.features].to_csv(train_csv, index=False)
        combos = []
        for bounds, eps in args.bounds:
            for trees in args.trees:
                for node in args.node_size:
                    combos.append(dict(
                        name=f'{bounds}_e{eps:g}_t{len(train_rows)}_n{trees}_m{node}_p{args.pool}',
                        finite_bounds=bounds, epsilon=eps,
                        num_trees=trees, min_node_size=node))
        tag = f'arf{len(train_rows)}'
        if not args.score_only:
            run_r(args.task, args.seed, train_csv, test_csv, features, args.repeats,
                  combos, tag, args.pool)
        support = task.frame.iloc[train_rows][task.features]
        support_low, support_high = support.min(), support.max()

        for combo in combos:
            name = combo['name']
            if args.score_only and not (DRAWS / f'grid_draws_{args.task}_{name}.csv').exists():
                print(f'[grid] score-only: no draws for {name}, skipped', flush=True)
                continue
            draws = pd.read_csv(DRAWS / f'grid_draws_{args.task}_{name}.csv')
            meta = json.loads((DRAWS / f'grid_meta_{args.task}_{name}.json').read_text())
            rows = []
            for feature in features:
                proposals = args.repeats * args.pool
                block = align_draws(draws, feature, len(test_frame), proposals)
                leaf_supported = np.isfinite(block).any(axis=1)
                dom_row = domains[domains.feature == feature]
                tol = (combo['epsilon'] * (float(dom_row.train_max.iloc[0])
                                           - float(dom_row.train_min.iloc[0])) / 2)
                block, snapped, snap_max = snap_boundary(block, tol)
                keep = constraint_keep(block, feature, enforced, test_frame_all, proposals)
                block, acceptance = select_constrained(block, keep, args.repeats)
                outside = [c for c in task.features if c != feature]
                low = support_low[outside].to_numpy(dtype=np.float64)
                high = support_high[outside].to_numpy(dtype=np.float64)
                values = test_frame_all[outside].to_numpy(dtype=np.float64)
                in_range = ((values >= low) & (values <= high)).all(axis=1)
                supported = in_range & leaf_supported
                scored = score_feature(
                    feature, block,
                    test_frame[feature].to_numpy(dtype=np.float64),
                    dom_row,
                    float(dom_row.train_min.iloc[0]),
                    float(dom_row.train_max.iloc[0]),
                    enforced, relaxed, test_frame_all, args.repeats, supported,
                    float(in_range.mean()), float(leaf_supported.mean()))
                scored.update(acceptance=acceptance, n_snapped=snapped,
                              snap_max=snap_max)
                rows.append(scored)
            per_feature = pd.DataFrame(rows)
            per_feature.insert(0, 'combo', name)
            all_rows.append(per_feature)
            summaries.append(summarise(per_feature, meta, combo))
            print(f'[grid] scored {name}', flush=True)

    grid = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary['recommended'] = False
    passing = summary[summary.pass_gate].sort_values(
        ['max_ks_synth', 'fit_seconds'])
    if len(passing):
        summary.loc[passing.index[0], 'recommended'] = True
    grid.to_csv(OUT / 'grid_features.csv', index=False)
    summary.to_csv(OUT / 'grid_summary.csv', index=False)
    write_manifest(args, features, summary)
    print(summary.to_string(index=False), flush=True)
    print(f'Grid artifacts written to {OUT.relative_to(ROOT)}', flush=True)


def summarise(per_feature, meta, combo):
    ks_excess = (per_feature.ks_synth
                 - np.maximum(2 * per_feature.ks_null, KS_FLOOR))
    out_excess = per_feature.out_of_range_synth - per_feature.out_of_range_real
    passes = bool(
        (per_feature.coverage_conditionable >= MIN_COVERAGE).all()
        and (per_feature.support_rate >= MIN_SUPPORT).all()
        and (per_feature.leaf_support_rate >= MIN_SUPPORT).all()
        and (per_feature.n_invalid == 0).all()
        and (out_excess <= MAX_OUT_OF_RANGE_EXCESS).all()
        and (per_feature.max_constraint_excess <= MAX_CONSTRAINT_EXCESS).all()
        and (ks_excess <= 0).all())
    return dict(
        combo=combo['name'], finite_bounds=combo['finite_bounds'],
        num_trees=combo['num_trees'], min_node_size=combo['min_node_size'],
        train_rows=meta['train_rows'], fit_seconds=meta['fit_seconds'],
        discriminator_accuracy=float(
            np.atleast_1d(meta['discriminator_accuracy'])[-1]),
        coverage_min=float(per_feature.coverage.min()),
        coverage_conditionable_min=float(per_feature.coverage_conditionable.min()),
        support_rate_min=float(per_feature.support_rate.min()),
        leaf_support_rate_min=float(per_feature.leaf_support_rate.min()),
        acceptance_min=float(per_feature.acceptance.min()),
        n_snapped_total=int(per_feature.n_snapped.sum()),
        snap_max=float(per_feature.snap_max.max()),
        total_invalid_rate=float(per_feature.n_invalid.sum()
                                 / max(per_feature.n_draws.sum(), 1)),
        max_negative_rate=float(per_feature.negative_rate.max()),
        max_above_one_rate=float(per_feature.above_one_rate.max()),
        max_non_integer_rate=float(per_feature.non_integer_rate.max()),
        max_ks_synth=float(per_feature.ks_synth.max()),
        max_ks_null=float(per_feature.ks_null.max()),
        max_ks_excess=float(ks_excess.max()),
        max_out_of_range_excess=float(out_excess.max()),
        max_constraint_excess=float(per_feature.max_constraint_excess.max()),
        max_relaxed_constraint_excess=float(
            per_feature.max_relaxed_constraint_excess.max()),
        min_cond_rank_corr=float(per_feature.cond_rank_corr.min()),
        max_cond_nrmse=float(per_feature.cond_nrmse.max()),
        pass_gate=passes,
    )


def write_manifest(args, features, summary):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/run_sampler_grid.py + r/sampler_grid.R',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        task=args.task, seed=args.seed, split=args.split,
        split_version=args.version, split_files=[
            str(p.relative_to(ROOT)) for p in
            sorted((ROOT / 'results' / 'splits' / args.version).glob(
                f'split_{args.task}_{args.seed}_{args.split}.joblib'))],
        features=features, repeats=args.repeats, test_samples=args.test_samples,
        arf_train_sizes=args.arf_train, pool=args.pool,
        bounds=[dict(bound=b, epsilon=e) for b, e in args.bounds],
        trees=args.trees, node_sizes=args.node_size, epsilon=args.epsilon,
        gate=dict(min_coverage=MIN_COVERAGE, min_support=MIN_SUPPORT,
                  max_invalid=0,
                  max_out_of_range_excess=MAX_OUT_OF_RANGE_EXCESS,
                  max_constraint_excess=MAX_CONSTRAINT_EXCESS,
                  constraint_hold=CONSTRAINT_HOLD,
                  ks_rule=f'ks_synth <= max({KS_FLOOR}, 2*ks_null)',
                  coverage_scope='coverage_conditionable: rows inside the '
                                 'ARF-train range; support_rate gated separately',
                  domains='derived from training rows of the locked split only',
                  constraint_enforcement='rejection: first repeats accepted '
                                          'proposals per row from a pool of '
                                          'repeats*pool; acceptance reported'),
        arf_version=json.loads((DRAWS / (
            f'grid_meta_{args.task}_{summary.combo.iloc[0]}.json')).read_text())['arf_version'],
    )
    (OUT / 'grid_manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='binary')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', default='source_aware')
    parser.add_argument('--version', default='source_aware_v1')
    parser.add_argument('--test-samples', type=int, default=500)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--arf-train', default='3000,10000')
    parser.add_argument('--bounds', default='no:0,local:1e-6,global:1e-6')
    parser.add_argument('--trees', default='20')
    parser.add_argument('--node-size', default='20')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='default epsilon for entries written without bound:eps')
    parser.add_argument('--pool', type=int, default=1,
                        help='proposals per kept draw; the first repeats accepted '
                             'proposals are kept (rejection sampling)')
    parser.add_argument('--features', default=','.join(DEFAULT_FEATURES))
    parser.add_argument('--out-name', default='sampler_v2',
                        help='results subdirectory; use a new name for each '
                             'independent grid so earlier artifacts are not '
                             'overwritten')
    parser.add_argument('--score-only', action='store_true',
                        help='re-score existing draws without fitting ARF')
    args = parser.parse_args()
    global OUT, DRAWS, INPUT
    OUT = ROOT / 'results' / args.out_name
    DRAWS = OUT / 'draws'
    INPUT = OUT / 'input'
    args.arf_train = parse_list(args.arf_train, int)
    args.bounds = parse_bounds(parse_list(args.bounds), args.epsilon)
    args.trees = parse_list(args.trees, int)
    args.node_size = parse_list(args.node_size, int)
    args.features = parse_list(args.features)
    run(args)


if __name__ == '__main__':
    main()
