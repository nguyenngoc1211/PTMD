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
                        ``coverage_conditionable`` restricts this to rows whose
                        evidence lies inside the ARF-training range, and
                        ``support_rate`` is the fraction of such rows. Rows
                        outside the ARF range cannot be conditioned on by any
                        density estimated from those rows, so they are a
                        property of the evidence sample and are gated
                        separately (>= 0.98) instead of counting as sampler
                        failure;
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
    results/group_experiments/<out-name>/...  (default out-name: cfi_v2)
    report/group_cfi_<suffix>.md  (suffix from --out-name; plain group_cfi.md
    only for the historical out-name ``cfi``)
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
from feature_domains import classify, discover_constraints, validate
from generate_report import table as table_text
from group_protocol import GROUPS, MODELS, TASKS, group_map, load_tasks, make_model

OUT = ROOT / 'results' / 'group_experiments' / 'cfi'
DRAWS = OUT / 'draws'
INPUT = OUT / 'input'


def set_out(name):
    """Point every writer at a versioned results directory."""
    global OUT, DRAWS, INPUT
    OUT = ROOT / 'results' / 'group_experiments' / name
    DRAWS = OUT / 'draws'
    INPUT = OUT / 'input'

# Gate thresholds, fixed before looking at any result.
MIN_COVERAGE = 0.99
MIN_SUPPORT = 0.98
MAX_NEGATIVE_RATE = 0.0
OUT_OF_RANGE_SLACK = 0.02
MAX_CONSTRAINT_EXCESS = 0.02
CONSTRAINT_HOLD = 0.999
KS_FLOOR = 0.10
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


def align_group(frame, columns, indices, repeats, constants=None):
    """Return draws as (rows, repeats, k), enforcing the R/Python contract.

    The sampler omits rows whose evidence has no support in the training data,
    so a missing (position, replicate) pair is a legitimate missing draw and
    becomes NaN (counted as missing coverage downstream). Duplicated or
    out-of-range positions are contract violations and raise.

    Members listed in ``constants`` are constant in the ARF training frame and
    therefore not sampled by R; their training value is filled in. A member
    missing for any other reason raises, so a silently dropped varying feature
    cannot pass as a degenerate one.
    """
    expected = pd.MultiIndex.from_product([indices, range(1, repeats + 1)],
                                          names=['test_position', 'replicate'])
    actual = pd.MultiIndex.from_frame(frame[['test_position', 'replicate']])
    if actual.has_duplicates:
        raise ValueError('duplicate (test_position, replicate) in draws')
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise ValueError(f'draws contain {len(unexpected)} unexpected positions')
    constants = constants or {}
    missing = [c for c in columns if c not in frame.columns]
    unknown = [c for c in missing if c not in constants]
    if unknown:
        raise ValueError(f'sampler returned no draws for varying features {unknown}')
    present = [c for c in columns if c in frame.columns]
    if present:
        ordered = frame.set_index(['test_position', 'replicate'])[present].reindex(expected)
    else:
        ordered = pd.DataFrame(index=expected)
    for name in missing:
        ordered[name] = constants[name]
    ordered = ordered[columns]
    return ordered.to_numpy(dtype=np.float64).reshape(len(indices), repeats, len(columns))


def row_loss(y, probability, n_classes):
    if n_classes == 2:
        p = np.clip(probability[:, 1], 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    p = np.clip(probability[np.arange(len(y)), y], 1e-15, 1.0)
    return -np.log(p)


def run_r(task_name, seed, task, train_rows, test_rows, repeats,
          finite_bounds='no', epsilon=0.0, pool=1):
    """Write the plan, run the R sampler, return the draw directory."""
    INPUT.mkdir(parents=True, exist_ok=True)
    DRAWS.mkdir(parents=True, exist_ok=True)
    features = task.features
    train_path = INPUT / f'{task_name}_{seed}_train.csv'
    test_path = INPUT / f'{task_name}_{seed}_test.csv'
    task.frame.iloc[train_rows][features].to_csv(train_path, index=False)
    task.frame.iloc[test_rows][features].to_csv(test_path, index=False)
    plan = dict(root=str(ROOT), task=task_name, seed=seed, repeats=repeats,
                pool=pool, features=features, groups=group_map(features),
                train_csv=str(train_path), test_csv=str(test_path),
                out_dir=str(DRAWS), arf_num_trees=CONFIG['arf_num_trees'],
                arf_min_node_size=CONFIG['arf_min_node_size'],
                finite_bounds=finite_bounds, epsilon=epsilon)
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


def arf_support(task, seed):
    """Per-feature (min, max) of the ARF training rows written by ``run_r``.

    Returns ``(None, None)`` when the training CSV is absent (direct API use
    without the R round-trip). The support gate is then not evaluated; callers
    report ``support_rate`` as NaN rather than assuming every row is inside a
    support range nobody measured.
    """
    path = INPUT / f'{task.name}_{seed}_train.csv'
    if not path.exists():
        return None, None
    frame = pd.read_csv(path)
    return (frame.min().to_numpy(dtype=np.float64),
            frame.max().to_numpy(dtype=np.float64))


def snap_values(block, tolerances):
    """Snap values in ``[-tol_j, 0)`` to 0 per member.

    ``finite_bounds=local`` widens each leaf range by ``epsilon/2`` on each
    side, so a non-negative feature can yield an epsilon-scale negative.
    Affected values are counted and their maximum magnitude returned so the
    correction is auditable.
    """
    if block.size == 0:
        return block, 0, 0.0
    tol = np.asarray(tolerances, dtype=np.float64)[None, None, :]
    hit = (block < 0) & (block >= -tol)
    count = int(hit.sum())
    worst = float(-block[hit].min()) if count else 0.0
    return np.where(hit, 0.0, block), count, worst


def group_constraint_keep(block, members, constraints, X_test, feature_index,
                          proposals):
    """(rows, proposals) mask of joint draws satisfying every constraint."""
    keep = np.isfinite(block).all(axis=2)
    if constraints is None or not len(constraints):
        return keep
    position = {c: j for j, c in enumerate(members)}
    for row in constraints.itertuples():
        left_in, right_in = row.lhs in position, row.rhs in position
        if not (left_in or right_in):
            continue
        if left_in:
            left = block[:, :, position[row.lhs]]
        else:
            left = np.repeat(X_test[:, feature_index[row.lhs]][:, None],
                             proposals, axis=1)
        if right_in:
            right = block[:, :, position[row.rhs]]
        else:
            right = np.repeat(X_test[:, feature_index[row.rhs]][:, None],
                              proposals, axis=1)
        ok = np.isfinite(left) & np.isfinite(right)
        keep &= np.where(ok, (left <= right) if row.op == '<=' else (left >= right),
                         False)
    return keep


def select_group(block, keep, repeats):
    """First ``repeats`` accepted joint proposals per row, NaN padded."""
    rows = np.arange(block.shape[0])[:, None]
    order = np.argsort(~keep, axis=1, kind='stable')[:, :repeats]
    values = block[rows, order]
    accepted = keep[rows, order]
    values[~accepted] = np.nan
    finite_proposals = int(np.isfinite(block).all(axis=2).sum())
    acceptance = float(keep.sum() / max(finite_proposals, 1))
    return values, acceptance


def group_importance(task, model, test_rows, repeats, seed, constraints=None,
                     domains=None, pool=1, epsilon=0.0):
    """Compute per-row group CFI and sampler diagnostics for one seed.

    Proposals (``repeats * pool`` per row) are snapped at the epsilon boundary,
    filtered by the enforced cross-feature constraints derived from training
    rows, and the first ``repeats`` accepted joint draws are kept. The group
    conditional importance is the mean log-loss increase over accepted draws,
    subtracted from the model's loss on the unmodified row.
    """
    features = task.features
    X = task.frame[features].to_numpy(dtype=np.float64)
    X_test = X[test_rows]
    y_test = task.y[test_rows]
    train_rows = np.setdiff1d(np.arange(len(X)), test_rows)
    baseline = row_loss(y_test, model.predict_proba(X_test), task.n_classes)
    support_low, support_high = arf_support(task, seed)
    feature_index = {c: j for j, c in enumerate(features)}
    proposals = repeats * pool

    rows, diagnostics = [], []
    for group in GROUPS:
        members = task.features_of(group)
        local = [features.index(c) for c in members]
        outside = [j for j, c in enumerate(features) if c not in members]
        if support_low is None:
            in_range = np.ones(len(y_test), dtype=bool)
            support_rate = np.nan
        else:
            in_range = ((X_test[:, outside] >= support_low[outside])
                        & (X_test[:, outside] <= support_high[outside])).all(axis=1)
            support_rate = float(in_range.mean())
        constants = {}
        for j in local:
            if support_low is not None and support_low[j] == support_high[j]:
                column = X[train_rows, j]
                if column.min() != column.max():
                    raise ValueError(
                        f'{features[j]} is constant in the ARF subset '
                        'but varies in the full training rows')
                constants[features[j]] = float(support_low[j])
        draws = pd.read_csv(DRAWS / f'draws_{task.name}_{seed}_{group}.csv')
        raw = align_group(draws, members, np.arange(len(test_rows)), proposals,
                          constants)
        # Rows where forge found no matching leaf come back all-NaN. That is a
        # model-support limit, not a rejection/coverage failure, so it is
        # measured separately and gated as support.
        leaf_supported = np.isfinite(raw).all(axis=2).any(axis=1)
        leaf_support_rate = float(leaf_supported.mean())
        supported = in_range & leaf_supported
        if domains is None:
            tolerances = np.zeros(len(members))
        else:
            table = domains.set_index('feature')
            tolerances = np.array([
                epsilon * (table.loc[c, 'train_max'] - table.loc[c, 'train_min']) / 2
                for c in members])
        raw, snapped, snap_max = snap_values(raw, tolerances)
        keep = group_constraint_keep(raw, members, constraints, X_test,
                                     feature_index, proposals)
        block, acceptance = select_group(raw, keep, repeats)
        draw_valid = np.isfinite(block).all(axis=2)
        valid = draw_valid.any(axis=1)
        differences = np.full(len(y_test), np.nan)
        if valid.any():
            selected = draw_valid.reshape(-1)
            repeated = np.repeat(X_test, repeats, axis=0)
            repeated[:, local] = block.reshape(-1, len(members))
            loss = row_loss(np.repeat(y_test, repeats)[selected],
                            model.predict_proba(repeated[selected]),
                            task.n_classes)
            flat_rows = np.repeat(np.arange(len(y_test)), repeats)[selected]
            totals = np.zeros(len(y_test))
            counts = np.zeros(len(y_test), dtype=int)
            np.add.at(totals, flat_rows, loss)
            np.add.at(counts, flat_rows, 1)
            differences[valid] = (totals[valid] / counts[valid]
                                  - baseline[valid])
        stats = inference(differences, CONFIG['alpha'])
        stats['conditional_importance'] = stats.pop('importance')
        stats['conditional_importance_se'] = stats.pop('se')
        stats['coverage_conditionable'] = (float(valid[supported].mean())
                                           if supported.any() else np.nan)
        stats['support_rate'] = support_rate
        stats['leaf_support_rate'] = leaf_support_rate
        rows.append(dict(task=task.name, seed=seed, group=group,
                         n_features=len(members),
                         n_sampled_features=len(members) - len(constants),
                         coverage=float(valid.mean()), acceptance=acceptance,
                         n_snapped=snapped, snap_max=snap_max, **stats))
        diagnostic = sampler_diagnostics(
            task, group, members, block, draw_valid, valid, supported, X_test,
            train_rows, X, seed, support_rate, constraints, domains,
            acceptance, snapped, snap_max, leaf_support_rate)
        diagnostic['n_sampled_features'] = len(members) - len(constants)
        diagnostics.append(diagnostic)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def sampler_diagnostics(task, group, members, block, draw_valid, valid, supported,
                        X_test, train_rows, X, seed, support_rate=np.nan,
                        constraints=None, domains=None, acceptance=np.nan,
                        snapped=0, snap_max=0.0, leaf_support_rate=np.nan):
    """Screen the conditional sampler for one group.

    ``support_rate`` is NaN when no ARF training frame is available, in which
    case the support gate is skipped instead of being silently satisfied.
    """
    positions = [task.features.index(c) for c in members]
    train_block = X[np.ix_(train_rows, positions)]
    low = train_block.min(axis=0)
    high = train_block.max(axis=0)

    synthetic_matrix = block.reshape(-1, len(members))
    synthetic_matrix = synthetic_matrix[np.isfinite(synthetic_matrix).all(axis=1)]
    real_matrix = X_test[:, positions]

    negative = float((synthetic_matrix < 0).mean()) if synthetic_matrix.size else np.nan
    synth_out = float(((synthetic_matrix < low) | (synthetic_matrix > high)).mean()) \
        if synthetic_matrix.size else np.nan
    real_out = float(((real_matrix < low) | (real_matrix > high)).mean()) \
        if real_matrix.size else np.nan

    if domains is None:
        invalid = 0
    else:
        table = domains[domains.feature.isin(members)].reset_index(drop=True)
        if synthetic_matrix.size:
            invalid = int(validate(synthetic_matrix, table, members).n_invalid.sum())
        else:
            invalid = 0

    # Synthetic draws (rows x repeats) are compared against one value per real
    # row; the noise floor comes from splitting real rows in half, so repeated
    # draws never masquerade as independent observations. Test rows follow the
    # dataset's class order, so halves are shuffled before comparing.
    ks_synth = float(np.mean([ks_2samp(synthetic_matrix[:, j], real_matrix[:, j]).statistic
                              for j in range(len(members))]))
    shuffled = np.random.default_rng(0).permutation(len(real_matrix))
    half = len(shuffled) // 2
    ks_null = float(np.mean([ks_2samp(real_matrix[shuffled[:half], j],
                                      real_matrix[shuffled[half:], j]).statistic
                             for j in range(len(members))]))

    if len(members) > 1 and synthetic_matrix.shape[0] > 10:
        corr_synth = pd.DataFrame(synthetic_matrix).corr(method='spearman').to_numpy()
        corr_real = pd.DataFrame(real_matrix).corr(method='spearman').to_numpy()
        corr_error = float(np.linalg.norm(np.nan_to_num(corr_synth - corr_real))
                           / np.sqrt(len(members) * (len(members) - 1)))
        corr_null = float(np.linalg.norm(np.nan_to_num(
            pd.DataFrame(real_matrix[shuffled[:half]]).corr(method='spearman').to_numpy()
            - pd.DataFrame(real_matrix[shuffled[half:]]).corr(method='spearman').to_numpy()))
            / np.sqrt(len(members) * (len(members) - 1)))
    else:
        corr_error = corr_null = 0.0

    enforced = relaxed = None
    if constraints is not None and len(constraints):
        enforced = constraints[constraints.hold_rate >= CONSTRAINT_HOLD]
        relaxed = constraints[constraints.hold_rate < CONSTRAINT_HOLD]
    excess_enforced = group_constraint_excess(members, block, draw_valid, enforced,
                                              X_test, task)
    excess_relaxed = group_constraint_excess(members, block, draw_valid, relaxed,
                                             X_test, task)

    reasons = []
    conditionable = float(valid[supported].mean()) if supported.any() else np.nan
    if float(valid.mean()) < MIN_SUPPORT:
        reasons.append('coverage (all rows)')
    if np.isfinite(conditionable) and conditionable < MIN_COVERAGE:
        reasons.append('coverage')
    if np.isfinite(support_rate) and support_rate < MIN_SUPPORT:
        reasons.append('test outside train support')
    if np.isfinite(leaf_support_rate) and leaf_support_rate < MIN_SUPPORT:
        reasons.append('no matching FORDE leaves')
    if invalid > 0:
        reasons.append('invalid domain values')
    if not np.isnan(negative) and negative > MAX_NEGATIVE_RATE:
        reasons.append('negative values')
    if not np.isnan(synth_out) and synth_out > real_out + OUT_OF_RANGE_SLACK:
        reasons.append('outside train support')
    if excess_enforced > MAX_CONSTRAINT_EXCESS:
        reasons.append('cross-feature constraints')
    if ks_synth > max(KS_FLOOR, 2 * ks_null):
        reasons.append('marginal KS')
    if corr_error > max(CORR_ERROR_FACTOR * corr_null, CORR_ERROR_FLOOR):
        reasons.append('within-group structure')

    return dict(task=task.name, seed=seed, group=group, n_features=len(members),
                coverage=float(valid.mean()),
                coverage_conditionable=conditionable,
                support_rate=support_rate,
                leaf_support_rate=leaf_support_rate,
                acceptance=acceptance,
                n_snapped=snapped, snap_max=snap_max,
                n_invalid=invalid,
                negative_rate=negative,
                out_of_range_synth=synth_out, out_of_range_real=real_out,
                max_constraint_excess=excess_enforced,
                max_relaxed_constraint_excess=excess_relaxed,
                ks_synth=ks_synth, ks_null=ks_null,
                corr_error=corr_error, corr_null=corr_null,
                gate_pass=not reasons, gate_reasons=';'.join(reasons))


def group_constraint_excess(members, block, draw_valid, constraints, X_test, task):
    """Max violation-rate excess of ``constraints`` over selected joint draws."""
    if constraints is None or not len(constraints):
        return 0.0
    position = {c: j for j, c in enumerate(members)}
    excesses = []
    for row in constraints.itertuples():
        left_in, right_in = row.lhs in position, row.rhs in position
        if not (left_in or right_in):
            continue
        if left_in:
            left = block[:, :, position[row.lhs]]
            real_left = X_test[:, task.features.index(row.lhs)]
        else:
            left = np.repeat(X_test[:, task.features.index(row.lhs)][:, None],
                             block.shape[1], axis=1)
            real_left = left[:, 0]
        if right_in:
            right = block[:, :, position[row.rhs]]
            real_right = X_test[:, task.features.index(row.rhs)]
        else:
            right = np.repeat(X_test[:, task.features.index(row.rhs)][:, None],
                              block.shape[1], axis=1)
            real_right = right[:, 0]
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
    return float(max(excesses)) if excesses else 0.0


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    tasks = load_tasks(args.tasks)
    cfi_rows, diagnostic_rows, meta_rows = [], [], []
    for task_name, task in tasks.items():
        for seed in args.seeds:
            if args.split == 'source_aware':
                from split_manager import load_locked
                train, test = load_locked(args.split_version, task_name, seed)
            else:
                train, test = task.split(seed)
            arf_train = stratified_subset(task.y[train], args.arf_train, seed)
            importance_test = stratified_subset(task.y[test], args.test_samples, seed)
            run_r(task_name, seed, task, train[arf_train], test[importance_test],
                  args.repeats, args.finite_bounds, args.epsilon, args.pool)
            train_frame = task.frame.iloc[train][task.features]
            domains = classify(train_frame, task.features)
            constraints = discover_constraints(train_frame, task.features)
            domains.to_csv(OUT / f'domains_{task_name}_{seed}.csv', index=False)
            constraints.to_csv(OUT / f'constraints_{task_name}_{seed}.csv', index=False)
            model = make_model(args.model, task, seed)
            model.fit(task.frame.iloc[train][task.features].to_numpy(dtype=np.float64),
                      task.y[train])
            importance, diagnostics = group_importance(
                task, model, test[importance_test], args.repeats, seed,
                constraints=constraints, domains=domains, pool=args.pool,
                epsilon=args.epsilon)
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
            coverage_conditionable_min=float(block.coverage_conditionable.min()),
            support_rate_min=(float(block.support_rate.min())
                              if block.support_rate.notna().any() else np.nan),
            leaf_support_rate_min=float(block.leaf_support_rate.min()),
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
                 'unconditional fallback. Hàng nằm ngoài support dữ liệu **hoặc** không '
                 'khớp leaf FORDE nào bị loại khỏi coverage và được gate riêng bằng '
                 'support_rate / leaf_support_rate; coverage chỉ đo trên phần giao.')
    lines.append('- draw được chọn trong một pool proposal độc lập theo điều kiện; '
                 'acceptance (tỉ lệ proposal được nhận) được báo cáo, không ẩn.')
    lines.append(f'- {args.repeats} draw/hàng từ pool {args.pool}× proposal; proposal vi '
                 f'phạm ràng buộc cross-feature (hold ≥ {CONSTRAINT_HOLD}) bị loại, '
                 f'{args.arf_train} hàng ARF train, {args.test_samples} hàng test, '
                 f'model `{args.model}`.')
    lines.append(f'- Chia dữ liệu: `{args.split}`'
                 + (f' (khóa `{args.split_version}`)' if args.split == 'source_aware'
                    else '') +
                 '; bộ sinh `finite_bounds=' + args.finite_bounds +
                 f'`, epsilon={args.epsilon:g}. Domain và ràng buộc chỉ suy từ hàng '
                 'train của split.')
    lines.append('- Ngưỡng gate **đặt trước**: coverage (hỗ trợ) ≥ '
                 f'{MIN_COVERAGE}, coverage toàn bộ hàng ≥ {MIN_SUPPORT}, support dữ '
                 f'liệu và leaf support ≥ {MIN_SUPPORT}, không có giá trị âm, '
                 f'KS biên ≤ max({KS_FLOOR}, 2×null), tỉ lệ ngoài support không vượt '
                 f'quá {OUT_OF_RANGE_SLACK} so với dữ liệu thật, sai lệch cấu trúc trong '
                 f'nhóm ≤ max({CORR_ERROR_FACTOR}× null, {CORR_ERROR_FLOOR}), '
                 f'vi phạm ràng buộc cross-feature ≤ {MAX_CONSTRAINT_EXCESS}.\n')
    lines.append('## 2. Chẩn đoán bộ sinh\n')
    display = diagnostics[['task', 'seed', 'group', 'coverage',
                           'coverage_conditionable', 'leaf_support_rate',
                           'negative_rate', 'out_of_range_synth',
                           'out_of_range_real', 'ks_synth', 'ks_null',
                           'corr_error', 'corr_null', 'gate_pass',
                           'gate_reasons']].copy()
    display.columns = ['task', 'seed', 'nhóm', 'coverage', 'coverage (hỗ trợ)',
                       'tỉ lệ có leaf', 'tỉ lệ âm',
                       'ngoài support (synth)', 'ngoài support (real)',
                       'KS synth', 'KS null', 'lỗi cấu trúc', 'null cấu trúc',
                       'gate', 'lý do']
    lines.append(table_text(display.round(5)) + '\n')
    figure_root = OUT.relative_to(ROOT)
    lines.append(f'![Chẩn đoán bộ sinh](../{figure_root}/figures/'
                 'sampler_diagnostics_binary.png)\n')
    lines.append('## 3. Group CFI\n')
    display = summary[['task', 'group', 'n_features', 'cfi_mean', 'cfi_sd',
                       'coverage_conditionable_min', 'support_rate_min',
                       'leaf_support_rate_min', 'n_seeds_significant',
                       'gate_pass', 'gate_reasons', 'rank']].copy()
    display.columns = ['task', 'nhóm', 'số feature', 'CFI TB', 'SD qua seed',
                       'coverage (hỗ trợ) min', 'support dữ liệu min',
                       'tỉ lệ có leaf min', 'số seed có q<0.05 & ci_lo>0', 'gate',
                       'lý do fail', 'hạng']
    lines.append(table_text(display.round(6)) + '\n')
    lines.append(f'![Group CFI](../{figure_root}/figures/'
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
    suffix = args.out_name.removeprefix('cfi').lstrip('_')
    name = f'group_cfi_{suffix}.md' if suffix else 'group_cfi.md'
    (ROOT / 'report' / name).write_text('\n'.join(lines))
    print(f'Generated report/{name}', flush=True)


def write_manifest(args):
    data_path = ROOT / CONFIG['data']
    manifest = dict(
        script='python/run_group_cfi.py',
        sampler='r/group_conditional_sampling.R',
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
        dataset=str(CONFIG['data']),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        config=CONFIG, tasks=args.tasks, seeds=args.seeds, model=args.model,
        repeats=args.repeats, pool=args.pool, arf_train=args.arf_train,
        test_samples=args.test_samples,
        split=args.split, split_version=args.split_version,
        finite_bounds=args.finite_bounds, epsilon=args.epsilon,
        conditioning='all features outside the group, group sampled jointly',
        constraint_enforcement='rejection of proposals violating train-derived '
                               f'constraints with hold >= {CONSTRAINT_HOLD}; '
                               'first repeats accepted per row',
        domains='derived from training rows of the locked split only',
        gate=dict(min_coverage=MIN_COVERAGE,
                  max_negative_rate=MAX_NEGATIVE_RATE,
                  out_of_range_slack=OUT_OF_RANGE_SLACK,
                  max_constraint_excess=MAX_CONSTRAINT_EXCESS,
                  constraint_hold=CONSTRAINT_HOLD,
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
    parser.add_argument('--pool', type=int, default=20,
                        help='proposals per row and kept draw; the first repeats '
                             'proposals satisfying enforced constraints are kept')
    parser.add_argument('--arf-train', type=int, default=3000)
    parser.add_argument('--test-samples', type=int, default=1500)
    parser.add_argument('--finite-bounds', default=CONFIG.get('finite_bounds', 'no'),
                        choices=['no', 'local', 'global'])
    parser.add_argument('--epsilon', type=float, default=CONFIG.get('epsilon', 0.0))
    parser.add_argument('--split', default='source_aware',
                        choices=['random', 'source_aware'])
    parser.add_argument('--split-version', default='source_aware_v1')
    parser.add_argument('--out-name', default='cfi_v2')
    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    args.seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if args.model not in MODELS:
        parser.error(f'unknown model: {args.model}')
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        parser.error(f'unknown tasks: {sorted(unknown)}')
    if args.out_name != 'cfi':
        set_out(args.out_name)
    run(args)


if __name__ == '__main__':
    main()
