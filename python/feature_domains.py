"""Feature value domains and cross-feature constraints for the ARF/FORDE sampler.

Why a domain table is needed
----------------------------
The sampler gate in ``run_group_cfi`` only asks "is this value negative?". The
audit showed that question is too blunt: 55 features do not share one domain.
``psxview.*_false_avg`` and ``ldrmodules.*_avg`` are ratios of an indicator
counted over the captured snapshots, so their true support is [0, 1]; a draw of
-0.3 is invalid, but so is a draw of 1.7, and the old gate only caught the
first. ``handles.nhandles`` reaches 1.0e6, where a negative value is the least
of the problems.

Domains are derived from the training rows rather than from column names,
because the names mislead: ``ldrmodules.not_in_load_avg`` sounds like a count
but its maximum in this dataset is 0.53.

Classes
-------
``constant``        one distinct value in train; carries no density, excluded
                    from ARF (the samplers already do this).
``binary``          exactly two distinct values, both in {0, 1}.
``bounded_ratio``   non-integer, min >= 0 and max <= 1: a proportion.
``count``           integer, min >= 0.
``continuous``      everything else, non-negative in practice.

Constraint discovery
--------------------
ARF models a joint density but not hard arithmetic relations between columns.
Two features can therefore be individually in-domain while their *combination*
is impossible (e.g. a count of processes flagged as missing from a plugin list
exceeding the process count). ``discover_constraints`` finds every ordered pair
with a strong monotone relation and records which inequality direction holds on
essentially all real rows. Those become checks applied to synthetic draws.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

DOMAINS = ('constant', 'binary', 'bounded_ratio', 'count', 'continuous')


def classify_column(values) -> tuple[str, str]:
    """Return (domain, rationale) for one training column."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError('column has no finite values')
    unique = np.unique(finite)
    if unique.size == 1:
        return 'constant', f'single value {unique[0]:g} in training rows'
    if unique.size == 2 and set(unique).issubset({0.0, 1.0}):
        return 'binary', f'two values {unique.tolist()} (indicator)'
    nonneg = finite.min() >= 0
    integer = np.allclose(finite, np.round(finite))
    if nonneg and finite.max() <= 1:
        return ('bounded_ratio',
                f'min={finite.min():.4g} max={finite.max():.4g} within [0,1] '
                f'and non-integer: proportion')
    if integer and nonneg:
        return 'count', f'integer, min={finite.min():.0f}, max={finite.max():.0f}'
    rationale = f'min={finite.min():.4g}, max={finite.max():.4g}'
    if not nonneg:
        rationale += ' (WARNING: negative values in real data)'
    return 'continuous', rationale


def classify(frame, features) -> pd.DataFrame:
    rows = []
    for name in features:
        domain, rationale = classify_column(frame[name].to_numpy())
        rows.append(dict(feature=name, group=name.split('.')[0], domain=domain,
                         rationale=rationale,
                         train_min=float(frame[name].min()),
                         train_max=float(frame[name].max()),
                         n_unique=int(frame[name].nunique())))
    table = pd.DataFrame(rows)
    unknown = set(table.domain) - set(DOMAINS)
    if unknown:
        raise ValueError(f'unknown domains {unknown}')
    return table


def domain_bounds(domains: pd.DataFrame) -> dict:
    """Per-feature (low, high) allowed interval; ``high`` is None when unbounded."""
    bounds = {}
    for row in domains.itertuples():
        low = 0.0 if row.domain in ('count', 'bounded_ratio', 'continuous',
                                    'binary') else None
        if row.domain == 'bounded_ratio':
            high = 1.0
        elif row.domain == 'binary':
            high = 1.0
        else:
            high = None
        bounds[row.feature] = (low, high)
    return bounds


def validate(matrix, domains, features) -> pd.DataFrame:
    """Count values that violate each feature's domain.

    ``matrix`` is (n_draws, n_features) in the order of ``features``.
    A value is invalid if it is non-finite, below the domain floor, above the
    domain ceiling, or non-integer for a count column. Violations are counted
    per class so a sampler that is bad only on ratios is distinguishable from
    one that is bad everywhere.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(features):
        raise ValueError('matrix shape does not match features')
    table = domains.set_index('feature')
    rows = []
    for j, name in enumerate(features):
        column = matrix[:, j]
        domain = table.loc[name, 'domain']
        nonfinite = int((~np.isfinite(column)).sum())
        finite = column[np.isfinite(column)]
        negative = int((finite < 0).sum())
        above_one = int((finite > 1).sum()) if domain in ('bounded_ratio', 'binary') else 0
        non_integer = (int((~np.isclose(finite, np.round(finite))).sum())
                       if domain == 'count' else 0)
        if domain == 'bounded_ratio':
            invalid = int(((finite < 0) | (finite > 1)).sum())
        elif domain == 'binary':
            invalid = int((~np.isin(finite, (0.0, 1.0))).sum())
        elif domain == 'count':
            invalid = int(((finite < 0) | ~np.isclose(finite, np.round(finite))).sum())
        else:
            invalid = int((finite < 0).sum())
        rows.append(dict(feature=name, group=name.split('.')[0], domain=domain,
                         n=len(column), n_nonfinite=nonfinite, n_negative=negative,
                         n_above_one=above_one, n_non_integer=non_integer,
                         n_invalid=invalid,
                         invalid_rate=invalid / len(column) if len(column) else np.nan))
    return pd.DataFrame(rows)


def discover_constraints(frame, features, min_hold=0.995, min_abs_spearman=0.5,
                         max_pairs=None) -> pd.DataFrame:
    """Find ordered pairs whose inequality holds on almost all real rows.

    Candidate pairs are those with a strong monotone relation on a stratified
    subsample (Spearman), which keeps the search affordable on 58k rows. Both
    directions are tested; the one that holds is recorded with its hold rate.
    A pair that holds in neither direction is not a hard constraint and is
    skipped rather than invented.
    """
    sample = frame[features]
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=0)
    corr = sample.corr(method='spearman').abs()
    pairs = [(a, b, float(corr.loc[a, b]))
             for a, b in itertools.combinations(features, 2)
             if np.isfinite(corr.loc[a, b]) and corr.loc[a, b] >= min_abs_spearman]
    pairs.sort(key=lambda item: item[2], reverse=True)
    if max_pairs:
        pairs = pairs[:max_pairs]

    values = {name: frame[name].to_numpy(dtype=np.float64) for name in features}
    rows = []
    for a, b, rho in pairs:
        holds_a_le_b = float(np.mean(values[a] <= values[b]))
        holds_a_ge_b = float(np.mean(values[a] >= values[b]))
        if holds_a_le_b >= min_hold:
            rows.append(dict(lhs=a, op='<=', rhs=b, abs_spearman=rho,
                             hold_rate=holds_a_le_b))
        elif holds_a_ge_b >= min_hold:
            rows.append(dict(lhs=a, op='>=', rhs=b, abs_spearman=rho,
                             hold_rate=holds_a_ge_b))
    return pd.DataFrame(rows)


def check_constraints(matrix, constraints, features) -> pd.DataFrame:
    """Violation rate of each discovered constraint on a value matrix."""
    position = {name: j for j, name in enumerate(features)}
    rows = []
    for row in constraints.itertuples():
        left = matrix[:, position[row.lhs]]
        right = matrix[:, position[row.rhs]]
        finite = np.isfinite(left) & np.isfinite(right)
        if not finite.any():
            rate = np.nan
        else:
            satisfied = (left[finite] <= right[finite] if row.op == '<='
                         else left[finite] >= right[finite])
            rate = float(1 - satisfied.mean())
        rows.append(dict(lhs=row.lhs, op=row.op, rhs=row.rhs,
                         real_hold_rate=row.hold_rate, synth_violation_rate=rate))
    return pd.DataFrame(rows)


def main():
    import argparse

    from common import ROOT
    from group_protocol import load_base_frame

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/sampler_v2/feature_domains.csv')
    args = parser.parse_args()
    frame, features = load_base_frame()
    domains = classify(frame, features)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    domains.to_csv(out, index=False)
    constraints = discover_constraints(frame, features)
    constraints.to_csv(out.parent / 'cross_constraints.csv', index=False)
    print(domains.domain.value_counts().to_string())
    print(f'\n{len(constraints)} cross-feature constraints discovered')
    print(constraints.head(15).to_string(index=False))
    print(f'\n-> {out.relative_to(ROOT)}')
    print(f'-> {(out.parent / "cross_constraints.csv").relative_to(ROOT)}')


if __name__ == '__main__':
    main()
