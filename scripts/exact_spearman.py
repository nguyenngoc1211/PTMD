"""Exact two-sided permutation p for Spearman's rho, next to scipy's asymptotic p.

Spearman reported by scipy.stats.spearmanr uses a t-approximation (df = n-2).
For n = 9 the exact null is reachable by enumerating all n! rank orderings, so
this script reports both and lets the write-up state which one is quoted.

Value columns must be scores where HIGHER = more important. Do not feed the
`rank` column of `permutation_summary.csv`: it is ascending (rank 1 = smallest
delta), so it flips the sign of rho and inverts `top1_match`.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from functools import lru_cache

import pandas as pd
from scipy.stats import spearmanr


@lru_cache(maxsize=None)
def null_d2_distribution(n: int) -> tuple[Counter, int]:
    """Exact null distribution of sum((i - pi[i])**2) over all n! orderings."""
    counts: Counter = Counter()
    identity = range(n)
    for perm in itertools.permutations(identity):
        d2 = sum((a - b) * (a - b) for a, b in zip(identity, perm))
        counts[d2] += 1
    return counts, n * (n * n - 1)


def rho_from_d2(d2: int, denom: int) -> float:
    return 1.0 - 6.0 * d2 / denom


def exact_two_sided_p(n: int, rho_obs: float) -> tuple[float, int]:
    counts, denom = null_d2_distribution(n)
    total = sum(counts.values())
    target = abs(rho_obs)
    hit = sum(c for d2, c in counts.items() if abs(rho_from_d2(d2, denom)) >= target - 1e-12)
    return hit / total, total


def ranks(series: pd.Series) -> pd.Series:
    return series.rank(method="average")


def compare(name_a: str, a: pd.Series, name_b: str, b: pd.Series) -> dict:
    common = a.index.intersection(b.index)
    ra, rb = ranks(a.loc[common]), ranks(b.loc[common])
    res = spearmanr(ra, rb)
    n = len(common)
    exact, total = exact_two_sided_p(n, res.statistic)
    rank_a = ra.rank(ascending=False, method="min").astype(int)
    rank_b = rb.rank(ascending=False, method="min").astype(int)
    return {
        "pair": f"{name_a} ~ {name_b}",
        "n_groups": n,
        "rho": round(float(res.statistic), 4),
        "p_scipy_asymptotic": round(float(res.pvalue), 6),
        "p_exact_all_orderings": round(exact, 6),
        "n_orderings": total,
        "top1_match": rank_a.idxmin() == rank_b.idxmin(),
        "max_rank_shift": int((rank_a - rank_b).abs().max()),
        "groups_within_1_rank": int(((rank_a - rank_b).abs() <= 1).sum()),
    }


def load_scores(path: str, group_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.set_index(group_col)[value_col]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_a")
    ap.add_argument("col_a")
    ap.add_argument("csv_b")
    ap.add_argument("col_b")
    ap.add_argument("--group-col", default="group")
    args = ap.parse_args()

    label_a = f"{args.col_a}@{args.csv_a.split('/')[-2]}"
    label_b = f"{args.col_b}@{args.csv_b.split('/')[-2]}"
    a = load_scores(args.csv_a, args.group_col, args.col_a)
    b = load_scores(args.csv_b, args.group_col, args.col_b)
    row = compare(label_a, a, label_b, b)
    for key, value in row.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
