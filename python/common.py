"""Shared paths, configuration and statistical helpers."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
CONFIG = json.loads((ROOT / "config.json").read_text())


def setup():
    import os
    os.chdir(ROOT)
    for directory in ("intermediate", "models", "tables", "figures"):
        (ROOT / "results" / directory).mkdir(parents=True, exist_ok=True)


def logloss_rows(y, probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-15, 1 - 1e-15)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def inference(differences, alpha=0.05):
    """One-sided paired t inference over rows, after averaging resample losses."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return dict(importance=np.nan, se=np.nan, statistic=np.nan,
                    p_value=np.nan, ci_lo=np.nan, n_valid=n)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    if se == 0:
        statistic = 0.0 if mean == 0 else float(np.sign(mean) * np.inf)
        p_value = 0.0 if mean > 0 else 1.0
    else:
        statistic = mean / se
        p_value = float(t.sf(statistic, n - 1))
    return dict(importance=mean, se=se, statistic=statistic, p_value=p_value,
                ci_lo=mean - float(t.ppf(1 - alpha, n - 1)) * se, n_valid=n)


def adjust_bh(p_values):
    """Benjamini–Hochberg adjustment; missing hypotheses count in family size."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        adjusted = p[order] * len(p) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return q
