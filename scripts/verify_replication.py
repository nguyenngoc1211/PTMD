#!/usr/bin/env python
"""Verify a replication run against its artifacts and apply the pre-registered decision rule.

Checks that the artifact exists, that its manifest matches the locked protocol, that every
group really has the claimed number of matched-width controls, and recomputes BH q from
python/common.py:adjust_bh instead of trusting summary columns. Prints the rows needed to
fill a locked report table, then the mechanical verdict.

usage: .venv/bin/python scripts/verify_replication.py importance_v3_s43 --seed 43 --controls 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from common import adjust_bh  # noqa: E402

METRICS = ["log_loss", "accuracy", "macro_f1", "roc_auc", "pr_auc", "balanced_accuracy"]
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))


def warn(label: str, detail: str = "") -> None:
    print(f"  [WARN] {label}" + (f" — {detail}" if detail else ""))


def load_dir(out_name: str) -> tuple[Path, dict]:
    d = ROOT / "results" / "group_experiments" / out_name
    if not d.is_dir():
        sys.exit(f"artifact dir not on disk: {d}  (no numbers exist yet)")
    manifest = json.loads((d / "manifest.json").read_text())
    return d, manifest


def verify_permutation(d: Path, manifest: dict, seed: int, controls: int, models: list[str]) -> pd.DataFrame:
    runs_f, sum_f = d / "permutation_runs.csv", d / "permutation_summary.csv"
    check("permutation_runs.csv present", runs_f.exists())
    check("permutation_summary.csv present", sum_f.exists())
    cfg = manifest.get("config", {})
    check("manifest seeds", manifest.get("seeds") == [seed], f"got {manifest.get('seeds')}")
    check("manifest models", manifest.get("models") == models, f"got {manifest.get('models')}")
    check("manifest permutation_controls", manifest.get("permutation_controls") == controls,
          f"got {manifest.get('permutation_controls')}")
    check("runs were freshly computed (not reused)", manifest.get("reused_runs") is False,
          f"got {manifest.get('reused_runs')}")
    check("split locked", cfg.get("split_version", manifest.get("split_version")) in (None, "source_aware_v1"),
          f"got {cfg.get('split_version', manifest.get('split_version'))}")

    runs = pd.read_csv(runs_f)
    n_group = runs[runs["condition"] == "permute_group"]
    n_ctrl = runs[runs["condition"] == "permute_random"]
    check("one observed row per group", len(n_group) == n_group["group"].nunique(),
          f"{len(n_group)} rows / {n_group['group'].nunique()} groups")
    per_group = n_ctrl.groupby("group").size()
    check(f"every group has exactly {controls} matched-width controls",
          bool((per_group == controls).all()),
          f"min {per_group.min()} max {per_group.max()} over {len(per_group)} groups")
    if "n_features" in n_ctrl.columns and "n_features" in n_group.columns:
        widths = n_group.set_index("group")["n_features"]
        match = n_ctrl.assign(w=n_ctrl["group"].map(widths)).pipe(
            lambda x: (x["n_features"] == x["w"]).all())
        check("control width matched observed group width", bool(match))
    else:
        warn("control width not recorded in permutation_runs.csv",
             "matched-by-construction in python/run_group_importance.py (controls drawn from the group complement)")

    summary = pd.read_csv(sum_f)
    summary = summary.sort_values("group").reset_index(drop=True)
    check("9 groups in summary", len(summary) == 9, f"got {len(summary)}")
    recomputed = adjust_bh(summary["p_empirical"].to_numpy())
    if "q_BH" in summary:
        drift = float((recomputed - summary["q_BH"].to_numpy()).max())
        check("published q_BH matches adjust_bh", abs(drift) < 1e-12, f"max drift {drift:.3e}")
    else:
        warn("summary stores p only", "q_BH recomputed here from adjust_bh over the same 9 groups")
    summary["q_BH"] = recomputed
    return summary


def verify_ablation(d: Path, manifest: dict, seed: int, controls: int, models: list[str]) -> pd.DataFrame:
    runs_f, sum_f = d / "runs.csv", d / "summary.csv"
    check("runs.csv present", runs_f.exists())
    check("summary.csv present", sum_f.exists())
    check("manifest seeds", manifest.get("seeds") == [seed], f"got {manifest.get('seeds')}")
    check("manifest models", manifest.get("models") == models, f"got {manifest.get('models')}")
    check("manifest controls_per_group", manifest.get("controls_per_group") == controls,
          f"got {manifest.get('controls_per_group')}")
    check("tuning budget is none", str(manifest.get("tuning_budget", "")).startswith("none"),
          str(manifest.get("tuning_budget")))

    runs = pd.read_csv(runs_f)
    ctrl = runs[runs["condition"] == "random_drop"]
    per_group = ctrl.groupby("group").size()
    check(f"every group has exactly {controls} random_drop controls",
          bool((per_group == controls).all()),
          f"min {per_group.min()} max {per_group.max()} over {len(per_group)} groups")
    check("control indices are 0..N-1 with no reuse",
          bool((ctrl.groupby("group")["control_index"].nunique() == controls).all()),
          f"unique indices {sorted(ctrl.groupby('group')['control_index'].nunique().unique())}")

    summary = pd.read_csv(sum_f).sort_values("group").reset_index(drop=True)
    check("9 groups in summary", len(summary) == 9, f"got {len(summary)}")
    live = [m for m in METRICS if f"{m}_p_empirical" in summary and summary[f"{m}_p_empirical"].notna().all()]
    check("metric set used for BH", len(live) in (6, 7), f"live metrics: {live}")
    for m in live:
        summary[f"q_{m}"] = adjust_bh(summary[f"{m}_p_empirical"].to_numpy())
    summary.attrs["live_metrics"] = live
    return summary


def report_and_verdict(summary: pd.DataFrame, kind: str, controls: int, label: str) -> None:
    floor = 1.0 / (controls + 1)
    print(f"\n{label} — delta log-loss / null / p / q_BH (BH over 9 groups, floor p = {floor:.6f}):")
    if kind == "permutation":
        cols = ["group", "delta_log_loss_mean", "null_mean", "null_sd", "p_empirical", "q_BH", "rank"]
        print(summary[cols].to_string(index=False))
        hits = {g: bool(abs(r.p_empirical - floor) < 1e-9 and r.q_BH < 0.05) for g, r in zip(summary.group, summary.itertuples())}
    else:
        live = summary.attrs["live_metrics"]
        show = ["group"] + [f"{m}_delta_mean" for m in live] + [f"{m}_p_empirical" for m in live] + [f"q_{m}" for m in live]
        print(summary[show].round(5).to_string(index=False))
        hits = {}
        for r in summary.itertuples():
            q_main = getattr(r, f"q_{live[0]}")
            p_main = getattr(r, f"{live[0]}_p_empirical")
            n_pass = sum(1 for m in live if getattr(r, f"q_{m}") < 0.05)
            hits[r.group] = (bool(abs(p_main - floor) < 1e-9 and q_main < 0.05), n_pass, len(live))
    print(f"\n{label} — decision under the pre-registered rule:")
    for g in ["handles", "ldrmodules"]:
        r = summary[summary.group == g]
        if r.empty:
            print(f"  {g}: MISSING from artifact")
            continue
        v = hits[g]
        if kind == "permutation":
            print(f"  {g}: confirmed={v}")
        else:
            print(f"  {g}: confirmed={v[0]}  metrics q<0.05 = {v[1]}/{v[2]}")
    def passes(g: str) -> bool:
        v = hits.get(g)
        if v is None:
            return False
        return bool(v) if kind == "permutation" else bool(v[0])

    others = [g for g in summary.group if g not in ("handles", "ldrmodules")]
    passed = [g for g in others if passes(g)]
    print(f"  any other group passing: {passed or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_name")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--controls", type=int, required=True)
    ap.add_argument("--kind", choices=["permutation", "ablation"])
    ap.add_argument("--models", default="xgboost", help="expected manifest models, comma-separated")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    d, manifest = load_dir(args.out_name)
    kind = args.kind or ("permutation" if (d / "permutation_summary.csv").exists() else "ablation")
    print(f"== {args.out_name} ({kind}, seed {args.seed}, {args.controls} controls) ==")
    summary = verify_permutation(d, manifest, args.seed, args.controls, models) if kind == "permutation" \
        else verify_ablation(d, manifest, args.seed, args.controls, models)
    report_and_verdict(summary, kind, args.controls, f"{kind}:{args.out_name}")
    print("\nGATE:", "PASS" if ok else "FAIL — do not publish these numbers until resolved")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
