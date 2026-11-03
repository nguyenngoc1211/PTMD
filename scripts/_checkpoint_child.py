"""One cell of run_group_importance, with OUT repointed to a scratch dir."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))
import run_group_importance as mod

mod.write_report = lambda args: None
mod.write_manifest = lambda args: None
out, ck, controls, repeats = Path(sys.argv[1]), sys.argv[2] == '1', int(sys.argv[3]), int(sys.argv[4])
mod.OUT = out
mod.run(argparse.Namespace(
    tasks=['family16'], models=['logreg'], seeds=[42], methods='permutation',
    permutation_repeats=repeats, controls=controls, split='source_aware',
    version='source_aware_v1', ablation_name='__no_ablation__', reuse_runs=False,
    checkpoint=ck, shapley_models=[], shapley_explain=400, shapley_background=64,
    out_name=out.name))
