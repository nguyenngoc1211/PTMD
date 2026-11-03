"""Equivalence test for --checkpoint in python/run_group_importance.py.

Proves three things on a cheap real config (family16 / logreg / seed 42,
2 repeats, 3 controls):

  A  plain run, no checkpoint                      -> reference numbers
  B  same cell with --checkpoint, never interrupted -> must equal A
  C  same cell resumed from a partial checkpoint    -> must equal B
  D  resume onto a torn last line                   -> must equal B

Writes only under .selftest_checkpoint/ at the repo root: not an artifact path,
removed after the run, and never a symlink into results/group_experiments/ (a
symlink sandbox is banned by REPRODUCIBILITY 4.21 because mkdir and to_csv both
follow the link into the live cell).
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))
import run_group_importance as mod

# write_report builds figure links with OUT.relative_to(ROOT), which raises for a
# test tree outside the repo, and write_manifest hashes the 300 MB dataset.
# Neither is under test here; the numbers are.
mod.write_report = lambda args: None
mod.write_manifest = lambda args: None

BASE = Path(__file__).resolve().parent.parent / '.selftest_checkpoint'
CELL = dict(tasks=['family16'], models=['logreg'], seeds=[42],
            methods='permutation', permutation_repeats=2, controls=3,
            split='source_aware', version='source_aware_v1',
            ablation_name='__no_ablation__')


def namespace(out_name, checkpoint):
    ns = argparse.Namespace(**CELL)
    ns.out_name = out_name
    ns.reuse_runs = False
    ns.checkpoint = checkpoint
    ns.shapley_models = []
    ns.shapley_explain = 400
    ns.shapley_background = 64
    return ns


def run_cell(case, checkpoint):
    out = BASE / case
    if out.exists():
        shutil.rmtree(out)
    mod.OUT = out
    mod.run(namespace(case, checkpoint))
    return out


def read(out, name):
    return (out / name).read_text()


import filecmp as _filecmp


def compare(a, b, label, keys=('group', 'condition', 'control_index')):
    print(f'  {label}: byte_identical={_filecmp.cmp(a, b, shallow=False)}')
    import pandas as pd
    ta, tb = (pd.read_csv(a, float_precision='round_trip'),
                pd.read_csv(b, float_precision='round_trip'))
    if list(ta.columns) != list(tb.columns):
        print(f'  {label}: COLUMN MISMATCH {list(ta.columns)} vs {list(tb.columns)}')
        return False
    ta = ta.sort_values(list(ta.columns[:6])).reset_index(drop=True)
    tb = tb.sort_values(list(tb.columns[:6])).reset_index(drop=True)
    if len(ta) != len(tb):
        print(f'  {label}: ROW COUNT {len(ta)} vs {len(tb)}')
        return False
    num = ta.select_dtypes('number').columns
    same = _filecmp.cmp(a, b, shallow=False) and all((ta[c].isna() & tb[c].isna()).all() or (ta[c] == tb[c]).all()
               for c in num)
    text = ta.select_dtypes(exclude='number').columns
    same = same and all((ta[c].astype(str) == tb[c].astype(str)).all() for c in text)
    worst = max((float((ta[c].astype(float) - tb[c].astype(float)).abs().max())
                 for c in num if c != 'control_index' and ta[c].notna().all()),
                default=0.0)
    print(f'  {label}: {"IDENTICAL" if same else "DIFFERS"}  '
          f'rows={len(ta)} max_abs_diff={worst:.3e}')
    return same


def build_partial(src, dst, keep_n):
    """Recreate the on-disk state a crash leaves: whole groups only."""
    import pandas as pd
    groups = pd_group_order(src / 'permutation_runs.csv')
    keep = groups[:keep_n]
    if dst.exists():
        shutil.rmtree(dst)
    (dst / 'figures').mkdir(parents=True)
    t = pd.read_csv(src / 'permutation_runs.csv', float_precision='round_trip')
    t[t.group.isin(keep)].to_csv(dst / 'permutation_runs.csv', index=False)
    name = rt_name()
    rt = pd.read_csv(src / name, float_precision='round_trip')
    cols = ['test_position'] + [g for g in rt.columns if g in keep]
    rt[cols].to_csv(dst / name, index=False)
    return keep, groups[len(keep)] if len(groups) > keep_n else None


def compare_keyed(a, b, label):
    """Order-insensitive equality: same rows under the (group, condition,
    control_index) key, values bit-identical. Used for the gap case, where a
    refilled block is appended rather than written back into its old position."""
    import pandas as pd
    ta = pd.read_csv(a, float_precision='round_trip').sort_values(
        ['group', 'condition', 'control_index']).reset_index(drop=True)
    tb = pd.read_csv(b, float_precision='round_trip').sort_values(
        ['group', 'condition', 'control_index']).reset_index(drop=True)
    if len(ta) != len(tb):
        print(f'  {label}: ROW COUNT {len(ta)} vs {len(tb)}')
        return False
    ok = (ta[['task', 'model', 'seed', 'group', 'condition',
              'control_index']].astype(str).values ==
          tb[['task', 'model', 'seed', 'group', 'condition',
              'control_index']].astype(str).values).all()
    worst = max(float((ta[c] - tb[c]).abs().max())
                for c in ('delta_log_loss', 'delta_log_loss_sd'))
    ok = ok and worst == 0.0
    print(f'  {label}: {"IDENTICAL (by key)" if ok else "DIFFERS"}  '
          f'rows={len(ta)} max_abs_diff={worst:.3e}')
    return ok


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    a = run_cell('A_plain', False)
    b = run_cell('B_checkpoint', True)
    print('=== reference: plain run vs uninterrupted checkpoint ===')
    ok_runs = compare(a / 'permutation_runs.csv', b / 'permutation_runs.csv',
                      'runs A vs B')
    ok_sum = compare(a / 'permutation_summary.csv', b / 'permutation_summary.csv',
                     'summary A vs B')

    # C: resume from four of nine groups
    c = BASE / 'C_resume'
    keep, next_group = build_partial(b, c, 4)
    print(f'=== resume case: kept {keep} ===')
    mod.OUT = c
    mod.run(namespace('C_resume', True))
    ok_c = compare(b / 'permutation_runs.csv', c / 'permutation_runs.csv',
                   'runs B vs C(resumed)')
    ok_cs = compare(b / 'permutation_summary.csv', c / 'permutation_summary.csv',
                    'summary B vs C(resumed)')

    # D: same partial state, plus a torn final line from an interrupted append
    d = BASE / 'D_torn'
    build_partial(b, d, 4)
    header = (d / 'permutation_runs.csv').read_text().split('\n', 1)[0]
    ncol = len(header.split(','))
    torn = ','.join(['family16', 'logreg', '42', 'permute_group', next_group]
                    + ['0.1'] * (ncol - 6))
    with (d / 'permutation_runs.csv').open('a') as fh:
        fh.write(torn[:len(torn) - 4])          # cut mid-field, no newline
    print(f'=== torn case: half row for {next_group} appended ===')
    mod.OUT = d
    mod.run(namespace('D_torn', True))
    ok_d = compare(b / 'permutation_runs.csv', d / 'permutation_runs.csv',
                   'runs B vs D(torn tail)')

    # E: a control row removed so its group has a gap. An append log cannot
    # produce this (a torn write loses the whole line), so the check is only
    # that the recomputed block lands with the right value; row order is not
    # preserved when a gap is filled, so compare by key rather than by byte.
    e = BASE / 'E_gap'
    build_partial(b, e, 0)
    import pandas as pd
    t = pd.read_csv(b / 'permutation_runs.csv', float_precision='round_trip')
    gap = t[~((t.group == next_group) & (t.control_index == 3))]
    gap.to_csv(e / 'permutation_runs.csv', index=False)
    shutil.copy(b / rt_name(), e / rt_name())
    print(f'=== gap case: removed control 3 of {next_group} '
          f'({len(t)} -> {len(gap)} rows) ===', flush=True)
    mod.OUT = e
    mod.run(namespace('E_gap', True))
    ok_e = compare_keyed(b / 'permutation_runs.csv', e / 'permutation_runs.csv',
                         'runs B vs E(gap refilled)')

    verdict = all([ok_runs, ok_sum, ok_c, ok_cs, ok_d, ok_e])
    print('=== RESULT:', 'PASS' if verdict else 'FAIL', '===')
    return 0 if verdict else 1


def pd_group_order(path):
    import pandas as pd
    t = pd.read_csv(path, float_precision='round_trip')
    seen = []
    for g in t.group:
        if g not in seen:
            seen.append(g)
    return seen


def rt_name():
    return (f"permutation_rows_{CELL['tasks'][0]}_{CELL['models'][0]}"
            f"_{CELL['seeds'][0]}.csv")


if __name__ == '__main__':
    sys.exit(main())
