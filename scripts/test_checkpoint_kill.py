"""Kill test: SIGTERM a checkpoint run mid-group, resume it, compare to reference.

Case C of run_equivalence.py hands the resume path a clean prefix. This case is
the one that actually happens: the process dies in the middle of a group, so the
runs.csv ends with rows from an incomplete group. The resumed output must still
equal an uninterrupted run.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / '.selftest_checkpoint'
PY = '/home/sealorl/PTMD/.venv/bin/python'
CHILD = str(Path(__file__).resolve().parent / '_checkpoint_child.py')
RUNS = 'permutation_runs.csv'
SUMMARY = 'permutation_summary.csv'
REPEATS, CONTROLS = '3', '60'


def launch(out, checkpoint, log):
    fh = open(log, 'w')
    proc = subprocess.Popen([PY, CHILD, str(out), '1' if checkpoint else '0',
                             CONTROLS, REPEATS], stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(Path(__file__).resolve().parent.parent))
    proc._fh = fh
    return proc


def wait(proc):
    rc = proc.wait()
    proc._fh.close()
    return rc


def main():
    shutil.rmtree(BASE, ignore_errors=True)
    BASE.mkdir(parents=True)
    ref, killed = BASE / 'F_ref', BASE / 'F_killed'
    print('reference run (no checkpoint)...', flush=True)
    rc = wait(launch(ref, False, BASE / 'f_ref.log'))
    if rc != 0:
        print('reference failed:', (BASE / 'f_ref.log').read_text()[-1500:])
        return 1

    print('checkpoint run, SIGTERM mid-group...', flush=True)
    proc = launch(killed, True, BASE / 'f_killed.log')
    deadline = time.time() + 900
    seen = 0
    while time.time() < deadline:
        rows = sum(1 for _ in open(killed / RUNS)) - 1 if (killed / RUNS).exists() else 0
        if rows >= 120:                     # more than one group, well inside another
            seen = rows
            break
        time.sleep(1)
    os.kill(proc.pid, signal.SIGTERM)
    rc = wait(proc)
    raw = (killed / RUNS).read_text()
    print(f'killed at {seen} rows rc={rc}; last char is newline: '
          f'{raw.endswith(chr(10))}; bytes={len(raw)}', flush=True)

    resumed = BASE / 'F_resumed'
    shutil.copytree(killed, resumed)
    print('resuming...', flush=True)
    rc = wait(launch(resumed, True, BASE / 'f_resume.log'))
    if rc != 0:
        print('resume failed:', (BASE / 'f_resume.log').read_text()[-1500:])
        return 1

    a, b = (ref / RUNS).read_text(), (resumed / RUNS).read_text()
    print('runs.csv  byte-identical:', a == b, flush=True)
    if a != b:
        import difflib
        for line in list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                              lineterm=''))[:14]:
            print('   ', line[:160], flush=True)
    sa, sb = (ref / SUMMARY).read_text(), (resumed / SUMMARY).read_text()
    print('summary.csv byte-identical:', sa == sb, flush=True)
    import pandas as pd
    ta, tb = pd.read_csv(ref / RUNS), pd.read_csv(resumed / RUNS)
    d = (ta.sort_values(['group', 'condition', 'control_index'])
         .delta_log_loss.to_numpy()
         - tb.sort_values(['group', 'condition', 'control_index'])
         .delta_log_loss.to_numpy())
    print(f'delta_log_loss max_abs_diff={float(abs(d).max()):.3e} '
          f'n_rows={len(ta)} vs {len(tb)}', flush=True)
    ok = a == b and sa == sb
    print('=== RESULT:', 'PASS' if ok else 'FAIL', '===', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
