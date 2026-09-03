"""Run each DSA/UB case in its own process, so one device fault cannot hide the next.

A timed-out vector core leaves the device in an error state on this card, and
every subsequent launch in the same process then reports "Failed to submit
kernel task" -- which reads like a second failure but is only contamination.
One process per case is the only way the results mean anything.

Output goes to files, never pipes: a Triton compiler grandchild holds inherited
pipes open, so capture_output loses the very stack dump a timeout produces.
Each child gets its own session so a timeout can kill the whole group.
"""

import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Outside the repo by default.  Writing per-case logs into a tracked reports/
# directory dirtied the work tree on every run, and the next `git pull` then
# refused with "please commit or stash them" -- three times before this moved.
LOGDIR = os.environ.get(
    "DSA_LOGDIR", os.path.join(os.path.expanduser("~"), "ft-reports", "dsa_cases"))
TIMEOUT = int(os.environ.get("DSA_CASE_TIMEOUT", "420"))
# Which per-case script to fork.  The defect matrix reuses this driver, since
# isolation matters there for the same reason: one fault poisons the process.
CASE_SCRIPT = os.environ.get(
    "DSA_CASE_SCRIPT", os.path.join(ROOT, "tools", "ascend_dsa_case.py"))

CASES = [
    "alloc_only",   # does allocation alone execute
    "to_tensor",    # can UB be read as a tensor           (hung in probe 2)
    "copy",         # GM -> UB -> GM staging, the Ascend idiom
    "ptr_store",    # is there any pointer to a UB buffer
    "atomic",       # can a histogram scatter reach UB at all
    "atomic2",      # same question, with the shape bug removed
    "capcopy2048",  # 8 KB   -- the histogram's own size
    "capcopy8192",  # 32 KB  -- near the measured ~36 KB ceiling
    "capcopy16384", # 64 KB  -- past it, to find the real wall
    "hist",         # is tl.histogram lowered at all
    "hist_accum",   # accumulate a row, flush bins once: the shape we would ship
]


def plog_tail(n=40):
    """The first failure is the one whose plog is worth reading."""
    roots = [os.path.expanduser("~/ascend/log"), "/root/ascend/log",
             os.path.expanduser("~/var/log/npu"), "/var/log/npu"]
    newest, newest_t = None, 0
    for r in roots:
        for dirpath, _d, names in os.walk(r) if os.path.isdir(r) else ():
            for nm in names:
                if not nm.startswith("plog"):
                    continue
                p = os.path.join(dirpath, nm)
                try:
                    t = os.path.getmtime(p)
                except OSError:
                    continue
                if t > newest_t:
                    newest, newest_t = p, t
    if newest is None:
        return "  (no plog found)"
    with open(newest, errors="replace") as fh:
        lines = [l.rstrip() for l in fh if "ERROR" in l or "EZ9999" in l]
    body = "\n".join(f"      {l}" for l in lines[-n:]) or "      (no ERROR lines)"
    return f"  newest plog {newest}\n{body}"


if len(sys.argv) > 1:                     # run only the named cases
    CASES = sys.argv[1:]

os.makedirs(LOGDIR, exist_ok=True)
first_failure_reported = False

for case in CASES:
    log = os.path.join(LOGDIR, f"{case}.log")
    print(f"\n{'=' * 72}\n=== {case}\n{'=' * 72}")
    sys.stdout.flush()
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.Popen(
            [sys.executable, CASE_SCRIPT, case],
            # Blocking launches make a fault attributable, but they also
            # serialise the very thing a timing case measures.
            stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
            env=dict(os.environ) if case.startswith("time_")
            else dict(os.environ, ASCEND_LAUNCH_BLOCKING="1"),
            start_new_session=True,
        )
        try:
            rc = p.wait(timeout=TIMEOUT)
            verdict = f"exit {rc}"
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
            verdict = f"TIMED OUT after {TIMEOUT}s (killed)"
    dt = time.time() - t0
    with open(log, errors="replace") as fh:
        body = fh.read()
    print(body.rstrip())
    print(f"--- {case}: {verdict}, {dt:.1f}s")
    failed = ("FAILED" in body) or verdict.startswith("TIMED")
    if failed and not first_failure_reported:
        first_failure_reported = True
        print("--- plog after the FIRST failure:")
        print(plog_tail())
    sys.stdout.flush()

print("\ndone.")
