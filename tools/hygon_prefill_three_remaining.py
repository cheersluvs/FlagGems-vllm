"""Run the three remaining Hygon prefill directions in one fair round.

The GPU trials are intentionally sequential: concurrent kernels would make
the per-arm timings incomparable. Each child is isolated in its own Python
process, and a nonzero child status stops the round.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from hygon_prefill_audit import emit

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = (
    ("non_tle_radix_final", "tools/hygon_prefill_radix_final.py"),
    ("scratch_reuse", "tools/hygon_prefill_scratch_reuse.py"),
    ("candidate_workset", "tools/hygon_prefill_split_workset.py"),
)


def main():
    emit("probe", arms=[name for name, _ in ARMS])
    failures = []
    for name, script in ARMS:
        emit("arm_start", arm=name, script=script)
        proc = subprocess.run(
            [sys.executable, "-u", script],
            cwd=ROOT,
            env=None,
            timeout=4200,
        )
        emit("arm_exit", arm=name, code=proc.returncode)
        if proc.returncode:
            failures.append([name, proc.returncode])
            break
    emit("probe_complete", ok=not failures, failures=failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())