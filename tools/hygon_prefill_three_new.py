"""Run the three new prefill optimization probes in one ordered round."""

from __future__ import annotations

import pathlib
import subprocess
import sys

from hygon_prefill_audit import emit

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = (
    ("decode_single_tail", "tools/hygon_prefill_decode_tail.py"),
    ("dynamic_geometry", "tools/hygon_prefill_dynamic_geometry.py"),
    ("exact_layout", "tools/hygon_prefill_exact_layout.py"),
)


def main():
    emit("probe", arms=[name for name, _ in ARMS])
    failures = []
    for name, script in ARMS:
        emit("arm_start", arm=name, script=script)
        proc = subprocess.run(
            [sys.executable, "-u", script],
            cwd=ROOT,
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