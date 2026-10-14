# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Run each reproduction/control in a fresh process; retain every exit status."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root = args.output or Path(tempfile.mkdtemp(prefix="flagtree-cp-async."))
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=args.output is None)
    print("Reports:", root, flush=True)
    here = Path(__file__).resolve().parent
    jobs = [
        ("replay-original", "replay.py", ["--variant", "original"]),
        ("replay-no-hint", "replay.py", ["--variant", "no-hint"]),
        ("replay-fraction-half", "replay.py", ["--variant", "fraction-half"]),
        ("source-original", "minimal.py", []),
        ("source-no-policy", "minimal.py", ["--policy", "none"]),
        ("source-stage1", "minimal.py", ["--stages", "1"]),
    ]
    summary = []
    for name, script, options in jobs:
        folder = root / name
        folder.mkdir(exist_ok=True)
        # Isolate disk caches and CUDA contexts, including after an illegal instruction.
        env = dict(os.environ, TRITON_CACHE_DIR=str(folder / "cache"))
        command = [
            sys.executable,
            str(here / script),
            *options,
            "--output",
            str(folder),
        ]
        row = {"job": name, "command": command}
        with (folder / "run.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                row["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                row["timeout_seconds"] = args.timeout
        summary.append(row)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    # Failures are diagnostic evidence, never silently convert them to success.
    return int(any(row.get("returncode") != 0 for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
