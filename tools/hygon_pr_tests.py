"""Run the PR's two test files, exactly as the PR carries them, on this card.

tests/test_top_k_per_row_{prefill,decode}.py on this branch are byte-copies of
the PR branch's. The prefill narrow-band test now also carries the two cases
first written against the Moore Threads override, (1, 129280, 1024) and
(64, 129280, 1024), which never ran on Hygon; at width 0.02 and 0.2 the second
sends every one of its 64 rows through the sampled path's exact redo.

Prints every narrow-band case verbosely, then each suite's summary.

    tools/vendor_probe.sh tools/hygon_pr_tests.py hygon_pr_tests
"""

import subprocess
import sys


def run(args):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *args], capture_output=True, text=True
    )
    return r.returncode, r.stdout + r.stderr


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tests"],
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise SystemExit("the tree is modified:\n" + dirty)

    print("### narrow band, verbose\n", flush=True)
    rc, out = run(
        [
            "-v",
            "-p",
            "no:cacheprovider",
            "tests/test_top_k_per_row_prefill.py",
            "-k",
            "narrow_band",
        ]
    )
    for ln in out.splitlines():
        if "narrow_band" in ln or "passed" in ln or "failed" in ln or "Error" in ln:
            print("  " + ln, flush=True)

    for suite in ("prefill", "decode"):
        rc, out = run(
            ["-q", "-p", "no:cacheprovider", f"tests/test_top_k_per_row_{suite}.py"]
        )
        tail = [ln for ln in out.splitlines() if ln.strip()][-1]
        print(f"\n### {suite}: {tail}  (exit {rc})", flush=True)
        if rc:
            for ln in out.splitlines()[-40:]:
                print("  | " + ln[:200])


if __name__ == "__main__":
    main()
