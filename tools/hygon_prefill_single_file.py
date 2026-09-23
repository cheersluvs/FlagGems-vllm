"""Validate the single-file prefill override (companions merged, final network cut).

The override is now one file: the carry-source builders moved in, and the
final-network copy and its two companions are gone, so the one-read route's
retry is built from the plain dense VEC2 copy. Locally the carry and VEC2
sources were checked byte-identical to before; what the card has to show:

PART 0 -- every route builds and loads on this box: each module copy present,
no "skipped" warning, the three companion modules absent from sys.modules.

PART 1 -- tools/hygon_prefill_dense_route_check.py part A: correctness of all
15 input x shape cases with the route ON and OFF, and its cost. The retry is
now the plain VEC2 copy, so the worst case (band, const: every row retries)
is where the cut shows. Previous ON times, with the network, for comparison:

    band   16383x4095 8628   12961x4100 6957   16380x5115 10747 us
    const  16383x4095 18185  12961x4100 14609  16380x5115 22563 us
    normal 16383x4095 700    12961x4100 595    16380x5115 778 us

PART 2 -- tools/hygon_prefill_before_after.py unchanged: both test suites, then
before/after interleaved twice. The PR table comes from here.

    tools/vendor_probe.sh tools/hygon_prefill_single_file.py hygon_prefill_single_file
"""

import os
import pathlib
import subprocess
import sys

PRE = r"""
import logging, sys
records = []
class H(logging.Handler):
    def emit(self, r):
        records.append(r.getMessage())
logging.getLogger().addHandler(H())
logging.getLogger().setLevel(logging.WARNING)
from importlib import import_module
ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
for name in ("_dense", "_sparse", "_dense_carry", "_dense_vec2", "_dense_short_bins",
             "_dense_retry"):
    m = getattr(ov, name, None)
    print(f"PRE {name:18s} {'ok ' + m.__name__ if m is not None else 'MISSING'}")
gone = [n for n in sys.modules if "_top_k_per_row_prefill_final" in n
        or "_top_k_per_row_prefill_carry_source" in n]
print("PRE companion modules loaded:", gone or "none")
print("PRE final-network attrs:", [a for a in dir(ov) if "final_network" in a.lower()
                                    or "vec2_final" in a.lower()] or "none")
print("PRE retry source has skip_ptr:", "skip_ptr" in open(ov._DENSE_RETRY_PATH).read())
print("PRE retry source references a network:",
      "_hygon_final_network" in open(ov._DENSE_RETRY_PATH).read())
print("PRE warnings:", [r for r in records if "hygon" in r.lower()] or "none")
"""


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
    ).stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)
    here = pathlib.Path(__file__).resolve().parent

    print("### part 0: module copies\n", flush=True)
    r = subprocess.run([sys.executable, "-c", PRE], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith("PRE"):
            print("  " + ln[4:], flush=True)
    if r.returncode:
        print("  ! import failed:")
        for ln in r.stderr.strip().splitlines()[-12:]:
            print(f"    | {ln[:200]}")
        raise SystemExit(1)

    print("\n### part 1: route check, part A\n", flush=True)
    sys.path.insert(0, str(here))
    import hygon_prefill_dense_route_check as rc

    res = {}
    for state in ("1", "0"):
        env = dict(os.environ)
        env["FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED"] = state
        r = subprocess.run(
            [sys.executable, "-c", rc.CHILD], capture_output=True, text=True, env=env
        )
        lines = [x[6:] for x in r.stdout.splitlines() if x.startswith("CHILD")]
        for ln in lines:
            print(f"  [{'on ' if state == '1' else 'off'}] {ln}", flush=True)
            shape, kind = ln.split()[:2]
            res[(state, shape, kind)] = ln
        if len(lines) != 15:
            print("  ! child incomplete:", flush=True)
            for ln in (r.stdout + r.stderr).strip().splitlines()[-10:]:
                print(f"    | {ln[:200]}", flush=True)
    print("\n  route ON vs OFF, do_bench us (x = OFF / ON)\n")
    for shape in ("16383x4095", "12961x4100", "16380x5115"):
        cells = []
        for kind in ("normal", "partial", "ties", "band", "const"):
            a, b = res.get(("1", shape, kind)), res.get(("0", shape, kind))
            if a and b:
                ua = float(a.split("us=")[1].split()[0])
                ub = float(b.split("us=")[1].split()[0])
                cells.append(f"{kind} {ub:.0f}->{ua:.0f} (x{ub / ua:.2f})")
        print(f"  {shape}: " + " | ".join(cells))
    wrong = [k for k, v in res.items() if "answer=ok" not in v]
    print(f"\n  wrong answers: {wrong or 'none'} ({len(res)} cases)", flush=True)

    print("\n### part 2: before / after\n", flush=True)
    subprocess.run([sys.executable, str(here / "hygon_prefill_before_after.py")])


if __name__ == "__main__":
    main()
