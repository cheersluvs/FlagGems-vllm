"""Interleave the codex/hygon-prefill-audit override against the shipped one.

WHY. That branch carries eight commits past the shipped point (+355 lines, five
patched module copies and a 512 MB scratch cache): a dense counter carry, a
VEC2 route, a 512-bin route for short dense rows, scratch reuse keyed by
stream, and a small final network. Its own production report
(hygon_prefill_final_production_v1) reads 0.725 with the flag off and 0.770 on.

That report cannot rank anything, and its own numbers say so. It ran the four
benchmarks in sequence rather than interleaved, and

  - the SAME configuration measured twice gives geomean 0.725 and 0.601, with
    five of seven shapes 47-67% apart between the two runs;
  - the vLLM BASELINE column moves on a fixed shape: 12961x4100 reads
    0.6824 / 0.6864 / 1.8333 / 0.6820 ms (2.69x), 4100x1025 2.33x, 4x16385
    1.53x, 4x8193 1.46x. A C++ kernel does not get 2.7x slower on fixed input,
    so something else was on the card, and the "3.790" and "1.660" speedups in
    its third run are baseline artefacts.

So the +6% sits far inside that report's own noise. This measures the same
question the way the rest of this operator's work has been measured: every arm
in one process, interleaved, each arm's FASTEST round, on an idle HCU.

ARMS

    vllm         torch.ops._C.top_k_per_row_prefill -- measured like the rest,
                 so its round-to-round spread is visible instead of assumed
    ship         the shipped override (imported normally)
    audit        the audit branch's override, loaded from git
    audit-nosb   audit with the 512-bin short-row route disabled. That route is
                 gated to top_k == 512 and vocab <= 1536, i.e. to (4100,1025)
                 alone, and a shape-gated bin count was measured and declined
                 once already (tools/hygon_prefill_bins256.py: 256/512 bins win
                 only on that shape)
    audit-nosr   audit with FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE=0, because a
                 cache that hands back buffers instead of allocating them can
                 move a measurement on its own

Ratios are against vllm, so they are the benchmark's own SpeedUp and can be
compared with the acceptance table directly. 'vllm spread' is that arm's
max/min across rounds: if it is not ~1.00 the run is contended and nothing
below it should be quoted.

The audit copy's five module-copy names are renamed so they cannot land on the
shipped override's entries in sys.modules.

Needs the branch fetched:  git fetch origin codex/hygon-prefill-audit

    tools/vendor_probe.sh tools/hygon_prefill_audit_ab.py hygon_prefill_audit_ab
"""

import importlib.util
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
ROUNDS = 7
OVERRIDE_PATH = (
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)
AUDIT_REFS = ("origin/codex/hygon-prefill-audit", "0b05008")


def occupancy(tag):
    """What else is on the card. Their report was taken while something was."""
    import shutil

    for cmd in (["hy-smi"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe] + cmd[1:], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            out = repr(exc)
        print(f"--- card occupancy {tag}: {cmd[0]}")
        print("\n".join(out.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


def audit_module():
    """The audit branch's override, loaded as its own module."""
    src = ref = None
    for candidate in AUDIT_REFS:
        r = subprocess.run(
            ["git", "show", f"{candidate}:{OVERRIDE_PATH}"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout:
            src, ref = r.stdout, candidate
            break
    if src is None:
        raise SystemExit(
            "cannot read the audit override from any of "
            f"{AUDIT_REFS}.\n    run:  git fetch origin codex/hygon-prefill-audit"
        )
    print(f"### audit override read from {ref}, {len(src.splitlines())} lines")
    n = 0
    for name in (
        "_DENSE_NAME",
        "_CARRY_NAME",
        "_VEC2_NAME",
        "_VEC2_FINAL_NAME",
        "_SHORT_BINS_NAME",
        "_SPARSE_NAME",
    ):
        old = f'{name} = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_'
        assert src.count(old) == 1, f"{name} not found once in the audit file"
        src = src.replace(old, f'{name} = "flaggems_vllm.ops._auditab_hygon_', 1)
        n += 1
    assert n == 6
    f = pathlib.Path(tempfile.mkdtemp(prefix="auditab_")) / "topk_prefill_audit.py"
    f.write_text(src)
    mod_name = "flaggems_vllm.runtime.backend._hygon.fused._topk_prefill_auditab"
    spec = importlib.util.spec_from_file_location(mod_name, str(f))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def device_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters


def main():
    import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

    if not hasattr(torch.ops._C, "top_k_per_row_prefill"):
        raise SystemExit("this vLLM build exposes no top_k_per_row_prefill")
    ship = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    audit = audit_module()
    sb = getattr(audit, "_dense_short_bins", None)
    print(f"### audit short-bins route present: {sb is not None}")
    dev = "cuda"
    occupancy("before")
    print(
        f"\ndevice us, every arm in one process, interleaved over {ROUNDS} rounds,"
        "\neach arm's FASTEST round. Ratios are vllm/arm, i.e. the benchmark's"
        "\nown SpeedUp. Read 'vllm spread' first: it is the baseline arm's"
        "\nmax/min over the same rounds, and their report's was up to 2.69.\n"
    )
    names = ["vllm", "ship", "audit", "audit-nosb", "audit-nosr"]
    head = f"  {'shape':>12} {'k':>5}"
    for n in names:
        head += f"{n:>11}"
    for n in names[1:]:
        head += f"{'x' + n:>12}"
    head += f"{'vllm sprd':>10}{'ans':>6}"
    print(head)

    logs = {n: [] for n in names[1:]}
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        tbuf = (buf * 4).round() / 4
        tied = torch.as_strided(tbuf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

        def call(mod, src, nosb=False, nosr=False):
            keep = None
            if nosb:
                keep = getattr(mod, "_dense_short_bins", None)
                mod._dense_short_bins = None
            old = os.environ.get("FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE")
            if nosr:
                os.environ["FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE"] = "0"
            try:
                mod.top_k_per_row_prefill(
                    src, starts, ends, idx, rows, stride0, 1, top_k
                )
            finally:
                if nosb:
                    mod._dense_short_bins = keep
                if nosr:
                    if old is None:
                        os.environ.pop("FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE", None)
                    else:
                        os.environ["FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE"] = old

        runners = {
            "vllm": lambda s: torch.ops._C.top_k_per_row_prefill(
                s, starts, ends, idx, rows, stride0, 1, top_k
            ),
            "ship": lambda s: call(ship, s),
            "audit": lambda s: call(audit, s),
            "audit-nosb": lambda s: call(audit, s, nosb=True),
            "audit-nosr": lambda s: call(audit, s, nosr=True),
        }

        ok = True
        for name, run in runners.items():
            for label, src in (("normal", logits), ("tied", tied)):
                want = torch.topk(src, top_k, dim=1).values.sort(dim=1).values
                idx.fill_(-9)
                run(src)
                torch.cuda.synchronize()
                got = src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                good = torch.allclose(got, want) and bool((idx >= 0).all())
                if not good:
                    print(f"      ! {name} WRONG on {label}", flush=True)
                ok = ok and good

        per_round = {n: [] for n in names}
        for _ in range(ROUNDS):
            for name, run in runners.items():
                per_round[name].append(device_us(lambda run=run: run(logits)))
        mins = {n: min(per_round[n]) for n in names}
        spread = max(per_round["vllm"]) / min(per_round["vllm"])

        line = f"  {f'{rows}x{vocab}':>12} {top_k:>5}"
        for n in names:
            line += f"{mins[n]:>11.1f}"
        for n in names[1:]:
            r = mins["vllm"] / mins[n]
            logs[n].append(math.log(r))
            line += f"{r:>12.3f}"
        line += f"{spread:>10.2f}{'OK' if ok else 'WRONG':>6}"
        print(line, flush=True)

    print()
    for n in names[1:]:
        print(
            f"  geomean vs vllm, {n:>11}: {math.exp(sum(logs[n]) / len(logs[n])):.3f}"
        )
    occupancy("after")
    print(
        "\n  'vllm spread' well above 1.00 on any row means the card was busy"
        "\n  during that row and its whole line should be thrown away."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
