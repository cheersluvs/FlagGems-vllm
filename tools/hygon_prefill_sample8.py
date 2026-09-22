"""Open the sampled prefill path's gate on the DENSE shapes.

WHY. The sampled prefill path (reverted in e1a25ee) already does exactly what
"sample = vocab/8" asks: `SSTRIDE = 8` and `_s_hist` reads every 8th TILE, so
it reads exactly 1/8 of the row. Its `_s_finish` already ranks the candidates
with four 8-bit radix rounds over the full 32-bit key, not by counting. Both
pieces of the idea are in that file and were measured at most 1.057.

But its gate is `vocab >= 64 * top_k`, which admits ONE of the seven benchmark
shapes -- and `_can_sample` carries a SECOND, independent one, `num_rows *
next_pow2(top_k * CAP_MULT) <= 1 << 24`, which excludes the three big dense
shapes by itself (16383 x 2048 = 33.5M against 16.8M). Round 1 of this probe
opened only the first and tripped its own routing assertion on shape 1; both
are opened here. The surviving shape either way is (64,129280), whose measurements are unusable
here: the same shipped call read 310.7 and 232.3 us eight minutes apart, and it
has already answered three questions contradictorily. **So the idea was only
ever measured where it cannot be measured.**

The gate came from this table in that file:

    shape            ratio   saves   ranking   net
    (64,129280)        63     165       40     1.73x
    (16380,5115)        5    1030     1000     ~1.0
    (16383,4095)        4    1090     1000     ~1.0

Two of its three columns have since been corrected ON THE OPERATOR:

- `ranking` assumed 0.06 us per row at 1024 candidates, from a replica
  (tools/hygon_merge_kernel.py). tools/hygon_prefill_radix_final.py measured a
  radix ranker at a flat **12.8 ns per row** and the counting ranker at
  ~0.2 ns per candidate per row. 1000 us over 16383 rows is 61 ns/row -- 4.8x
  the measured radix cost.
- `saves` used "the histogram pass is 82% of the operator", which was T(top_k=1)
  run at two row_ends and so counted BOTH passes. The corrected share is ~45%.

Redoing (16383,4095) with those: saves 7/8 x 0.45 x 1338 = 527 us, ranking goes
88 -> 210 us, net -405 us on 1338 -> ratio 0.863 -> ~1.23. That is a model, and
models on this operator have collapsed twice (predicted 1.73 measured 1.057;
predicted "narrowing is unblocked" measured 0.830). This measures it.

ARMS. All but `ship` live in ONE copy of the reverted file, so the A/B is
internal and the missing one-scan patch cancels out:

    off      SAMPLED_MIN_VOCAB_PER_TOPK = 0 -- that file's own slotscan +
             geometry path. THE CONTROL.
    s8/1.5   gate open on every shape, SSTRIDE 8, TARGET_MULT 1.5
    s8/2.0   gate open, SSTRIDE 8, TARGET_MULT 2.0
    s16/1.5  gate open, SSTRIDE 16, TARGET_MULT 1.5
    ship     today's production override, for absolute position only -- it has
             the one-scan patch the others lack, so compare it to nothing but
             itself.

READ 'outside' FIRST. It is the share of rows whose collected count lands
outside [top_k, CAP], each of which pays a full redo inside `_s_finish`. The
decode sweep (reports/hygon_decode_sample_sweep.txt) showed 4.2% of rows
outside costing 7.4x. Prefill gives each row its own program, so a straggler
holds up one program rather than the whole kernel -- the risk should be a
notch lower, but it is a measurement, not an argument.

(4100,1025) is expected to lose: vocab/8 is 128 samples to estimate rank 512
of 1025, at 50% density. It is also the only shape currently above parity, so
it is here to place the gate, not to be improved.

    tools/vendor_probe.sh tools/hygon_prefill_sample8.py hygon_prefill_sample8
"""

import ast
import importlib.util
import math
import pathlib
import subprocess
import sys
import tempfile
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

SHAPES = [
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (64, 129280, 1024, 129280),
]
ROUNDS = 5
REVERT_COMMIT = "e1a25ee^"
OVERRIDE_PATH = (
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)

# (tag, sampled_ratio, sstride, target_mult); ratio 0 disables sampling
ARMS = [
    ("off", 0, 8, 1.5),
    ("s8/1.5", 1, 8, 1.5),
    ("s8/2.0", 1, 8, 2.0),
    ("s16/1.5", 1, 16, 1.5),
]


def occupancy(tag):
    """What else is on the card; this box is shared."""
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


def _assert_no_range_kind_clash(src):
    """No function may name a tl.range and a tl.static_range loop variable the
    same: Triton merges a loop variable's type across the function and refuses
    the second kind. This is what stopped the sampled path's last round."""
    tree = ast.parse(src)
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        kinds = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                call = node.iter
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    kinds.setdefault(node.target.id, set()).add(call.func.attr)
        clash = {k: sorted(v) for k, v in kinds.items() if len(v) > 1}
        assert not clash, f"{fn.name} mixes range kinds on {clash}"


def sampled_module():
    """The reverted override, loaded as its own module.

    Its `_DENSE_NAME` is renamed so its generic copy does not land on the
    production override's entry in sys.modules; everything else is the file as
    it stood at e1a25ee^, byte for byte.
    """
    src = subprocess.run(
        ["git", "show", f"{REVERT_COMMIT}:{OVERRIDE_PATH}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    subs = [
        # keep this copy's generic module off the production override's entry
        (
            '_DENSE_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_dense"',
            '_DENSE_NAME = "flaggems_vllm.ops._topk_prefill_sample8_dense"',
        ),
        # `_can_sample` has a SECOND gate, and it excludes exactly the three big
        # dense shapes on its own: 16383 rows x 2048 candidates is 33.5M
        # elements against a hardcoded 1 << 24 = 16.8M. Round 1 of this probe
        # opened only the ratio gate and tripped its own routing assertion.
        (
            "        and num_rows * triton.next_power_of_2(top_k * CAP_MULT) <= (1 << 24)",
            "        and num_rows * triton.next_power_of_2(top_k * CAP_MULT)"
            " <= _MAX_CAND_ELEMS",
        ),
        (
            "CAP_MULT = 4  # candidate buffer; the acceptance window is [top_k, CAP]",
            "CAP_MULT = 4  # candidate buffer; the acceptance window is [top_k, CAP]\n"
            "_MAX_CAND_ELEMS = 1 << 24  # made tunable by tools/hygon_prefill_sample8",
        ),
        # `_s_finish` never compiled: it names both a tl.range and a
        # tl.static_range loop variable `t`, and Triton merges a loop
        # variable's type across the whole function ("initial value for `t` is
        # of type int32[], but the then block redefines it as constexpr[0]").
        # That is why the path's last round was reverted without a number. The
        # two static_range loops are renamed here; nothing else changes, and an
        # AST check below proves no other function mixes the two kinds.
        (
            "        for t in tl.static_range(NB // BLOCK):\n"
            "            tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))\n",
            "        for ct in tl.static_range(NB // BLOCK):\n"
            "            tl.store(base + ct * BLOCK + lane, tl.zeros([BLOCK], tl.int32))\n",
        ),
        (
            "        for t in tl.static_range((TOPK + BLOCK - 1) // BLOCK):\n"
            "            j = t * BLOCK + lane\n",
            "        for ft in tl.static_range((TOPK + BLOCK - 1) // BLOCK):\n"
            "            j = ft * BLOCK + lane\n",
        ),
    ]
    for a, b in subs:
        assert src.count(a) == 1, f"the reverted file changed shape near: {a[:60]!r}"
        src = src.replace(a, b, 1)
    assert "SSTRIDE = int(" in src and "TARGET_MULT = float(" in src
    _assert_no_range_kind_clash(src)
    f = pathlib.Path(tempfile.mkdtemp(prefix="sample8_")) / "topk_prefill_sampled.py"
    f.write_text(src)
    name = "flaggems_vllm.runtime.backend._hygon.fused._topk_prefill_sample8"
    spec = importlib.util.spec_from_file_location(name, str(f))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_arm(mod, ratio, sstride, target_mult):
    mod.SAMPLED_MIN_VOCAB_PER_TOPK = ratio
    mod.SSTRIDE = sstride
    mod.TARGET_MULT = target_mult
    # 16383 x 2048 candidates needs 2^26; the buffers are torch.empty, so this
    # costs address space rather than traffic (cand_idx + cand_val + hist is
    # ~400 MB at the largest shape, against ~270 MB for the generic path).
    mod._MAX_CAND_ELEMS = 1 << 27
    with mod._SPLAN_LOCK:
        mod._SPLANS.clear()


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
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    mod = sampled_module()
    dev = "cuda"
    occupancy("before")
    print(
        f"\ndevice us, interleaved over {ROUNDS} rounds, each arm's FASTEST round."
        "\n'off' is the same binary with the gate shut and is the denominator."
        "\n'ship' is today's production override (it HAS the one-scan patch the"
        "\nothers lack) and is there for absolute position only."
        "\n'outside' = share of rows landing outside [top_k, CAP]; each pays a"
        "\nfull redo inside _s_finish. Read it before the timings.\n"
    )
    head = f"  {'shape':>12} {'k':>5} {'ship':>8} {'off':>8}"
    for tag, r, _, _ in ARMS:
        if r:
            head += f"{tag:>9}"
    for tag, r, _, _ in ARMS:
        if r:
            head += f"{'off/' + tag:>11}"
    head += f"{'cand med':>9}{'outside':>8}{'ans':>6}"
    print(head)

    logs = {t: [] for t, r, _, _ in ARMS if r}
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        tbuf = (buf * 4).round() / 4
        tied = torch.as_strided(tbuf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

        want = {}
        for label, src in (("normal", logits), ("tied", tied)):
            want[label] = torch.topk(src, top_k, dim=1).values.sort(dim=1).values

        def check(run):
            ok = True
            for label, src in (("normal", logits), ("tied", tied)):
                idx.fill_(-9)
                run(src)
                torch.cuda.synchronize()
                got = src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                ok = ok and torch.allclose(got, want[label]) and bool((idx >= 0).all())
            return ok

        def ship(src):
            ov.top_k_per_row_prefill(src, starts, ends, idx, rows, stride0, 1, top_k)

        ok = check(ship)
        t_ship = min(device_us(lambda: ship(logits)) for _ in range(ROUNDS))

        times, cmed, outside = {}, {}, {}
        for tag, ratio, sstride, tmult in ARMS:
            set_arm(mod, ratio, sstride, tmult)

            def run(src):
                mod.top_k_per_row_prefill(
                    src, starts, ends, idx, rows, stride0, 1, top_k
                )

            ok = check(run) and ok
            # routing assertion: did this arm actually take the sampled path?
            took = bool(mod._SPLANS)
            assert took == bool(ratio), f"{tag}: sampled={took}, expected {bool(ratio)}"
            if ratio:
                plan = next(iter(mod._SPLANS.values()))
                got = (
                    plan.prepare.constexprs["STRIDE"],
                    plan.prepare.constexprs["TARGET"],
                )
                assert got == (sstride, int(top_k * tmult)), f"{tag}: constexprs {got}"
                plan.prepare(
                    logits, starts, ends, plan.hist, plan.thr, plan.cnt, stride0
                )
                plan.collect(
                    logits,
                    starts,
                    ends,
                    plan.thr,
                    plan.cnt,
                    plan.cand_idx,
                    plan.cand_val,
                    stride0,
                )
                torch.cuda.synchronize()
                c = plan.cnt.float()
                lo = min(top_k, vocab)
                outside[tag] = float(((c < lo) | (c > plan.cap)).float().mean()) * 100
                cmed[tag] = int(c.median())
            times[tag] = min(device_us(lambda: run(logits)) for _ in range(ROUNDS))

        line = (
            f"  {f'{rows}x{vocab}':>12} {top_k:>5} {t_ship:>8.1f} {times['off']:>8.1f}"
        )
        for tag, r, _, _ in ARMS:
            if r:
                line += f"{times[tag]:>9.1f}"
        for tag, r, _, _ in ARMS:
            if r:
                ratio = times["off"] / times[tag]
                logs[tag].append(math.log(ratio))
                line += f"{ratio:>11.3f}"
        best = min((t for t, r, _, _ in ARMS if r), key=lambda t: times[t])
        line += f"{cmed.get(best, 0):>9}{outside.get(best, 0):>7.1f}%"
        line += f"{'OK' if ok else 'WRONG':>6}"
        print(line, flush=True)
        for tag, r, _, _ in ARMS:
            if r and outside.get(tag, 0) > 0.05:
                print(
                    f"      ! {tag}: {outside[tag]:.2f}% of rows outside the window,"
                    f" median {cmed[tag]} candidates",
                    flush=True,
                )

    print()
    for tag, r, _, _ in ARMS:
        if r:
            g = math.exp(sum(logs[tag]) / len(logs[tag]))
            print(f"  geomean off/{tag}: {g:.3f}")
    occupancy("after")
    print(
        "\n  A ratio > 1 means the sampled path is faster than the same binary"
        "\n  with its gate shut. The 'cand med' / 'outside' columns belong to"
        "\n  whichever sampled arm was fastest on that row."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
