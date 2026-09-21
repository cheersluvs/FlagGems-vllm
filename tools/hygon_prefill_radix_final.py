"""De-TLE the final select, then re-ask the bin-count question.

WHY. The non-TLE branch of `_top_k_per_row_job` ranks the threshold bin's
candidates by COUNTING, O(final_cnt^2):

    for sort_chunk in tl.range(0, sort_chunks):
        for j in tl.range(0, final_cnt):        # scalar load per j
            logit_j = tl.load(s_final_logits_ptr + j)

`_final_select_radix`, directly above it, does the same job in four linear
8-bit radix rounds over the full 32-bit ordered key -- and is gated on
`USE_RADIX_FINAL and HAS_TLE`, so no non-TLE backend can reach it. Its only
TLE dependencies are `tle.gpu.alloc` for 256 int32 and `tle.cumsum`. Both have
proven substitutes here: a global scratch slice, and `tl.cumsum(c) - c` (the
same rewrite the shipped one-scan step already runs). Decode's `_tail` IS this
algorithm written without TLE, so the code is not speculative.

WHAT THIS IS REALLY FOR. On its own the swap should be a WASH or slightly
NEGATIVE at 2048 bins: final_cnt is 27-36 per row on the dense shapes
(tools/hygon_prefill_step0.py), so the quadratic loop is 27 scalar loads while
the radix pays four rounds of 256-wide clear + cumsum whatever final_cnt is.
Read arm `rx2048` as a cost floor, not as the result.

The result is the narrowed arms. Narrowing 2048 -> 512 -> 256 multiplies
final_cnt by ~4 and ~8 (27 -> 104 -> 217; 177 -> 768 -> 2126), which the
QUADRATIC ranker charges 16x and 64x for. tools/hygon_prefill_bins256.py
measured 512 bins at 0.59-0.87 on the operator while the harness said the
atomic alone gets 1.32x cheaper there -- that verdict was taken against the
quadratic ranker and is what this re-takes. And the prize is bigger than the
atomic: on (16383,4095) the counters read WRITE_SIZE 180.7 MB of which the
output indices are 33.5, leaving ~134 MB that is the per-row 2048-bin clear
(16383 x 8 KB) -- 74% of all writes. At 512 bins it is 33 MB. Separately, that
shape runs 128 threads/program and arch_vgpr 96 caps a CU at 16 programs, so
~1280 programs are live x 8 KB = 10.2 MB of histogram scratch against an 8 MB
L2 (hit rate 0.79); at 512 bins it is 2.6 MB and fits.

ARMS, all on top of the shipped one-scan patch, all with production routing
and geometry, interleaved, each arm's FASTEST round reported (the box is
shared; contention only ever adds time):

    base     2048 bins, quadratic final select   <- what production runs today
    rx2048   2048 bins, radix final select
    rx512     512 bins, radix final select
    rx256     256 bins, radix final select

Ratios are base/arm, so > 1 means the arm is faster. 256 bins is expected to
lose on (64,129280): 63 of its 64 rows overflow NUM_FINAL_ITEMS there, which
forces STEP 1 to run. Measured anyway, since a shape-gated bin count would
need the number.

Correctness is checked on every arm against torch.topk, on normal logits and
on rounded ones (which pack the threshold bin and make STEP 1-3 run). The
oracle compares SORTED VALUES, not order: the radix select appends rather than
ranks, so its output order differs from the quadratic path's by design, and
tests/test_top_k_per_row_prefill.py compares the same way.

    tools/vendor_probe.sh tools/hygon_prefill_radix_final.py hygon_prefill_radix_final
"""

import importlib.util
import math
import pathlib
import sys
import tempfile
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 16385, 512, 16648),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
ROUNDS = 7

# (tag, step0 bins, radix final select?)
ARMS = [
    ("base", 2048, False),
    ("rx2048", 2048, True),
    ("rx512", 512, True),
    ("rx256", 256, True),
]


def occupancy(tag):
    """What else is on the card. This box is shared, and one earlier probe came
    back with per-round ratios spread 0.08-4.2 while the same shapes had held
    within 10% that morning."""
    import shutil
    import subprocess

    for cmd in (["hy-smi"], ["rocm-smi", "--showpids"], ["rocm-smi"]):
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
        print(f"--- card occupancy {tag}: {' '.join(cmd)}")
        print("\n".join(out.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


CLEAR_OLD = """    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if STEP == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_histogram_ptr + clear_bins, 0)
    tl.debug_barrier()
"""
CLEAR_NEW = """    RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE
    radix_bins = tl.arange(0, RADIX_SIZE)
    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))
    tl.debug_barrier()
"""
SCAN_OLD = """    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    threshold_found = tl.full((), False, dtype=tl.int1)
    for round_idx in tl.static_range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(s_histogram_ptr + bins)
            if HAS_TLE:
                prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            else:
                counts_total = tl.sum(counts)
                prefix_sum = counts_total - tl.cumsum(counts, axis=0, reverse=True)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            if STEP == 3:
                tl.store(s_histogram_ptr + bins, prefix_sum)
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    tl.debug_barrier()
    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
"""
SCAN_NEW = """    counts = tl.load(s_histogram_ptr + radix_bins)
    incl = last_value + tl.cumsum(counts, axis=0)
    prefix_sum = incl - counts
    threshold_mask = (prefix_sum < TOPK) & (incl >= TOPK)
    threshold_bin_idx = tl.min(
        tl.where(threshold_mask, radix_bins, RADIX_SIZE), axis=0
    ).to(tl.int32)
    final_bin_size = tl.max(tl.where(threshold_mask, counts, 0), axis=0)
    if STEP == 3:
        tl.store(s_histogram_ptr + radix_bins, prefix_sum)
        tl.debug_barrier()
"""

RADIX_OLD = (
    "    RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE\n"
)
KEY_OLD = "        bin_idx = (mapped >> 5).to(tl.uint32)\n"

# The histogram scratch gains 256 int32 per row so the radix rounds have
# somewhere to count. NOTE for anyone shipping this: the widened stride is
# only safe because HAS_TLE is False here, so tle_top_k_per_row_prefill --
# which reaches the same else-branch with TLE-allocated NUM_BINS-wide buffers
# -- is never launched on this card. A shipped version has to take the count
# buffer from somewhere both paths agree on. Every existing use of s_histogram_ptr stays inside
# [0, 2048): the clear and scan are RADIX_SIZE <= 2048 wide, _process_bins
# writes candidate indices at final_pos < NUM_FINAL_ITEMS and, at STEP 3, a
# prefix sum at bin_idx < 1024. Only the row STRIDE changes.
ALLOC_OLD = "            (num_rows, NUM_BINS), device=device, dtype=torch.int32\n"
ALLOC_NEW = "            (num_rows, NUM_BINS + 256), device=device, dtype=torch.int32\n"
STRIDE_OLD = "    s_histogram_ptr += row_id * NUM_BINS\n"
STRIDE_NEW = "    s_histogram_ptr += row_id * (NUM_BINS + 256)\n"

# Markers for the block that gets replaced. Spliced by index, not by
# str.replace: a reformat upstream would make a replace MISS SILENTLY and the
# arm would quietly measure the baseline again.
FINAL_HEAD = "        else:\n            base_idx = tl.load(s_found_topk_values_ptr)"
FINAL_TAIL = "\n    # out_indices_ptr is identical to s_out_indices_ptr for non-tle"

FINAL_NEW = """        else:
            RADIX_F: tl.constexpr = 256
            RADIX_BASE: tl.constexpr = 2048
            base_idx = tl.load(s_found_topk_values_ptr)
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
            remain = tl.minimum(TOPK - base_idx, final_cnt)
            rbins = tl.arange(0, RADIX_F)
            rcount_ptr = s_histogram_ptr + RADIX_BASE
            found_ptrs = s_found_topk_values_ptr + tl.zeros([BLOCK_SIZE], tl.int32)
            ones_f = tl.full([BLOCK_SIZE], 1, tl.int32)
            cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
            if remain > 0:
                desired = tl.zeros((), dtype=tl.uint32)
                desired_mask = tl.zeros((), dtype=tl.uint32)
                k_to_find = remain + 1
                for digit_pos in tl.static_range(24, -1, -8):
                    if k_to_find > 1:
                        tl.store(rcount_ptr + rbins, tl.zeros([RADIX_F], tl.int32))
                        tl.debug_barrier()
                        for rt in tl.range(0, cnt_tiles):
                            rpos = rt * BLOCK_SIZE + lane
                            rvalid = rpos < final_cnt
                            rx = tl.load(
                                s_final_logits_ptr + rpos, mask=rvalid, other=0
                            )
                            rkey = _convert_to_uint32(rx)
                            rdigit = ((rkey >> digit_pos) & (RADIX_F - 1)).to(tl.int32)
                            tl.atomic_add(
                                rcount_ptr + rdigit,
                                ones_f,
                                mask=rvalid & ((rkey & desired_mask) == desired),
                                sem="relaxed",
                                scope="cta",
                            )
                        tl.debug_barrier()
                        rcounts = tl.load(rcount_ptr + rbins)
                        rincl = tl.cumsum(rcounts, axis=0)
                        rprefix = rincl - rcounts
                        rhit = (rprefix < k_to_find) & (rincl >= k_to_find)
                        rb = tl.min(tl.where(rhit, rbins, RADIX_F), axis=0).to(tl.int32)
                        rb = tl.where(rb == RADIX_F, RADIX_F - 1, rb)
                        rlt = tl.max(tl.where(rbins == rb, rprefix, 0), axis=0).to(
                            tl.int32
                        )
                        desired = desired | (rb.to(tl.uint32) << digit_pos)
                        desired_mask = desired_mask | (
                            tl.full((), RADIX_F - 1, tl.uint32) << digit_pos
                        )
                        k_to_find = k_to_find - rlt
                thr_key = desired
                # everything strictly better than the k-th, then its equals
                for requal in tl.static_range(2):
                    for rt2 in tl.range(0, cnt_tiles):
                        rpos2 = rt2 * BLOCK_SIZE + lane
                        rvalid2 = rpos2 < final_cnt
                        rx2 = tl.load(
                            s_final_logits_ptr + rpos2, mask=rvalid2, other=0
                        )
                        rkey2 = _convert_to_uint32(rx2)
                        if requal == 0:
                            rtake = rvalid2 & (rkey2 < thr_key)
                        else:
                            rtake = rvalid2 & (rkey2 == thr_key)
                        rq = tl.atomic_add(
                            found_ptrs,
                            ones_f,
                            mask=rtake,
                            sem="relaxed",
                            scope="cta",
                        )
                        ridx = tl.load(s_histogram_ptr + rpos2, mask=rtake, other=0)
                        tl.store(
                            s_out_indices_ptr + rq, ridx, mask=rtake & (rq < TOPK)
                        )
                        if MULTIPLE_BLOCKS_PER_ROW:
                            tl.store(
                                s_out_logits_ptr + rq, rx2, mask=rtake & (rq < TOPK)
                            )
                    tl.debug_barrier()
            tl.debug_barrier()
"""


def slotscan_source():
    """The override's prefix-sum collection, sliced out of the override FILE.

    Borrowing the function object instead resolves its _extract_bin_idx in the
    OVERRIDE's globals -- the unpatched production copy -- so a narrowed arm
    would histogram narrow and collect wide. That produced wrong answers
    reported as a 1.40-1.45x speedup once already. Appending the SOURCE keeps
    every reference inside one module.
    """
    ov_src = pathlib.Path(
        import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
        ).__file__
    ).read_text()
    a = ov_src.index("@triton.jit\ndef _alloc_slots(")
    b = ov_src.index("# Rebind in the COPY only, before anything compiles.")
    return (
        "\n\n" + ov_src[a:b].rstrip() + "\n\n_process_bins = _process_bins_slotscan\n"
    )


def patched_source(step0_bins, radix_final, dense):
    """One-scan, optionally the radix final select, optionally narrowed bins,
    and for a dense arm the override's prefix-sum collection appended."""
    src = pathlib.Path(_generic.__file__).read_text()
    subs = [(CLEAR_OLD, CLEAR_NEW), (SCAN_OLD, SCAN_NEW)]
    if radix_final:
        subs += [(ALLOC_OLD, ALLOC_NEW), (STRIDE_OLD, STRIDE_NEW)]
    if step0_bins != 2048:
        shift = 5 + (11 - step0_bins.bit_length() + 1)
        subs += [
            (
                RADIX_OLD,
                "    RADIX_SIZE: tl.constexpr = (\n"
                "        RADIX10_SIZE\n"
                "        if STEP == 3\n"
                f"        else ({step0_bins} if STEP == 0 else RADIX11_SIZE)\n"
                "    )\n",
            ),
            (KEY_OLD, f"        bin_idx = (mapped >> {shift}).to(tl.uint32)\n"),
        ]
    for old, new in subs:
        n = src.count(old)
        assert n == 1, f"block found {n} times -- the operator changed:\n{old[:60]}"
        src = src.replace(old, new, 1)
    if radix_final:
        assert src.count(FINAL_HEAD) == 1, "final-select head not unique"
        a = src.index(FINAL_HEAD)
        assert src.count(FINAL_TAIL) == 1, "final-select tail not unique"
        b = src.index(FINAL_TAIL, a)
        src = src[:a] + FINAL_NEW.rstrip("\n") + src[b:]
    return src + (slotscan_source() if dense else "")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
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


def threshold_bin_size(logits, top_k, bins):
    """How many elements land in the bin holding rank top_k -- i.e. how much
    work the final select is handed. The operator's own STEP-0 key, on the
    host, so the table reads without cross-referencing another report."""
    shift = 5 + (11 - bins.bit_length() + 1)
    h = logits.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    mapped = torch.where(h & 0x8000 != 0, h, (~h) & 0x7FFF)
    binned = mapped >> shift
    thr = binned.sort(dim=1).values[:, top_k - 1]
    cnt = (binned == thr[:, None]).sum(1)
    return int(cnt.median()), int(cnt.max())


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="radixfinal_"))
    mods = {}
    for tag, nb, rx in ARMS:
        for kind in ("generic", "dense"):
            f = tmpdir / f"topk_prefill_{tag}_{kind}.py"
            f.write_text(patched_source(nb, rx, kind == "dense"))
            mods[(tag, kind)] = load(
                f"flaggems_vllm.ops._topk_prefill_rx_{tag}_{kind}", str(f)
            )
    dev = "cuda"
    occupancy("before")
    print(
        "\ndevice us, interleaved over"
        f" {ROUNDS} rounds, each arm's FASTEST round. Production routing and"
        "\ngeometry in every arm. 'fc' is the threshold bin's size (median /"
        "\nmax over rows) at that arm's bin count -- the work the final select"
        "\nis handed. Ratios are base/arm, so > 1 means the arm is faster.\n"
    )
    head = f"  {'shape':>18} {'module':>8}"
    for tag, _, _ in ARMS:
        head += f"{tag:>9}"
    head += f"{'fc 2048':>12}{'fc 512':>11}{'fc 256':>11}"
    for tag, _, _ in ARMS[1:]:
        head += f"{'b/' + tag:>9}"
    head += f"{'normal':>8}{'tied':>6}"
    print(head)

    logs = {tag: [] for tag, _, _ in ARMS[1:]}
    for rows, vocab, top_k, stride0 in SHAPES:
        kind = "dense" if vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k else "generic"
        geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        tbuf = (buf * 4).round() / 4
        tied = torch.as_strided(tbuf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        fcs = [threshold_bin_size(logits, top_k, nb) for nb in (2048, 512, 256)]
        torch.cuda.empty_cache()  # the host-side key pass peaks ~1 GB

        gos = []
        oks = {"normal": True, "tied": True}
        for tag, _, _ in ARMS:
            m = mods[(tag, kind)]
            if geo is None:
                m.NUM_THREADS_PER_BLOCK = _generic.NUM_THREADS_PER_BLOCK
                m._num_warps = _generic._num_warps
            else:
                m.NUM_THREADS_PER_BLOCK = geo[0]
                m._num_warps = lambda b, w=geo[1]: w

            def run(src, m=m):
                m.top_k_per_row_prefill(src, starts, ends, idx, rows, stride0, 1, top_k)

            for label, src in (("normal", logits), ("tied", tied)):
                want = torch.topk(src, top_k, dim=1).values.sort(dim=1).values
                idx.fill_(-9)
                run(src)
                torch.cuda.synchronize()
                got = src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                oks[label] = (
                    oks[label] and torch.allclose(got, want) and bool((idx >= 0).all())
                )
            gos.append(lambda run=run: run(logits))

        mins = [min(device_us(g) for _ in range(ROUNDS)) for g in gos]
        line = f"  {f'{rows}x{vocab}':>18} {kind:>8}"
        line += "".join(f"{v:>9.1f}" for v in mins)
        line += "".join(
            f"{f'{a}/{b}':>12}" if i == 0 else f"{f'{a}/{b}':>11}"
            for i, (a, b) in enumerate(fcs)
        )
        for (tag, _, _), v in zip(ARMS[1:], mins[1:]):
            logs[tag].append(math.log(mins[0] / v))
            line += f"{mins[0] / v:>9.3f}"
        line += f"{'OK' if oks['normal'] else 'WRONG':>8}"
        line += f"{'OK' if oks['tied'] else 'WRONG':>6}"
        print(line, flush=True)

    print()
    for tag, _, _ in ARMS[1:]:
        g = math.exp(sum(logs[tag]) / len(logs[tag]))
        print(f"  geomean base/{tag}: {g:.3f}")
    occupancy("after")
    print(
        "\n  rx2048 is the radix's FIXED cost, not the result: at 27-36"
        "\n  candidates the quadratic loop is 27 scalar loads while the radix"
        "\n  pays four 256-wide rounds regardless. The question is whether"
        "\n  rx512 clears base once narrowing stops being charged 16x."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
