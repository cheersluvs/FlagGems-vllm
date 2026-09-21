"""Index the histogram scratch by PROGRAM instead of by row.

WHY. The non-TLE prefill launches one program per row and gives each row its
own 2048-bin scratch slice, which the kernel clears itself. The counters on
(16383,4095) read WRITE_SIZE 180.7 MB against 33.5 MB of output indices, so
about 134 MB -- 74% of all writes, 18% of all traffic -- is 16383 rows x 8 KB
of clearing. Every arm of every probe so far has paid it, including both bin
count sweeps, so it has never actually been priced.

The stores cannot be removed: each row needs zeroed bins. But they do not have
to land on 134 MB of DISTINCT addresses. That shape runs 128 threads/program
and arch_vgpr 96 caps a CU at 16 resident programs, so ~1280 programs are live
at a time -- a working set of 10.2 MB against an 8 MB L2 (hit rate 0.79). Hold
the grid at P programs, index the scratch by program id, and let each program
loop over rows: the same number of stores now rewrite P x 8 KB, and below
~1000 programs that fits in L2 and stops reaching HBM.

This is NOT the stage split (refuted at 0.833). That split work across LAUNCHES
and made every chunk re-run the radix; this is one launch, one program per row
at a time, exactly the same work in the same order. The only changes are the
grid size, which pointer the scratch comes from, and a row loop.

ARMS, all on the shipped one-scan patch, production routing and geometry:

    base    one program per row, scratch per row       <- production
    pfull   persistent kernel, P = num_rows            <- isolates the loop
                                                          and pid indexing at
                                                          an UNCHANGED footprint
    p1280   P = 1280, working set 10.2 MB   (today's live count)
    p640    P =  640,  5.1 MB
    p320    P =  320,  2.6 MB               (fits L2)

`pfull` is the control that matters: if base and pfull differ, the row loop
itself costs something and the P arms have to beat that, not base. On the
few-row shapes (4 and 64 rows) P clamps to num_rows, so every P arm collapses
onto pfull -- those rows are there to show the loop's own cost, not locality.

RISK, flagged rather than designed around: `_top_k_per_row_job` returns early
when row_len <= TOPK, and that return now sits inside a loop. No benchmark
shape takes it, but it must still COMPILE. Each arm is built and launched
inside its own try/except so one failure reports itself instead of killing the
run.

    tools/vendor_probe.sh tools/hygon_prefill_persistent.py hygon_prefill_persistent
"""

import importlib.util
import math
import pathlib
import sys
import tempfile
import traceback
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

# (tag, programs; 0 = one per row, None = not persistent at all)
ARMS = [
    ("base", None),
    ("pfull", 0),
    ("p1280", 1280),
    ("p640", 640),
    ("p320", 320),
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

# Spliced by index, not by str.replace: a reformat upstream would make a
# replace MISS SILENTLY and the arm would quietly measure the baseline again.
KERNEL_HEAD = "@triton.jit\ndef non_tle_top_k_per_row_prefill("
KERNEL_TAIL = "\n\ndef top_k_per_row_prefill("
HOST_HEAD = "    else:\n        # based on tle version"

KERNEL_NEW = """@triton.jit
def non_tle_top_k_per_row_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    num_rows,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_BINS: tl.constexpr = 2048
    NUM_FILNAL_ITEMS: tl.constexpr = 2048

    pid = tl.program_id(0) + ROW_OFFSET
    # Scratch is indexed by PROGRAM, so the whole histogram working set is
    # NUM_PROGRAMS x 8 KB however many rows there are, and each program
    # re-clears its own slice for every row it takes. Rows are strided across
    # programs so that programs running together read neighbouring rows.
    p_hist = s_histogram_ptr + pid * NUM_BINS
    p_final = s_final_logits_ptr + pid * NUM_FILNAL_ITEMS
    p_cnt = s_final_cnt_ptr + pid
    p_thr = s_threshold_bin_idx_ptr + pid
    p_size = s_final_bin_size_ptr + pid
    p_found = s_found_topk_values_ptr + pid
    for row_id in tl.range(pid, num_rows, NUM_PROGRAMS):
        row_start = tl.load(row_starts + row_id)
        row_end = tl.load(row_ends + row_id)
        # float4 align
        x_off_mod = (row_id * stride0 + row_start) % VEC
        skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
        row_out_ptr = out_indices_ptr + row_id * TOPK
        _top_k_per_row_job(
            logits_ptr + row_id * stride0,
            row_out_ptr,
            row_start,
            row_end,
            stride1,
            vocab_size,
            skip_elems,
            None,
            None,
            p_hist,
            p_final,
            p_cnt,
            p_thr,
            p_size,
            p_found,
            row_out_ptr,
            None,
            TOPK=TOPK,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_RADIX_FINAL=False,
            HAS_TLE=False,
            MULTIPLE_BLOCKS_PER_ROW=False,
            MERGE_BLOCKS=False,
        )
"""

HOST_NEW = """    else:
        # based on tle version
        device = logits.device
        nprog = num_rows if not PERSISTENT_PROGRAMS else PERSISTENT_PROGRAMS
        nprog = min(nprog, num_rows)
        s_histogram_ptr = torch.empty(
            (nprog, NUM_BINS), device=device, dtype=torch.int32
        )
        s_final_logits_ptr = torch.empty(
            (nprog, NUM_FILNAL_ITEMS), device=device, dtype=torch.float32
        )
        s_final_cnt_ptr = torch.empty((nprog,), device=device, dtype=torch.int32)
        s_threshold_bin_idx_ptr = torch.empty(
            (nprog,), device=device, dtype=torch.int32
        )
        s_final_bin_size_ptr = torch.empty((nprog,), device=device, dtype=torch.int32)
        s_found_topk_values_ptr = torch.empty(
            (nprog,), device=device, dtype=torch.int32
        )
        non_tle_top_k_per_row_prefill[(nprog,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            vocab_size,
            num_rows,
            s_histogram_ptr,
            s_final_logits_ptr,
            s_final_cnt_ptr,
            s_threshold_bin_idx_ptr,
            s_final_bin_size_ptr,
            s_found_topk_values_ptr,
            TOPK=top_k,
            BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
            ROW_OFFSET=0,
            NUM_PROGRAMS=nprog,
            num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
        )
"""


def slotscan_source():
    """The override's prefix-sum collection, sliced out of the override FILE.

    Borrowing the function object instead resolves its _extract_bin_idx in the
    OVERRIDE's globals -- the unpatched production copy -- so a patched arm
    would histogram one way and collect another. That produced wrong answers
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


def patched_source(persistent, dense):
    """One-scan, optionally the persistent grid, and for a dense arm the
    override's prefix-sum collection appended."""
    src = pathlib.Path(_generic.__file__).read_text()
    for old, new in ((CLEAR_OLD, CLEAR_NEW), (SCAN_OLD, SCAN_NEW)):
        n = src.count(old)
        assert n == 1, f"block found {n} times -- the operator changed"
        src = src.replace(old, new, 1)
    if persistent:
        for marker in (KERNEL_HEAD, KERNEL_TAIL, HOST_HEAD):
            assert src.count(marker) == 1, f"marker not unique: {marker[:40]!r}"
        a = src.index(KERNEL_HEAD)
        b = src.index(KERNEL_TAIL, a)
        src = src[:a] + KERNEL_NEW.rstrip("\n") + src[b:]
        a = src.index(HOST_HEAD)
        src = src[:a] + HOST_NEW.rstrip("\n") + "\n"
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


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="persistent_"))
    mods = {}
    for tag, prog in ARMS:
        for kind in ("generic", "dense"):
            f = tmpdir / f"topk_prefill_{tag}_{kind}.py"
            f.write_text(patched_source(prog is not None, kind == "dense"))
            m = load(f"flaggems_vllm.ops._topk_prefill_ps_{tag}_{kind}", str(f))
            if prog is not None:
                m.PERSISTENT_PROGRAMS = prog
            mods[(tag, kind)] = m
    dev = "cuda"
    occupancy("before")
    print(
        f"\ndevice us, interleaved over {ROUNDS} rounds, each arm's FASTEST"
        "\nround. Production routing and geometry in every arm. 'set' is the"
        "\nhistogram working set in MB (programs x 8 KB) -- the L2 is 8 MB."
        "\nRatios are base/arm, so > 1 means the arm is faster. On the 4- and"
        "\n64-row shapes every P clamps to num_rows, so those rows show the"
        "\nrow loop's own cost, not locality.\n"
    )
    head = f"  {'shape':>18} {'module':>8}"
    for tag, _ in ARMS:
        head += f"{tag:>9}"
    head += f"{'set p320':>10}"
    for tag, _ in ARMS[1:]:
        head += f"{'b/' + tag:>9}"
    head += f"{'normal':>8}{'tied':>6}"
    print(head)

    logs = {tag: [] for tag, _ in ARMS[1:]}
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

        mins, oks, notes = [], {}, []
        for tag, _ in ARMS:
            m = mods[(tag, kind)]
            if geo is None:
                m.NUM_THREADS_PER_BLOCK = _generic.NUM_THREADS_PER_BLOCK
                m._num_warps = _generic._num_warps
            else:
                m.NUM_THREADS_PER_BLOCK = geo[0]
                m._num_warps = lambda b, w=geo[1]: w

            def run(src, m=m):
                m.top_k_per_row_prefill(src, starts, ends, idx, rows, stride0, 1, top_k)

            ok = True
            try:
                for label, src in (("normal", logits), ("tied", tied)):
                    want = torch.topk(src, top_k, dim=1).values.sort(dim=1).values
                    idx.fill_(-9)
                    run(src)
                    torch.cuda.synchronize()
                    got = (
                        src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                    )
                    good = torch.allclose(got, want) and bool((idx >= 0).all())
                    oks[(tag, label)] = good
                    ok = ok and good
                mins.append(min(device_us(lambda: run(logits)) for _ in range(ROUNDS)))
            except Exception:  # noqa: BLE001 - one arm failing must not end the run
                notes.append(
                    f"{tag}: {traceback.format_exc().strip().splitlines()[-1]}"
                )
                oks[(tag, "normal")] = oks[(tag, "tied")] = False
                mins.append(float("nan"))

        wset = min(320, rows) * 8192 / 1e6
        line = f"  {f'{rows}x{vocab}':>18} {kind:>8}"
        line += "".join(f"{v:>9.1f}" for v in mins)
        line += f"{wset:>10.1f}"
        for (tag, _), v in zip(ARMS[1:], mins[1:]):
            r = mins[0] / v if v == v and v > 0 else float("nan")
            if r == r:
                logs[tag].append(math.log(r))
            line += f"{r:>9.3f}"
        allok = all(oks.get((t, lb), False) for t, _ in ARMS for lb in ("normal",))
        alltd = all(oks.get((t, lb), False) for t, _ in ARMS for lb in ("tied",))
        line += f"{'OK' if allok else 'WRONG':>8}{'OK' if alltd else 'WRONG':>6}"
        print(line, flush=True)
        for n in notes:
            print(f"      ! {n}", flush=True)
        if not (allok and alltd):
            bad = [f"{t}/{lb}" for (t, lb), v in oks.items() if not v]
            print(f"      ! failing arms: {', '.join(sorted(bad))}", flush=True)

    print()
    for tag, _ in ARMS[1:]:
        if logs[tag]:
            g = math.exp(sum(logs[tag]) / len(logs[tag]))
            print(f"  geomean base/{tag}: {g:.3f}  ({len(logs[tag])} shapes)")
    occupancy("after")
    print(
        "\n  Read pfull FIRST: it is the persistent kernel at an unchanged"
        "\n  footprint, so base/pfull is the row loop's own price and the P"
        "\n  arms have to beat THAT, not base, for locality to be the story."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
