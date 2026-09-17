# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""top_k_per_row_prefill on Hygon BW1000: on DENSE rows, allocate output slots
by prefix sum instead of one atomic per selected element.

WHY. prefill loses on all seven benchmark shapes against vLLM's C++ kernel here
(geomean 0.355), worst on the small-vocabulary ones. Per program it fits
base + ~19 ns x top_k + ~1.3 ns x vocab, and the k term is
`tl.atomic_add(found_topk_values_ptrs, ones, mask=take_lt)` in _process_bins:
one atomic per selected element, all to one address, ~12 ns each here (the
ablation removed 7.55 of 18.07 us per program with it). Shared memory does not
help on this card: smem scatter atomics measured 2.5-3.5x slower than global.

WHAT. A prefix sum over the tile's take-mask gives each selected element a
distinct offset; one atomic of the tile's COUNT gives the base. Its cost scales
with the tile's SIZE, the atomics' with how many it selects -- so the deciding
quantity is DENSITY, the fraction of elements selected, which for a full-range
row is top_k / vocab. Measured crossover ~9.4% (48 of 512); dispatched here at
vocab <= 10 * top_k.

WHY TWO MODULE COPIES, NOT A BRANCH. The first version branched inside the
kernel on each tile's own count. The A/B on this box (kernel mode, both passes
agreeing to 0.03):

    shape            dense?  branch-in-kernel
    (4100, 1025)       yes        1.97x
    (12961, 4100)      yes        1.57x
    (16380, 5115)      yes        1.41x
    (16383, 4095)      yes        1.40x
    (4, 8193)          no         0.77x   threshold was per-count, not density
    (4, 16385)         no         0.71x   same
    (64, 129280)       no         0.85x   took the atomic branch and STILL lost

The last row is the branch's own cost, ~0.18 us per call whether taken or not,
on the production shape. And the choice cannot move to the host by rebinding:
Triton binds module globals at compile time and caches, so one module can only
ever hold one _process_bins. So the generic module is loaded a SECOND time
under another name, the copy gets the prefix-sum _process_bins, and the host
picks a module per call. Sparse rows run the untouched generic kernel.

Density from vocab is conservative for partial-range rows (a shorter row is
denser than vocab suggests), so a miss falls back to generic, never to
something slower. FLAGGEMS_HYGON_TOPK_SLOTSCAN=0 always uses generic.
"""

import functools
import importlib.util
import os
import sys
import threading
from importlib import import_module

import triton
import triton.language as tl

_GENERIC_NAME = "flaggems_vllm.ops.top_k_per_row_prefill"
_DENSE_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_dense"

_generic = import_module(_GENERIC_NAME)

# Dense iff vocab_size <= DENSE_VOCAB_PER_TOPK * top_k, i.e. density >= 10%.
DENSE_VOCAB_PER_TOPK = 10


def _load_copy(name):
    """The generic module, executed again as a separate module. @triton.jit
    needs its functions' source on disk, which the generic file provides."""
    spec = importlib.util.spec_from_file_location(name, _generic.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dense = _load_copy(_DENSE_NAME)
_extract_bin_idx = _dense._extract_bin_idx


@triton.jit
def _alloc_slots(ptrs, take):
    """Same contract as tl.atomic_add(ptrs, 1, mask=take) with `ptrs` all at
    one counter: every taken lane gets a distinct slot. 1-D or [BLOCK, VEC]."""
    ti = take.to(tl.int32)
    flat = tl.reshape(ti, (ti.numel,))
    total = tl.sum(flat, axis=0)
    # One atomic, on lane 0 only, adds the tile's count and returns the
    # counter's previous value: the base of this tile's run of slots.
    first = tl.arange(0, ti.numel) == 0
    prev = tl.atomic_add(
        tl.reshape(ptrs, (ti.numel,)),
        flat * 0 + total,
        mask=first,
        sem="relaxed",
        scope="cta",
    )
    start = tl.sum(tl.where(first, prev, 0), axis=0)
    return tl.reshape(start + tl.cumsum(flat, axis=0) - flat, ti.shape)


@triton.jit
def _process_bins_slotscan(
    logits,
    in_range,
    ones,
    offs,  # row_start based
    found_topk_values_ptrs,
    final_cnt_ptrs,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    use_final,
    row_start,
    indices_ptr,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    bin_idx, is_partial_match = _extract_bin_idx(
        logits,
        in_range,
        logit_pattern,
        STEP=STEP,
    )
    take_lt = is_partial_match & (bin_idx < threshold_bin_idx) & write_directly
    # The only change from the generic function: slots for the definitely-in
    # elements come from a prefix sum, not from one atomic per element.
    out_pos_lt = _alloc_slots(found_topk_values_ptrs, take_lt)
    if MERGE_BLOCKS:
        indices = tl.load(
            indices_ptr + offs,
            mask=take_lt,
        )
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            indices,
            mask=take_lt,
        )
    elif MULTIPLE_BLOCKS_PER_ROW:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            (offs + row_start).to(tl.int32),
            mask=take_lt,
        )
        tl.store(
            s_out_logits_ptr + out_pos_lt,
            logits,
            mask=take_lt,
        )
    else:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            offs.to(tl.int32),
            mask=take_lt,
        )

    if STEP < 3:
        if use_final:
            take_eq_final = is_partial_match & (bin_idx == threshold_bin_idx)
            final_pos = tl.atomic_add(
                final_cnt_ptrs,
                ones,
                mask=take_eq_final,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_final_logits_ptr + final_pos,
                logits,
                mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
            )
            # s_histogram_ptr being used for indices in final sort
            if MERGE_BLOCKS:
                indices = tl.load(
                    indices_ptr + offs,
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
                tl.store(
                    s_histogram_ptr + final_pos,
                    indices,
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
            elif MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    s_histogram_ptr + final_pos,
                    (offs + row_start).to(tl.int32),
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
            else:
                tl.store(
                    s_histogram_ptr + final_pos,
                    offs.to(tl.int32),
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
    else:
        take_eq = is_partial_match & (bin_idx == threshold_bin_idx)
        # s_histogram_ptr being used for exclude prefix sum
        out_pos_eq = tl.atomic_add(
            s_histogram_ptr + bin_idx,
            ones,
            mask=take_eq,
            sem="relaxed",
            scope="cta",
        )
        if MERGE_BLOCKS:
            indices = tl.load(
                indices_ptr + offs,
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                indices,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        elif MULTIPLE_BLOCKS_PER_ROW:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                (offs + row_start).to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_logits_ptr + out_pos_eq,
                logits,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        else:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                offs.to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )


# Rebind in the COPY only, before anything compiles.
_dense._process_bins = _process_bins_slotscan


def _slotscan_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_SLOTSCAN", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


_ENABLED = _slotscan_enabled()


# ---------------------------------------------------------------------------
# Launch geometry by occupancy.
#
# Which BLOCK_SIZE x num_warps is fastest depends on how many rows share this
# card's SMs, not on elements per lane. Full operator, every point checked
# against torch.topk, ratio vs vLLM (tools/hygon_prefill_launch_sweep.py and
# tools/hygon_prefill_rows_sweep.py):
#
#   rows/SM         row 4096, k 512                row 129280, k 1024
#   < 4             all configs within noise       B512 w8 (today) best
#   4 - 16          B512 w4: +3% .. +16%           B512 w4: +3% .. +10%
#   32 - 52         B256 w2: +47% .. +57%          B256 w4: +13% .. +32%
#   204 (16383 r)   B256 w2: 1.95x                 --
#
# Many rows want narrow programs: past one row per SM the grid is the
# parallelism, and wider programs only crowd each other out. Long rows keep
# more threads per program than short ones at high occupancy. Where the switch
# between w2 and w4 falls for row lengths between 8192 and 129280 is UNMEASURED;
# those take w4, the choice measured for long rows.
#
# num_warps=1 is never used: it returned WRONG answers in the sweep.
#
# NUM_THREADS_PER_BLOCK and _num_warps are host-side globals read at each
# launch -- unlike the jit globals above, they are not baked into a compiled
# kernel -- so they are set per call. A lock keeps set-and-launch atomic.
# FLAGGEMS_HYGON_TOPK_GEOMETRY=0 leaves them at generic's values.

SHORT_ROW_MAX = 8192


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 80
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 80


def _geometry(num_rows, row_len):
    """(BLOCK_SIZE, num_warps) for this call, or None for generic's own."""
    sms = _sm_count()
    if num_rows < 4 * sms:
        return None
    if num_rows < 32 * sms:
        return 512, 4
    return (256, 2) if row_len <= SHORT_ROW_MAX else (256, 4)


def _geometry_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_GEOMETRY", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


_GEOMETRY = _geometry_enabled()
_LAUNCH_LOCK = threading.Lock()
_GENERIC_DEFAULTS = {
    id(m): (m.NUM_THREADS_PER_BLOCK, m._num_warps) for m in (_generic, _dense)
}


def _direct_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_PREFILL_DIRECT", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


_DIRECT = _direct_enabled()

# Scratch is (num_rows, 2048) int32 plus (num_rows, 2048) float32 -- 16 KB per
# row. Holding it in a plan is what removes the per-call allocation, but at
# 16383 rows that is 268 MB pinned per plan, so above this bound the plan keeps
# only the launcher and the scratch is allocated per call as before. The shapes
# that need the saving are the small ones, where the host cost is several times
# the kernel's.
SCRATCH_CACHE_BYTES = 64 << 20


class _Launch:
    """One kernel: JIT on first use, direct afterwards.

    The same recipe as the decode override's; copied rather than imported so
    the two operators do not depend on each other.
    """

    __slots__ = ("jit", "grid", "grid3", "constexprs", "num_warps", "runner")

    def __init__(self, jit, grid, constexprs, num_warps):
        self.jit = jit
        self.grid = grid
        self.grid3 = tuple(grid) + (1,) * (3 - len(grid))
        self.constexprs = constexprs
        self.num_warps = num_warps
        self.runner = None

    def __call__(self, *args):
        if self.runner is not None:
            self.runner(*args, *self.constexprs.values())
            return
        ck = self.jit.run(
            *args,
            **self.constexprs,
            num_warps=self.num_warps,
            grid=self.grid,
            warmup=False,
        )
        if ck is not None:
            self.runner = ck[self.grid3]


def _scratch(mod, num_rows, device):
    import torch

    return (
        torch.empty((num_rows, mod.NUM_BINS), dtype=torch.int32, device=device),
        torch.empty(
            (num_rows, mod.NUM_FILNAL_ITEMS), dtype=torch.float32, device=device
        ),
        torch.empty((num_rows,), dtype=torch.int32, device=device),
        torch.empty((num_rows,), dtype=torch.int32, device=device),
        torch.empty((num_rows,), dtype=torch.int32, device=device),
        torch.empty((num_rows,), dtype=torch.int32, device=device),
    )


class _Plan:
    """A cached launcher for one (module, shape, geometry, specialisation)."""

    __slots__ = ("launch", "scratch")

    def __init__(self, mod, device, num_rows, top_k, block, warps):
        self.launch = _Launch(
            mod.non_tle_top_k_per_row_prefill,
            (num_rows,),
            {"TOPK": top_k, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
            warps,
        )
        per_row = (mod.NUM_BINS + mod.NUM_FILNAL_ITEMS) * 4
        self.scratch = (
            _scratch(mod, num_rows, device)
            if num_rows * per_row <= SCRATCH_CACHE_BYTES
            else None
        )


_PLANS = {}
_PLANS_MAX = 32
_PLAN_LOCK = threading.Lock()


def _aligned(t):
    return t.data_ptr() % 16 == 0


def _direct_ok(mod, logits, row_starts, row_ends, num_rows):
    import torch

    return (
        _DIRECT
        and not getattr(mod, "HAS_TLE", False)
        and num_rows > 0
        and num_rows == logits.shape[0]
        and logits.dtype == torch.float32
        and row_starts.dtype == torch.int32
        and row_ends.dtype == torch.int32
    )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Dense rows through the prefix-sum copy, everything else through generic,
    each launched at the geometry its occupancy wants -- and, where the shape
    allows it, through a cached CompiledKernel rather than Triton's dispatch.

    The launch mechanism is worth nothing in `--mode kernel`, which times the
    kernel: it is worth ~125 us per call of HOST time on the small shapes,
    where the operator's own dispatch measured 155 us against 27 us of device
    work. `--mode operator` and real serving pay that; kernel mode cannot see
    it. FLAGGEMS_HYGON_TOPK_PREFILL_DIRECT=0 restores the dispatch path.
    """
    vocab_size = logits.shape[1]
    if _ENABLED and vocab_size <= DENSE_VOCAB_PER_TOPK * top_k:
        mod = _dense
    else:
        mod = _generic
    geo = _geometry(num_rows, vocab_size) if _GEOMETRY else None
    if geo is None:
        block, warps_of = _GENERIC_DEFAULTS[id(mod)]
        warps = warps_of(block)
    else:
        block, warps = geo

    if _direct_ok(mod, logits, row_starts, row_ends, num_rows):
        key = (
            logits.device,
            id(mod),
            num_rows,
            vocab_size,
            top_k,
            stride0,
            stride1,
            block,
            warps,
            _aligned(logits),
            _aligned(row_starts),
            _aligned(row_ends),
            _aligned(indices),
        )
        with _PLAN_LOCK:
            plan = _PLANS.get(key)
            if plan is None:
                if len(_PLANS) >= _PLANS_MAX:
                    _PLANS.pop(next(iter(_PLANS)))
                plan = _PLANS[key] = _Plan(
                    mod, logits.device, num_rows, top_k, block, warps
                )
            scratch = plan.scratch or _scratch(mod, num_rows, logits.device)
            plan.launch(
                logits,
                indices,
                row_starts,
                row_ends,
                stride0,
                stride1,
                vocab_size,
                *scratch,
            )
        return indices

    # The module's own dispatch, which reads the geometry from its globals.
    with _LAUNCH_LOCK:
        if geo is None:
            mod.NUM_THREADS_PER_BLOCK, mod._num_warps = _GENERIC_DEFAULTS[id(mod)]
        else:
            mod.NUM_THREADS_PER_BLOCK = block
            mod._num_warps = lambda block_size, w=warps: w
        return mod.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )
