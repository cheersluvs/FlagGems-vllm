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

"""top_k_per_row_prefill on Hygon BW1000: on dense rows, allocate output slots
by prefix sum and carry their counter through the histogram step.

The dense route uses the validated VEC=2 layout by default. Set
``FLAGGEMS_HYGON_TOPK_VEC2=0`` to fall back to the carried VEC=4 layout.

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

The carried-counter dense copy removes one global atomic per collection tile:
it loads the output count once at the start of each histogram step, derives
tile-local offsets by cumsum, and stores the accumulated count before its
barrier. The BW1000 v3 audit validated the exact generated source on padded
rows, ties, short and partial ranges; the follow-up B-C-C-B run passed 19
functional tests and improved the four dense benchmark shapes by 1.06-1.08x.
Set FLAGGEMS_HYGON_TOPK_CARRY=0 to retain the preceding dense implementation.
"""

import functools
import hashlib
import importlib.util
import logging
import os
import stat
import sys
import tempfile
import threading
from collections import OrderedDict
from importlib import import_module

import torch
import triton
import triton.language as tl

from ._top_k_per_row_prefill_carry_source import build_carry_source, set_vector_width
from ._top_k_per_row_prefill_final_source import build_final_source

_GENERIC_NAME = "flaggems_vllm.ops.top_k_per_row_prefill"
_DENSE_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_dense"
_CARRY_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_carry"
_VEC2_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_vec2"
_VEC2_FINAL_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_vec2_final"
_SHORT_BINS_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_short_bins"
_SPARSE_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_sparse"

_generic = import_module(_GENERIC_NAME)
_log = logging.getLogger(__name__)

# Dense iff vocab_size <= DENSE_VOCAB_PER_TOPK * top_k, i.e. density >= 10%.
DENSE_VOCAB_PER_TOPK = 10


# ---------------------------------------------------------------------------
# One threshold scan instead of a carried chain of rounds.
#
# The generic histogram step clears its bins in RADIX_SIZE // BLOCK_SIZE
# stores and finds the threshold in as many rounds -- each a BLOCK_SIZE-wide
# cumsum whose running total feeds the next, plus two masked stores of
# block-uniform scalars into global scratch, a barrier after the loop and two
# global reloads. The DSA bin_topk kernel does the same job in one scan. Here:
# one vectorised clear, one RADIX_SIZE-wide cumsum, the bin as a min-reduction
# and its size as a max-reduction, both kept in registers. Nothing downstream
# reads the two global scalar buffers (the job takes threshold_bin_idx from
# the return value), and a threshold always exists at that point because rows
# no longer than top_k return first.
#
# Measured on the operator, both arms at production routing and geometry,
# seven interleaved rounds, each arm's fastest round (tools/
# hygon_prefill_onescan.py; the card was shared, and contention only adds):
#
#     (64,129280)  1.093    (16383,4095)  1.014    (4100,1025)  1.309
#     (4,16385)    1.088    (12961,4100)  1.027
#     (4,8193)     1.117    (16380,5115)  1.038    geomean      1.094
#
# Largest where fixed cost is the largest share: the many-row shapes run at
# BLOCK_SIZE 256, so their chain was eight rounds deep. Correct on standard
# normal logits and on rounded ones, which overflow the threshold bin and make
# STEP 1-3 run.
#
# The change is applied as two exact text replacements to the generic
# module's source. If either block is not found exactly once -- upstream
# edited it -- the copies load the generic source unchanged and a warning is
# logged; an exception here would take every Hygon override down with it.
_ONESCAN_CLEAR_OLD = """    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if STEP == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_histogram_ptr + clear_bins, 0)
    tl.debug_barrier()
"""
_ONESCAN_CLEAR_NEW = """    RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE
    radix_bins = tl.arange(0, RADIX_SIZE)
    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))
    tl.debug_barrier()
"""
_ONESCAN_SCAN_OLD = """    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
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
_ONESCAN_SCAN_NEW = """    counts = tl.load(s_histogram_ptr + radix_bins)
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


def _onescan_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_ONESCAN", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _private_dir():
    """A directory only this user can write. The patched source is executed
    as code, and this box is shared, so a world-writable /tmp is not an
    acceptable place to look for it."""
    for base in (
        os.path.join(os.path.expanduser("~"), ".cache", "flaggems_vllm"),
        os.path.join(tempfile.gettempdir(), f"flaggems_vllm_{os.getuid()}"),
    ):
        try:
            os.makedirs(base, mode=0o700, exist_ok=True)
            st = os.stat(base)
            if st.st_uid == os.getuid() and not (
                st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                return base
        except OSError:
            continue
    return None


def _onescan_path():
    """Path of the patched generic source, or None to use the generic file."""
    if not _onescan_enabled():
        return None
    try:
        with open(_generic.__file__) as fh:
            src = fh.read()
        for old, new in (
            (_ONESCAN_CLEAR_OLD, _ONESCAN_CLEAR_NEW),
            (_ONESCAN_SCAN_OLD, _ONESCAN_SCAN_NEW),
        ):
            n = src.count(old)
            if n != 1:
                _log.warning(
                    "hygon top_k_per_row_prefill: one-scan patch skipped, a "
                    "block was found %d times; the generic step is used",
                    n,
                )
                return None
            src = src.replace(old, new, 1)
        base = _private_dir()
        if base is None:
            return None
        digest = hashlib.sha256(src.encode()).hexdigest()[:16]
        path = os.path.join(base, f"top_k_per_row_prefill_onescan_{digest}.py")
        if not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(dir=base, suffix=".py")
            with os.fdopen(fd, "w") as fh:
                fh.write(src)
            os.replace(tmp, path)
        # Never execute a file this process did not verify byte for byte.
        with open(path) as fh:
            if fh.read() != src:
                return None
        return path
    except Exception as exc:  # noqa: BLE001 - never break the override import
        _log.warning("hygon top_k_per_row_prefill: one-scan patch failed: %r", exc)
        return None


_ONESCAN_PATH = _onescan_path()


def _load_copy(name, path=None):
    """The generic module -- or the one-scan patch of it -- executed as a
    separate module. @triton.jit needs its functions' source on disk."""
    spec = importlib.util.spec_from_file_location(name, path or _generic.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dense = _load_copy(_DENSE_NAME, _ONESCAN_PATH)
# Sparse rows used the generic module itself; with the patch they get their
# own copy, which also stops this override mutating the shared module's
# launch globals for them.
_sparse = _load_copy(_SPARSE_NAME, _ONESCAN_PATH) if _ONESCAN_PATH else _generic
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


def _carry_path():
    """Build the self-contained dense copy tested by the BW1000 audit."""
    if os.environ.get("FLAGGEMS_HYGON_TOPK_CARRY", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return None
    if _ONESCAN_PATH is None:
        _log.warning("hygon prefill carry requires the one-scan source")
        return None
    try:
        with open(_ONESCAN_PATH) as fh:
            generic_source = fh.read()
        with open(__file__) as fh:
            override_source = fh.read()
        source = build_carry_source(generic_source, override_source)
        base = _private_dir()
        if base is None:
            return None
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = os.path.join(base, f"top_k_per_row_prefill_carry_{digest}.py")
        if not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(dir=base, suffix=".py")
            with os.fdopen(fd, "w") as fh:
                fh.write(source)
            os.replace(tmp, path)
        with open(path) as fh:
            if fh.read() != source:
                return None
        return path
    except Exception as exc:  # noqa: BLE001 - preserve the shipped dense path
        _log.warning("hygon prefill carry source skipped: %r", exc)
        return None


_CARRY_PATH = _carry_path()
try:
    _dense_carry = _load_copy(_CARRY_NAME, _CARRY_PATH) if _CARRY_PATH else None
except Exception as exc:  # noqa: BLE001 - preserve the shipped dense path
    _log.warning("hygon prefill carry module skipped: %r", exc)
    _dense_carry = None


def _vec2_path():
    """Build the validated dense VEC=2 path; retain VEC=4 as fallback."""
    if os.environ.get("FLAGGEMS_HYGON_TOPK_VEC2", "1").strip().lower() not in (
        "1",
        "true",
        "on",
        "yes",
    ):
        return None
    if _CARRY_PATH is None:
        return None
    try:
        with open(_CARRY_PATH) as fh:
            source = set_vector_width(fh.read(), 2)
        base = _private_dir()
        if base is None:
            return None
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = os.path.join(base, f"top_k_per_row_prefill_vec2_{digest}.py")
        if not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(dir=base, suffix=".py")
            with os.fdopen(fd, "w") as fh:
                fh.write(source)
            os.replace(tmp, path)
        with open(path) as fh:
            if fh.read() != source:
                return None
        return path
    except Exception as exc:  # noqa: BLE001 - preserve carried VEC=4
        _log.warning("hygon prefill VEC=2 source skipped: %r", exc)
        return None


_VEC2_PATH = _vec2_path()
try:
    _dense_vec2 = _load_copy(_VEC2_NAME, _VEC2_PATH) if _VEC2_PATH else None
except Exception as exc:  # noqa: BLE001 - preserve carried VEC=4
    _log.warning("hygon prefill VEC=2 module skipped: %r", exc)
    _dense_vec2 = None


def _final_network_path():
    """Build a separate exact final-selector copy of the dense VEC2 route."""
    if _VEC2_PATH is None:
        return None
    if os.environ.get("FLAGGEMS_HYGON_TOPK_FINAL_NETWORK", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return None
    try:
        with open(_VEC2_PATH) as fh:
            source = build_final_source(fh.read())
        base = _private_dir()
        if base is None:
            return None
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = os.path.join(base, f"top_k_per_row_prefill_vec2_final_{digest}.py")
        if not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(dir=base, suffix=".py")
            with os.fdopen(fd, "w") as fh:
                fh.write(source)
            os.replace(tmp, path)
        with open(path) as fh:
            if fh.read() != source:
                return None
        return path
    except Exception as exc:  # noqa: BLE001 - use the validated VEC2 route
        _log.warning("hygon prefill final-network source skipped: %r", exc)
        return None


_VEC2_FINAL_PATH = _final_network_path()
try:
    _dense_vec2_final = (
        _load_copy(_VEC2_FINAL_NAME, _VEC2_FINAL_PATH) if _VEC2_FINAL_PATH else None
    )
except Exception as exc:  # noqa: BLE001 - use the validated VEC2 route
    _log.warning("hygon prefill final-network module skipped: %r", exc)
    _dense_vec2_final = None


def _short_bins_path():
    """Build the measured 512-bin STEP-0 dense specialization."""
    if _VEC2_PATH is None:
        return None
    if os.environ.get("FLAGGEMS_HYGON_TOPK_SHORT_BINS", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return None
    try:
        with open(_VEC2_PATH) as fh:
            source = fh.read()
        old_key = "bin_idx = (mapped >> 5).to(tl.uint32)"
        if source.count(old_key) != 1:
            raise ValueError("STEP-0 key extraction source drift")
        source = source.replace(old_key, "bin_idx = (mapped >> 7).to(tl.uint32)", 1)
        old_radix = (
            "RADIX_SIZE: tl.constexpr = " "RADIX10_SIZE if STEP == 3 else RADIX11_SIZE"
        )
        new_radix = (
            "RADIX_SIZE: tl.constexpr = ("
            "RADIX10_SIZE if STEP == 3 else "
            "(512 if STEP == 0 else RADIX11_SIZE))"
        )
        if source.count(old_radix) != 1:
            raise ValueError("one-scan radix source drift")
        source = source.replace(old_radix, new_radix, 1)
        base = _private_dir()
        if base is None:
            return None
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = os.path.join(base, f"top_k_per_row_prefill_short_bins_{digest}.py")
        if not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(dir=base, suffix=".py")
            with os.fdopen(fd, "w") as fh:
                fh.write(source)
            os.replace(tmp, path)
        with open(path) as fh:
            if fh.read() != source:
                return None
        return path
    except Exception as exc:  # noqa: BLE001 - preserve the dense fallback
        _log.warning("hygon prefill short-bins source skipped: %r", exc)
        return None


_SHORT_BINS_PATH = _short_bins_path()
try:
    _dense_short_bins = (
        _load_copy(_SHORT_BINS_NAME, _SHORT_BINS_PATH) if _SHORT_BINS_PATH else None
    )
except Exception as exc:  # noqa: BLE001 - preserve the dense fallback
    _log.warning("hygon prefill short-bins module skipped: %r", exc)
    _dense_short_bins = None


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
# The crossover probe showed a stable STEP-0 win with 512 bins for the
# benchmark's top_k=512 rows up through vocab/row_len 1536.  Do not apply it
# to larger rows: 1792 was already neutral and the 4095/5115 cases regressed.
SHORT_BINS_TOPK = 512
SHORT_BINS_MAX_VOCAB = 1536


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
    id(m): (m.NUM_THREADS_PER_BLOCK, m._num_warps)
    for m in (
        _sparse,
        _dense,
        _dense_carry,
        _dense_vec2,
        _dense_vec2_final,
        _dense_short_bins,
    )
    if m is not None
}


# The non-TLE host wrapper allocates six scratch tensors for every call. On
# BW1000, reusing one plan per module/device/row-count removes 4-52% of wall
# time on the benchmark's small-row and four-row shapes. Keep only the most
# recently used row-count for each loaded route and cap live storage so a
# serving process cannot accumulate one full scratch set for every request
# shape. The caller already holds _LAUNCH_LOCK, so cache mutation is serialized.
_SCRATCH_CACHE = OrderedDict()
_SCRATCH_CACHE_BYTES = 0
_SCRATCH_CACHE_LIMIT = 512 * 1024 * 1024


def _scratch_reuse_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _scratch_buffers(mod, device, num_rows):
    """Return the non-TLE scratch set, reusing one active shape per route."""
    global _SCRATCH_CACHE_BYTES

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = torch.cuda.current_stream(device).cuda_stream
    key = (id(mod), device.type, device_index, stream_id)
    num_bins = int(mod.NUM_BINS)
    num_final_items = int(mod.NUM_FILNAL_ITEMS)
    cached = _SCRATCH_CACHE.get(key)
    if cached is not None:
        cached_rows, cached_bins, cached_final, _, buffers = cached
        if (
            cached_rows == num_rows
            and cached_bins == num_bins
            and cached_final == num_final_items
        ):
            _SCRATCH_CACHE.move_to_end(key)
            return buffers
        del _SCRATCH_CACHE[key]
        _SCRATCH_CACHE_BYTES -= cached[3]

    allocation_bytes = (
        num_rows * num_bins * 4 + num_rows * num_final_items * 4 + num_rows * 4 * 4
    )
    while (
        _SCRATCH_CACHE
        and _SCRATCH_CACHE_BYTES + allocation_bytes > _SCRATCH_CACHE_LIMIT
    ):
        _, old = _SCRATCH_CACHE.popitem(last=False)
        _SCRATCH_CACHE_BYTES -= old[3]

    buffers = (
        torch.empty((num_rows, num_bins), device=device, dtype=torch.int32),
        torch.empty((num_rows, num_final_items), device=device, dtype=torch.float32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
    )
    if allocation_bytes <= _SCRATCH_CACHE_LIMIT:
        _SCRATCH_CACHE[key] = (
            num_rows,
            num_bins,
            num_final_items,
            allocation_bytes,
            buffers,
        )
        _SCRATCH_CACHE_BYTES += allocation_bytes
    return buffers


def _top_k_per_row_prefill_reuse(
    mod, logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    scratch = _scratch_buffers(mod, logits.device, num_rows)
    return mod.non_tle_top_k_per_row_prefill[(num_rows,)](
        logits,
        indices,
        row_starts,
        row_ends,
        stride0,
        stride1,
        logits.shape[1],
        *scratch,
        TOPK=top_k,
        BLOCK_SIZE=mod.NUM_THREADS_PER_BLOCK,
        ROW_OFFSET=0,
        num_warps=mod._num_warps(mod.NUM_THREADS_PER_BLOCK),
    )


def _select_module(logits, num_rows, top_k):
    """Pick the original path or the measured Hygon dense final specialization."""
    vocab = logits.shape[1]
    if _ENABLED and vocab <= DENSE_VOCAB_PER_TOPK * top_k:
        if (
            _dense_short_bins is not None
            and top_k == SHORT_BINS_TOPK
            and vocab <= SHORT_BINS_MAX_VOCAB
        ):
            return _dense_short_bins
        if (
            _dense_vec2_final is not None
            and not _dense_vec2_final.HAS_TLE
            and num_rows >= 8192
            and top_k == 512
            and 2048 <= vocab <= 5120
            and logits.dtype == torch.float32
        ):
            return _dense_vec2_final
        return (
            _dense_vec2
            if _dense_vec2 is not None
            else (_dense_carry if _dense_carry is not None else _dense)
        )
    return _sparse


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Dense rows through the prefix-sum copy, everything else through generic,
    each launched at the geometry its occupancy wants."""
    mod = _select_module(logits, num_rows, top_k)
    geo = _geometry(num_rows, logits.shape[1]) if _GEOMETRY else None
    with _LAUNCH_LOCK:
        if geo is None:
            mod.NUM_THREADS_PER_BLOCK, mod._num_warps = _GENERIC_DEFAULTS[id(mod)]
        else:
            block, warps = geo
            mod.NUM_THREADS_PER_BLOCK = block
            mod._num_warps = lambda block_size, w=warps: w
        if _scratch_reuse_enabled() and not getattr(mod, "HAS_TLE", False):
            return _top_k_per_row_prefill_reuse(
                mod,
                logits,
                row_starts,
                row_ends,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
            )
        return mod.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )
