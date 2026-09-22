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


# ---------------------------------------------------------------------------
# A sampled threshold for very sparse rows.
#
# The generic step reads each row twice: once to build a 2048-bin histogram
# whose only output is the threshold, once to compact against it. Where the
# row is far longer than top_k the first pass can come from a SAMPLE instead.
# _s_hist reads every SSTRIDE-th TILE -- whole tiles, not strided elements,
# which touch one cache line per value and cost a full pass for a fraction of
# the data -- and the threshold is aimed at TARGET_MULT * top_k so that one
# pass collects a superset, which the finish ranks down exactly.
#
# WHEN IT PAYS. Collecting m * top_k candidates means ranking them afterwards,
# which the generic operator does not pay: its final stage sees the threshold
# bin alone, 27-36 elements. The saving scales with vocab and the ranking with
# m * top_k, so the figure of merit is vocab / (m * top_k), and the gate is the
# ratio. Measured on this card, benchmark SpeedUp on (64,129280) against the
# same binary with the gate shut (tools/hygon_prefill_sample_tight.py):
#
#     TARGET_MULT   1.00    1.10    1.25    1.50    3.00
#     rows outside  28.1%   10.9%    1.6%    0.0%     --
#     vs gate shut  0.434   0.494   1.402   1.361   0.862
#
# The knee is sharp and it is not where a safety factor would put it: below
# 1.25 the estimate undershoots the window's lower edge -- which is top_k
# itself, so there is no margin under it -- and every row that lands short
# pays a full redo. 28% of rows redoing costs 2.3x.
SAMPLED_MIN_VOCAB_PER_TOPK = int(
    os.environ.get("FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO", "64")
)
SSTRIDE = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_SSTRIDE", "16"))
TARGET_MULT = float(os.environ.get("FLAGGEMS_HYGON_PREFILL_TARGET_MULT", "1.25"))
CAP_MULT = 4  # candidate buffer; the acceptance window is [top_k, CAP]
SBLOCK = 512
SWARPS = 8
SRADIX = 256
_MAX_CAND_ELEMS = 1 << 24

# The sample and the collect key off the operator's own 11-bit STEP-0 key; the
# retry in _s_finish keys off the full 32-bit one, because the 11-bit key
# collapses on a narrow band and the retry is what has to be exact.
_key11 = _generic._convert_to_trt_uint16_hi11
_key32 = _generic._convert_to_uint32


@triton.jit
def _s_scan(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin whose inclusive prefix reaches `target`, and that bin's
    exclusive prefix. Bin 0 holds the largest values."""
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    tb = tl.full([], NB - 1, tl.int32)
    lt = tl.zeros([], tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0) - c
        hit = (pre < target) & (pre + c >= target) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        candlt = tl.max(tl.where(hit, pre, 0), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            tb = cand
            lt = candlt
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    return tb, lt


@triton.jit
def _s_hist(
    logits_ptr, base, row, stride0, s, e, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    """Histogram every STRIDE-th TILE of [s, e). Tiles rather than strided
    elements: at SSTRIDE 8 a strided element sample is 32 bytes apart and so
    touches every other cache line, reading half the bytes for an eighth of
    the values. Tiles read exactly 1/STRIDE. The cost is that the sample is
    spatially clustered, which is only unbiased if the row has no spatial
    structure -- true of the benchmark's iid inputs, and the reason the
    fallback below is not optional."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(e - s, BLOCK * STRIDE)):
        i = s + t * BLOCK * STRIDE + lane
        m = i < e
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key11(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _s_prepare(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TARGET: tl.constexpr,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero, sample and threshold, one program per row -- so the histogram is
    this program's alone and a barrier is all the ordering needed."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    _s_hist(logits_ptr, base, row, stride0, s, e, STRIDE, BLOCK)
    tl.debug_barrier()
    # Take the WHOLE boundary bin (+1, exclusive): coarser estimates should
    # over-collect, since falling short of top_k forces the exact retry while
    # overshooting only costs a slightly larger ranking.
    tb, _ = _s_scan(base, tl.cdiv(TARGET, STRIDE), NB, BLOCK)
    tl.store(thr_ptr + row, tb + 1)


@triton.jit
def _s_collect(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """One pass: append every element strictly better than the threshold bin.
    Indices are stored relative to row_start, which is what the operator
    returns. Only the INDEX is stored: the values are re-read from global in
    _s_finish, in one short parallel pass over the candidates. On S5000 the
    two scattered stores per hit were the largest remaining cost of the MTT
    version of this pass, and dropping the value store netted 8.6 us after the
    re-read; here collect measured 126.7 us against the generic's ~86 for the
    same bytes.

    The bulk loop is UNMASKED and the remainder is handled separately, which
    is how the generic operator writes its own passes. Masking every iteration
    instead measured 131.3 us here against a modelled 55 -- 252 GB/s where the
    generic's collection pass reaches ~1270, essentially read speed -- and a
    mask on every load is the one structural difference between them.
    """
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    thr = tl.load(thr_ptr + row)
    base = logits_ptr + row * stride0 + s
    lane = tl.arange(0, BLOCK)
    off = lane[:, None] * VEC + tl.arange(0, VEC)[None, :]
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    ones1 = tl.full([BLOCK], 1, tl.int32)
    cnt2 = cnt_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    cnt1 = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)

    n_vec = span // (BLOCK * VEC)
    for t in tl.range(0, n_vec):
        i = t * BLOCK * VEC + off
        x = tl.load(base + i)
        # Cast explicitly: the key is uint32 and thr int32, and leaving that
        # promotion implicit selects every element (the MTT override records
        # the same bug).
        take = _key11(x).to(tl.int32) < thr
        pos = tl.atomic_add(cnt2, ones2, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)

    tail = n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(span - tail, BLOCK)):
        i = tail + t * BLOCK + lane
        m = i < span
        x = tl.load(base + i, mask=m, other=0.0)
        take = m & (_key11(x).to(tl.int32) < thr)
        pos = tl.atomic_add(cnt1, ones1, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)


@triton.jit
def _s_finish(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    out_ptr,
    counts_ptr,
    slot_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The retry decision and the exact answer, one program per row.

    A sample can under- or overshoot, so a row whose collected count falls
    outside [TOPK, CAP] is redone here -- over the FULL 32-bit ordered key,
    not the 11-bit one the sample and the collect use, because that key
    collapses on a narrow band and the overflow guarantee goes with it.

    Rows inside the window take the exact top-k of their candidates: four
    8-bit radix rounds over the same 32-bit key.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    c = tl.load(cnt_ptr + row)
    if (c < tl.minimum(TOPK, span)) | (c > CAP):
        # The 11-bit fp16 key can COLLAPSE: a row whose values sit in a narrow
        # band away from zero (relative spread below about 1%, the key's
        # resolution being magnitude/32) maps to one or two bins, and then
        # "an overflow can only drop what shares the k-th element's key" is
        # true but vacuous -- everything shares it, so the true top-k can be
        # dropped. The generic operator escapes through STEP 1-3, which refine
        # over the full 32 bits; this path has no STEP 1-3, so the redo uses
        # the full 32-bit ordered key directly. It is injective on distinct
        # floats, so only exact ties can ever be dropped. Same fix, same
        # reason, as top_k_per_row_decode's.
        obase_r = out_ptr + row * TOPK
        cbase_r = counts_ptr + row * RADIX
        rbase = logits_ptr + row * stride0 + s
        rdesired = tl.zeros((), dtype=tl.uint32)
        rmask = tl.zeros((), dtype=tl.uint32)
        r_to_find = TOPK + 1
        row_tiles = tl.cdiv(span, BLOCK)
        for rdpos in tl.static_range(24, -1, -8):
            if r_to_find > 1:
                tl.store(cbase_r + bins, tl.zeros([RADIX], tl.int32))
                tl.debug_barrier()
                for rt in tl.range(0, row_tiles):
                    ri = rt * BLOCK + lane
                    rvalid = ri < span
                    rkey = _key32(tl.load(rbase + ri, mask=rvalid, other=0.0))
                    rdigit = ((rkey >> rdpos) & (RADIX - 1)).to(tl.int32)
                    tl.atomic_add(
                        cbase_r + rdigit,
                        ones,
                        mask=rvalid & ((rkey & rmask) == rdesired),
                        sem="relaxed",
                        scope="cta",
                    )
                tl.debug_barrier()
                rcounts = tl.load(cbase_r + bins)
                rprefix = tl.cumsum(rcounts, axis=0) - rcounts
                rhit = (rprefix < r_to_find) & (rprefix + rcounts >= r_to_find)
                rb0 = tl.min(tl.where(rhit, bins, RADIX), axis=0).to(tl.int32)
                rb0 = tl.where(rb0 == RADIX, RADIX - 1, rb0)
                rlt = tl.max(tl.where(bins == rb0, rprefix, 0), axis=0).to(tl.int32)
                rdesired = rdesired | (rb0.to(tl.uint32) << rdpos)
                rmask = rmask | (tl.full((), RADIX - 1, tl.uint32) << rdpos)
                r_to_find = r_to_find - rlt
        rthr = rdesired
        tl.store(slot_ptr + row, 0)
        tl.debug_barrier()
        rslots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
        # strictly better than the k-th, then its exact ties
        for req in tl.static_range(2):
            for rt2 in tl.range(0, row_tiles):
                ri2 = rt2 * BLOCK + lane
                rvalid2 = ri2 < span
                rkey2 = _key32(tl.load(rbase + ri2, mask=rvalid2, other=0.0))
                if req == 0:
                    rtake = rvalid2 & (rkey2 < rthr)
                else:
                    rtake = rvalid2 & (rkey2 == rthr)
                rq = tl.atomic_add(rslots, ones, mask=rtake, sem="relaxed", scope="cta")
                tl.store(obase_r + rq, ri2.to(tl.int32), mask=rtake & (rq < TOPK))
            tl.debug_barrier()
        # a row shorter than TOPK leaves the rest of the output padded
        rfilled = tl.load(slot_ptr + row)
        for rp in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            rj = rp * BLOCK + lane
            tl.store(obase_r + rj, -1, mask=(rj >= rfilled) & (rj < TOPK))
        return

    n = tl.minimum(tl.load(cnt_ptr + row), CAP)
    vbase = cand_val_ptr + row * CAP
    ibase = cand_idx_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX
    tiles = tl.cdiv(n, BLOCK)

    # _s_collect stores indices only; gather the candidate values back from
    # global memory. The retry above stores values as well, so this re-read is
    # redundant there -- but correct, and the retry does not fire in practice.
    # The barrier is required: this program loads from vbase right after
    # storing to it.
    row_base = logits_ptr + row * stride0 + s
    for t in tl.range(0, tiles):
        pos = t * BLOCK + lane
        valid = pos < n
        ci = tl.load(ibase + pos, mask=valid, other=0)
        tl.store(vbase + pos, tl.load(row_base + ci, mask=valid, other=0.0), mask=valid)
    tl.debug_barrier()

    if n <= TOPK:
        for ft in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = ft * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < n, other=-1)
            tl.store(obase + j, tl.where(j < n, idx, -1), mask=j < TOPK)
        return

    desired = tl.zeros((), tl.uint32)
    desired_mask = tl.zeros((), tl.uint32)
    k_to_find = TOPK + 1
    for digit_pos in tl.static_range(24, -1, -8):
        if k_to_find > 1:
            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            cnts = tl.load(cbase + bins)
            prefix = tl.cumsum(cnts, axis=0) - cnts
            hit = (prefix < k_to_find) & (prefix + cnts >= k_to_find)
            rb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            rb = tl.where(rb == RADIX, RADIX - 1, rb)
            lt = tl.max(tl.where(bins == rb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (rb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - lt

    thr_key = desired
    tl.store(slot_ptr + row, 0)
    tl.debug_barrier()
    slots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            q = tl.atomic_add(slots, ones, mask=take, sem="relaxed", scope="cta")
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))
        tl.debug_barrier()


class _SLaunch:
    """One kernel: JIT on first use, direct afterwards. Same recipe as the
    decode override's; copied so the two operators stay independent."""

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


class _SPlan:
    """Buffers and the three launches for one sampled shape."""

    def __init__(self, dev, dtype, num_rows, vocab, top_k):
        import torch

        cap = max(SBLOCK, triton.next_power_of_2(top_k * CAP_MULT))
        self.cap = cap
        nb = _generic.NUM_BINS
        self.hist = torch.empty((num_rows, nb), dtype=torch.int32, device=dev)
        self.thr = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cnt = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cand_idx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)
        self.cand_val = torch.empty((num_rows, cap), dtype=dtype, device=dev)
        self.counts = torch.empty((num_rows, SRADIX), dtype=torch.int32, device=dev)
        self.slot = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.prepare = _SLaunch(
            _s_prepare,
            (num_rows,),
            {
                "TARGET": int(top_k * TARGET_MULT),
                "NB": nb,
                "STRIDE": SSTRIDE,
                "BLOCK": SBLOCK,
            },
            SWARPS,
        )
        self.collect = _SLaunch(
            _s_collect,
            (num_rows,),
            {"CAP": cap, "BLOCK": SBLOCK, "VEC": 4},
            SWARPS,
        )
        self.finish = _SLaunch(
            _s_finish,
            (num_rows,),
            {"TOPK": top_k, "NB": nb, "CAP": cap, "RADIX": SRADIX, "BLOCK": SBLOCK},
            SWARPS,
        )

    def run(self, logits, starts, ends, indices, stride0):
        self.prepare(logits, starts, ends, self.hist, self.thr, self.cnt, stride0)
        self.collect(
            logits,
            starts,
            ends,
            self.thr,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            stride0,
        )
        self.finish(
            logits,
            starts,
            ends,
            self.hist,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            indices,
            self.counts,
            self.slot,
            stride0,
        )


_SPLANS = {}
_SPLANS_MAX = 8
_SPLAN_LOCK = threading.Lock()


def _s_aligned(t):
    return t.data_ptr() % 16 == 0


def _can_sample(logits, row_starts, row_ends, num_rows, stride0, stride1, top_k):
    import torch

    vocab = logits.shape[1]
    return (
        SAMPLED_MIN_VOCAB_PER_TOPK > 0
        and vocab >= SAMPLED_MIN_VOCAB_PER_TOPK * top_k
        and stride1 == 1
        and num_rows > 0
        and num_rows == logits.shape[0]
        and logits.dtype == torch.float32
        and row_starts.dtype == torch.int32
        and row_ends.dtype == torch.int32
        and not getattr(_generic, "HAS_TLE", False)
        and num_rows * triton.next_power_of_2(top_k * CAP_MULT) <= (1 << 24)
    )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Dense rows through the prefix-sum copy, everything else through generic,
    each launched at the geometry its occupancy wants."""
    if _can_sample(logits, row_starts, row_ends, num_rows, stride0, stride1, top_k):
        skey = (
            logits.device,
            num_rows,
            logits.shape[1],
            top_k,
            stride0,
            _s_aligned(logits),
            _s_aligned(row_starts),
            _s_aligned(row_ends),
            _s_aligned(indices),
        )
        with _SPLAN_LOCK:
            plan = _SPLANS.get(skey)
            if plan is None:
                if len(_SPLANS) >= _SPLANS_MAX:
                    _SPLANS.pop(next(iter(_SPLANS)))
                plan = _SPLANS[skey] = _SPlan(
                    logits.device, logits.dtype, num_rows, logits.shape[1], top_k
                )
            plan.run(logits, row_starts, row_ends, indices, stride0)
        return indices

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
