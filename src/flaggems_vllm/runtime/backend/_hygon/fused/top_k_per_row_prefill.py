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

"""Hygon BW1000 top_k_per_row_prefill, routed per call.

Very sparse rows (vocab >= 64 * top_k) take a sampled threshold. Dense rows
(vocab <= 10 * top_k) run copies of the generic kernel with a prefix-sum slot
allocator and a VEC=2 layout; the large dense shapes take a one-read sampled
kernel instead, with that copy as its retry. Everything else runs the generic
kernel with one threshold scan.

Each route is a separate copy of the generic module patched as source text:
Triton binds a kernel's globals at compile time, so one module can hold only
one _process_bins. A patch whose anchor is not found exactly once skips its
route with a warning, and the generic kernel runs.
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

# Dense iff vocab_size <= DENSE_VOCAB_PER_TOPK * top_k, i.e. density >= 10%. A
# prefix sum over the take mask beats one atomic per selected element above a
# measured ~9.4% (48 of 512).
DENSE_VOCAB_PER_TOPK = 10


# ---------------------------------------------------------------------------
# One threshold scan instead of a carried chain of rounds: one vectorised
# clear, one RADIX_SIZE-wide cumsum, the bin and its size as reductions kept in
# registers. Measured on the operator at production routing: 1.014x-1.309x on
# the seven benchmark shapes, geomean 1.094x. Applied as two exact text
# replacements; if either is not found exactly once the copies load the
# generic source unchanged.
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
# Launch geometry by occupancy: past one row per SM the grid is the
# parallelism, so many rows want narrow programs. Ratio vs vLLM:
#
#   rows/SM   row 4096, k 512        row 129280, k 1024
#   < 4       all within noise       B512 w8 best
#   4 - 16    B512 w4: +3..16%       B512 w4: +3..10%
#   32 - 52   B256 w2: +47..57%      B256 w4: +13..32%
#   204       B256 w2: 1.95x         --
#
# num_warps=1 returned wrong answers and is never used. NUM_THREADS_PER_BLOCK
# and _num_warps are host-side globals read at launch, so they are set per
# call under a lock. FLAGGEMS_HYGON_TOPK_GEOMETRY=0 keeps generic's values.

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
# A sampled threshold for very sparse rows. The generic step reads each row
# twice, the first time only to find the threshold; here that pass reads every
# SSTRIDE-th tile and aims at TARGET_MULT * top_k, so one pass collects a
# superset that _s_finish ranks exactly. Benchmark SpeedUp on (64,129280)
# against the same binary with the gate shut:
#
#     TARGET_MULT   1.00    1.10    1.25    1.50    3.00
#     rows outside  28.1%   10.9%    1.6%    0.0%     --
#     vs gate shut  0.434   0.494   1.402   1.361   0.862
#
# Below 1.25 the estimate undershoots top_k and every short row pays a redo.
SAMPLED_MIN_VOCAB_PER_TOPK = int(
    os.environ.get("FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO", "64")
)
SSTRIDE = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_SSTRIDE", "16"))
TARGET_MULT = float(os.environ.get("FLAGGEMS_HYGON_PREFILL_TARGET_MULT", "1.25"))
CAP_MULT = 4  # candidate buffer; the acceptance window is [top_k, CAP]
SBLOCK = 512
SWARPS = 8
SRADIX = 256
# Programs per row in the collect pass, each with its own counter and segment:
# a partially-masked atomic to one address costs ~12 ns per taken lane here,
# serialised, so a shared counter queued every program of the row. Benchmark
# SpeedUp on (64,129280):
#
#     split                 2       4       8      16
#     shared counter      0.631   0.605   0.569   0.531
#     private counters    0.702   0.736   0.634   0.728
#
# 8 is slow in both designs, for a reason not yet understood. A power of two,
# because prepare zeroes a row's counters with one arange.
_ssplit = max(1, int(os.environ.get("FLAGGEMS_HYGON_PREFILL_SSPLIT", "4")))
SSPLIT = 1 << (_ssplit.bit_length() - 1)
_MAX_CAND_ELEMS = 1 << 24


def _s_geometry(vocab, top_k):
    """(CAP, CHUNK, SEG) for a sampled plan. A segment holds min(CAP, CHUNK),
    so it can overflow only when the row already exceeds CAP; CHUNK's 2048
    granularity can leave programs idle, which CAP // SSPLIT did not survive."""
    cap = max(SBLOCK, triton.next_power_of_2(top_k * CAP_MULT))
    chunk = triton.cdiv(triton.cdiv(vocab, SSPLIT), SBLOCK * 4) * SBLOCK * 4
    return cap, chunk, min(cap, chunk)


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
    """Histogram every STRIDE-th TILE of [s, e): whole tiles read 1/STRIDE of
    the bytes, where strided elements would touch every cache line. The sample is
    unbiased only for rows without spatial structure; _s_finish's exact retry
    covers the rest."""
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
    SPLIT: tl.constexpr,
):
    """Zero, sample and threshold, one program per row -- so the histogram is
    this program's alone and a barrier is all the ordering needed."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row * SPLIT + tl.arange(0, SPLIT), tl.zeros([SPLIT], tl.int32))
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
    SPLIT: tl.constexpr,
    CHUNK: tl.constexpr,
    SEG: tl.constexpr,
):
    """Append every element strictly better than the threshold bin, as an
    index relative to row_start; _s_finish re-reads the values.

    The bulk loop is unmasked and the remainder handled separately, as in the
    generic passes: a mask on every load cost 131.3 us against a modelled 55.
    SPLIT programs divide the row, each appending through its own counter into its
    own SEG-long segment. CHUNK is a multiple of BLOCK * VEC so the bulk loop stays
    unmasked; the last part runs to the end of the row regardless.
    """
    pid = tl.program_id(0)
    row = pid // SPLIT
    part = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    thr = tl.load(thr_ptr + row)
    base = logits_ptr + row * stride0 + s
    lane = tl.arange(0, BLOCK)
    off = lane[:, None] * VEC + tl.arange(0, VEC)[None, :]
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    ones1 = tl.full([BLOCK], 1, tl.int32)
    cnt2 = cnt_ptr + pid + tl.zeros([BLOCK, VEC], tl.int32)
    cnt1 = cnt_ptr + pid + tl.zeros([BLOCK], tl.int32)

    start = part * CHUNK
    stop = tl.minimum(start + CHUNK, span)
    stop = tl.where(part == SPLIT - 1, span, stop)
    have = tl.maximum(stop - start, 0)

    n_vec = have // (BLOCK * VEC)
    # Two stages hide load latency inside a wave. Benchmark SpeedUp on
    # (64,129280) at one program per row, stages 1/2/3/4: 0.582 / 0.600 / 0.542 /
    # 0.544; deeper stages cost registers, and so waves.
    for t in tl.range(0, n_vec, num_stages=2):
        i = start + t * BLOCK * VEC + off
        x = tl.load(base + i)
        # Cast explicitly: the key is uint32 and thr int32, and leaving that
        # promotion implicit selects every element (the MTT override records
        # the same bug).
        take = _key11(x).to(tl.int32) < thr
        pos = tl.atomic_add(cnt2, ones2, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < SEG)
        tl.store(cand_idx_ptr + pid * SEG + pos, i.to(tl.int32), mask=keep)

    tail = start + n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(tl.maximum(stop - tail, 0), BLOCK)):
        i = tail + t * BLOCK + lane
        m = i < stop
        x = tl.load(base + i, mask=m, other=0.0)
        take = m & (_key11(x).to(tl.int32) < thr)
        pos = tl.atomic_add(cnt1, ones1, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < SEG)
        tl.store(cand_idx_ptr + pid * SEG + pos, i.to(tl.int32), mask=keep)


@triton.jit
def _s_finish(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    cidx_ptr,
    out_ptr,
    counts_ptr,
    slot_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT: tl.constexpr,
    SEG: tl.constexpr,
):
    """The retry decision and the exact answer, one program per row.

    A row outside [TOPK, CAP], or with a full segment, is redone over the full
    32-bit ordered key; the 11-bit key the sample and the collect use can collapse
    on a narrow band. Rows inside take the exact top-k of their candidates: four
    8-bit radix rounds over the same 32-bit key.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    c = tl.zeros((), tl.int32)
    over = tl.zeros((), tl.int32)
    for sg in tl.static_range(SPLIT):
        craw = tl.load(cnt_ptr + row * SPLIT + sg)
        c += tl.minimum(craw, SEG)
        over += (craw > SEG).to(tl.int32)
    if (c < tl.minimum(TOPK, span)) | (c > CAP) | (over > 0):
        # The 11-bit fp16 key resolves magnitude/32, so a row in a narrow band away
        # from zero collapses into one or two bins and an overflow can drop the true
        # top-k. This path has no STEP 1-3 to refine through, so the redo ranks the
        # full 32-bit ordered key, which is injective on distinct floats. Same fix as
        # top_k_per_row_decode's.
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

    ibase = cidx_ptr + row * CAP
    vbase = cand_val_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX

    # _s_collect stores indices only, in per-program segments: compact them and
    # gather each candidate's value on the way. The barrier is required; this
    # program reads vbase and ibase right after storing them.
    row_base = logits_ptr + row * stride0 + s
    n = tl.zeros((), tl.int32)
    for sg in tl.static_range(SPLIT):
        cseg = tl.minimum(tl.load(cnt_ptr + row * SPLIT + sg), SEG)
        sbase = cand_idx_ptr + (row * SPLIT + sg) * SEG
        for t in tl.range(0, tl.cdiv(cseg, BLOCK)):
            p = t * BLOCK + lane
            pv = p < cseg
            ci = tl.load(sbase + p, mask=pv, other=0)
            tl.store(ibase + n + p, ci, mask=pv)
            tl.store(vbase + n + p, tl.load(row_base + ci, mask=pv, other=0.0), mask=pv)
        n += cseg
    tiles = tl.cdiv(n, BLOCK)
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
    # One program owns the row, so output positions need no atomic: a running
    # offset plus an exclusive prefix over the take mask. The slot atomic cost
    # 31.7 of finish's 68.1 us on (64,129280).
    filled = tl.zeros((), tl.int32)
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            ti = take.to(tl.int32)
            q = filled + tl.cumsum(ti, axis=0) - ti
            filled += tl.sum(ti, axis=0)
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))


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

        cap, schunk, seg = _s_geometry(vocab, top_k)
        self.cap = cap
        nb = _generic.NUM_BINS
        self.hist = torch.empty((num_rows, nb), dtype=torch.int32, device=dev)
        self.thr = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cnt = torch.empty((num_rows * SSPLIT,), dtype=torch.int32, device=dev)
        self.cand_idx = torch.empty(
            (num_rows, SSPLIT * seg), dtype=torch.int32, device=dev
        )
        self.cand_val = torch.empty((num_rows, cap), dtype=dtype, device=dev)
        self.cidx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)
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
                "SPLIT": SSPLIT,
            },
            SWARPS,
        )
        self.collect = _SLaunch(
            _s_collect,
            (num_rows * SSPLIT,),
            {
                "CAP": cap,
                "BLOCK": SBLOCK,
                "VEC": 4,
                "SPLIT": SSPLIT,
                "CHUNK": schunk,
                "SEG": seg,
            },
            SWARPS,
        )
        self.finish = _SLaunch(
            _s_finish,
            (num_rows,),
            {
                "TOPK": top_k,
                "NB": nb,
                "CAP": cap,
                "RADIX": SRADIX,
                "BLOCK": SBLOCK,
                "SPLIT": SSPLIT,
                "SEG": seg,
            },
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
            self.cidx,
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
        and num_rows * SSPLIT * _s_geometry(vocab, top_k)[2] <= _MAX_CAND_ELEMS
        and num_rows * _s_geometry(vocab, top_k)[0] <= _MAX_CAND_ELEMS
    )


# ---------------------------------------------------------------------------
# One read for the large dense shapes.
#
# The dense copy reads each row twice and fires one global atomic per element
# for its histogram. On 12961x4100 (895 us) the atomics cost ~474 us, bound by
# the distinct addresses they touch; the second read ~184 us, not served by L2;
# the first read already runs at 88% of this card's 1302 GB/s. This route reads
# the row once and issues no per-element atomic: two thresholds from a
# 512-element sample, one pass that writes the sure set straight out and keeps
# the band between the thresholds, then the missing top_k - S from the band by
# bitwise lifting on the 32-bit ordered key. At one warp per program every
# reduction and scan stays inside a wave; two warps were 1.34x slower. do_bench,
# us, before the retry below:
#
#                  dense copy   this route
#     16383x4095      1110          615
#     12961x4100       890          524
#     16380x5115      1343          687
#
# A row the sample misjudges -- sure set past top_k, band short of it, or band
# over DS_BCAP -- is flagged: 0.5-1.1% of standard-normal rows, every row of a
# narrow band. The dense copy then runs for every row and returns at once
# unless its row was flagged, so the answer is always that copy's or exact.
# 70/150 flags only 0.06-0.2% but needs a 1024-slot band, and was 1.7x slower.
DENSE_SAMPLED = os.environ.get(
    "FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED", "1"
).strip().lower() not in ("0", "false", "off", "no")
DS_BLOCK = 512
DS_NS = 512
DS_BCAP = 512
DS_HI = 75  # percent of top_k expected above T_hi: surely in
DS_LO = 135  # percent of top_k expected above T_lo: the band's end
_DENSE_RETRY_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_dense_retry"


@triton.jit
def _d_kth(keys, valid, r):
    """The r-th smallest 11-bit key (1-based) among the valid lanes."""
    res = tl.zeros((), tl.int32)
    for b in tl.static_range(10, -1, -1):
        probe = res | (1 << b)
        cnt = tl.sum((valid & (keys < probe)).to(tl.int32), axis=0)
        res = tl.where(cnt < r, probe, res)
    return res


@triton.jit
def _d_classify(
    x, i, m, t_hi, t_lo, S, B, obase, kb, ib, TOPK: tl.constexpr, BCAP: tl.constexpr
):
    k = _key11(x).to(tl.int32)
    sure = m & (k < t_hi)
    band = m & (k >= t_hi) & (k < t_lo)
    # both slot positions from one scan: the sure count in the low 16 bits
    packed = sure.to(tl.int32) + (band.to(tl.int32) << 16)
    cs = tl.cumsum(packed, axis=0) - packed
    tot = tl.sum(packed, axis=0)
    ps = S + (cs & 0xFFFF)
    pb = B + (cs >> 16)
    tl.store(obase + ps, i, mask=sure & (ps < TOPK))
    keep = band & (pb < BCAP)
    tl.store(kb + pb, _key32(x).to(tl.int32, bitcast=True), mask=keep)
    tl.store(ib + pb, i, mask=keep)
    return S + (tot & 0xFFFF), B + (tot >> 16)


@triton.jit
def _d_sampled(
    x_ptr,
    starts_ptr,
    ends_ptr,
    out_ptr,
    bkey_ptr,
    bidx_ptr,
    flag_ptr,
    stride0,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    NS: tl.constexpr,
    BCAP: tl.constexpr,
    HI: tl.constexpr,
    LO: tl.constexpr,
):
    """One program per row: writes the row's top-k and a flag of 0, or a flag
    of 1 and leaves the row to the dense copy."""
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    base = x_ptr + row * stride0 + s

    CH: tl.constexpr = NS // 8
    sl = tl.arange(0, NS)
    si = (sl // CH) * (span // 8) + (sl % CH)
    sv = si < span
    sk = _key11(tl.load(base + si, mask=sv, other=float("-inf"))).to(tl.int32)
    ns = tl.sum(sv.to(tl.int32), axis=0)
    expect = TOPK * ns.to(tl.float32) / tl.maximum(span, 1).to(tl.float32)
    t_hi = _d_kth(sk, sv, (expect * HI / 100).to(tl.int32))
    t_lo = _d_kth(sk, sv, (expect * LO / 100).to(tl.int32) + 1) + 1

    lane = tl.arange(0, BLOCK)
    obase = out_ptr + row * TOPK
    kb = bkey_ptr + row * BCAP
    ib = bidx_ptr + row * BCAP
    S = tl.zeros((), tl.int32)
    B = tl.zeros((), tl.int32)
    n_full = span // BLOCK
    for t in tl.range(0, n_full):
        i = t * BLOCK + lane
        S, B = _d_classify(
            tl.load(base + i), i, i >= 0, t_hi, t_lo, S, B, obase, kb, ib, TOPK, BCAP
        )
    i = n_full * BLOCK + lane
    m = i < span
    x = tl.load(base + i, mask=m, other=float("-inf"))
    S, B = _d_classify(x, i, m, t_hi, t_lo, S, B, obase, kb, ib, TOPK, BCAP)

    need = TOPK - S
    good = (S <= TOPK) & (need <= B) & (B <= BCAP)
    tl.store(flag_ptr + row, 1 - good.to(tl.int32))
    # the band select reads back what this program just stored
    tl.debug_barrier()
    if good:
        q = tl.arange(0, BCAP)
        bv = q < B
        bk = tl.load(kb + q, mask=bv, other=0).to(tl.uint32, bitcast=True)
        # Lift only the bits where the band's keys differ. The highest one is
        # the float exponent of min ^ max, which can round up, never down.
        one = tl.full((), 1, tl.uint32)
        kmin = tl.min(tl.where(bv, bk, tl.full([BCAP], 0xFFFFFFFF, tl.uint32)), axis=0)
        kmax = tl.max(tl.where(bv, bk, tl.zeros([BCAP], tl.uint32)), axis=0)
        hb = ((kmin ^ kmax).to(tl.float32).to(tl.int32, bitcast=True) >> 23) - 127
        hb = tl.minimum(tl.maximum(hb, 0), 31)
        nb = hb + 1
        low = tl.where(
            nb >= 32,
            tl.full((), 0xFFFFFFFF, tl.uint32),
            (one << nb.to(tl.uint32)) - one,
        )
        kth = kmin & ~low
        for j in tl.range(0, nb):
            probe = kth | (one << (hb - j).to(tl.uint32))
            cnt = tl.sum((bv & (bk < probe)).to(tl.int32), axis=0)
            kth = tl.where(cnt < need, probe, kth)
        idx = tl.load(ib + q, mask=bv, other=0)
        lt = bv & (bk < kth)
        lti = lt.to(tl.int32)
        nlt = tl.sum(lti, axis=0)
        tl.store(obase + S + tl.cumsum(lti, axis=0) - lti, idx, mask=lt)
        eq = bv & (bk == kth)
        eqi = eq.to(tl.int32)
        pe = S + nlt + tl.cumsum(eqi, axis=0) - eqi
        tl.store(obase + pe, idx, mask=eq & (pe < TOPK))


def _in_function(source, name, old, new):
    import ast

    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    lines = source.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    a = sum(map(len, lines[: first - 1]))
    b = sum(map(len, lines[: node.end_lineno]))
    body = source[a:b]
    if body.count(old) != 1:
        raise ValueError(f"{name}: anchor found {body.count(old)} times")
    return source[:a] + body.replace(old, new, 1) + source[b:]


def _dense_retry_path():
    """The dense final copy, each program returning at once unless _d_sampled
    flagged its row."""
    if not DENSE_SAMPLED or _VEC2_FINAL_PATH is None:
        return None
    try:
        with open(_VEC2_FINAL_PATH) as fh:
            source = fh.read()
        kernel = "non_tle_top_k_per_row_prefill"
        source = _in_function(
            source,
            kernel,
            "    s_found_topk_values_ptr,\n    TOPK: tl.constexpr,\n",
            "    s_found_topk_values_ptr,\n    skip_ptr,\n    TOPK: tl.constexpr,\n",
        )
        source = _in_function(
            source,
            kernel,
            "    row_id = tl.program_id(0) + ROW_OFFSET\n",
            "    row_id = tl.program_id(0) + ROW_OFFSET\n"
            "    if tl.load(skip_ptr + row_id) == 0:\n"
            "        return\n",
        )
        compile(source, "<hygon-prefill-dense-retry>", "exec")
        base = _private_dir()
        if base is None:
            return None
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        path = os.path.join(base, f"top_k_per_row_prefill_dense_retry_{digest}.py")
        if not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(dir=base, suffix=".py")
            with os.fdopen(fd, "w") as fh:
                fh.write(source)
            os.replace(tmp, path)
        with open(path) as fh:
            if fh.read() != source:
                return None
        return path
    except Exception as exc:  # noqa: BLE001 - keep the dense copy
        _log.warning("hygon prefill dense sampled route skipped: %r", exc)
        return None


_DENSE_RETRY_PATH = _dense_retry_path()
try:
    _dense_retry = (
        _load_copy(_DENSE_RETRY_NAME, _DENSE_RETRY_PATH) if _DENSE_RETRY_PATH else None
    )
except Exception as exc:  # noqa: BLE001 - keep the dense copy
    _log.warning("hygon prefill dense retry module skipped: %r", exc)
    _dense_retry = None
if _dense_retry is not None:
    _GENERIC_DEFAULTS[id(_dense_retry)] = (
        _dense_retry.NUM_THREADS_PER_BLOCK,
        _dense_retry._num_warps,
    )


class _DPlan:
    """Buffers and the sampled launch for one dense shape."""

    def __init__(self, dev, num_rows, top_k):
        self.bkey = torch.empty((num_rows, DS_BCAP), dtype=torch.int32, device=dev)
        self.bidx = torch.empty((num_rows, DS_BCAP), dtype=torch.int32, device=dev)
        self.flags = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.launch = _SLaunch(
            _d_sampled,
            (num_rows,),
            {
                "TOPK": top_k,
                "BLOCK": DS_BLOCK,
                "NS": DS_NS,
                "BCAP": DS_BCAP,
                "HI": DS_HI,
                "LO": DS_LO,
            },
            1,
        )


_DPLANS = {}
_DPLAN_LOCK = threading.Lock()


def _can_dense_sample(logits, row_starts, row_ends, num_rows, stride1, top_k):
    vocab = logits.shape[1]
    return (
        _dense_retry is not None
        and top_k == 512
        and num_rows >= 8192
        and 2048 <= vocab <= 5120
        and stride1 == 1
        and num_rows == logits.shape[0]
        and logits.dtype == torch.float32
        and row_starts.dtype == torch.int32
        and row_ends.dtype == torch.int32
        and not getattr(_generic, "HAS_TLE", False)
        and not _dense_retry.HAS_TLE
        and num_rows * DS_BCAP <= _MAX_CAND_ELEMS
    )


def _dense_sampled(logits, row_starts, row_ends, indices, num_rows, stride0, top_k):
    dev = logits.device
    key = (
        dev,
        torch.cuda.current_stream(dev).cuda_stream,
        num_rows,
        logits.shape[1],
        top_k,
        stride0,
        _s_aligned(logits),
        _s_aligned(row_starts),
        _s_aligned(row_ends),
        _s_aligned(indices),
    )
    mod = _dense_retry
    geo = _geometry(num_rows, logits.shape[1]) if _GEOMETRY else None
    with _DPLAN_LOCK:
        plan = _DPLANS.get(key)
        if plan is None:
            if len(_DPLANS) >= _SPLANS_MAX:
                _DPLANS.pop(next(iter(_DPLANS)))
            plan = _DPLANS[key] = _DPlan(dev, num_rows, top_k)
        plan.launch(
            logits,
            row_starts,
            row_ends,
            indices,
            plan.bkey,
            plan.bidx,
            plan.flags,
            stride0,
        )
        with _LAUNCH_LOCK:
            if geo is None:
                mod.NUM_THREADS_PER_BLOCK, mod._num_warps = _GENERIC_DEFAULTS[id(mod)]
            else:
                block, warps = geo
                mod.NUM_THREADS_PER_BLOCK = block
                mod._num_warps = lambda block_size, w=warps: w
            scratch = _scratch_buffers(mod, dev, num_rows)
            mod.non_tle_top_k_per_row_prefill[(num_rows,)](
                logits,
                indices,
                row_starts,
                row_ends,
                stride0,
                1,
                logits.shape[1],
                *scratch,
                plan.flags,
                TOPK=top_k,
                BLOCK_SIZE=mod.NUM_THREADS_PER_BLOCK,
                ROW_OFFSET=0,
                num_warps=mod._num_warps(mod.NUM_THREADS_PER_BLOCK),
            )
    return indices


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

    if _can_dense_sample(logits, row_starts, row_ends, num_rows, stride1, top_k):
        return _dense_sampled(
            logits, row_starts, row_ends, indices, num_rows, stride0, top_k
        )

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
