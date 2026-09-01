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

"""Triton top_k_per_row_prefill for DeepSeek V4 prefill-phase topk selection.

Implement based on file python/tutorials/tle/deepseek_v32/01-topk_selector.py from repo
https://github.com/flagos-ai/FlagTree.git, align with vLLM implementation.

"""

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime

from flaggems_vllm.utils.triton_version_utils import has_triton_tle


_LAUNCH_GEOMETRY = None


def _launch_geometry():
    """(warp_size, max_threads_per_block) for this device, cached."""
    global _LAUNCH_GEOMETRY
    if _LAUNCH_GEOMETRY is None:
        warp, maxt = 32, 1024
        try:
            props = runtime.torch_device_fn.get_device_properties(0)
            warp = getattr(props, "warp_size", 0) or 32
            maxt = getattr(props, "max_threads_per_block", 0) or 1024
        except Exception:  # noqa: BLE001 - never let detection break dispatch
            pass
        _LAUNCH_GEOMETRY = (warp, maxt)
    return _LAUNCH_GEOMETRY


def _num_warps(block_size):
    """Warps needed to cover a BLOCK_SIZE-wide tile, within the thread ceiling.

    This used to be `block_size // 32`, which silently assumes a 32-lane warp.
    On MetaX C550 the warp is 64 lanes and the per-block ceiling is 512 threads,
    so BLOCK_SIZE=512 asked for 16 warps x 64 = 1024 threads and every launch
    failed with OutOfResources -- the op could not run on that card at all.

    Dividing by the real warp size is an identity on 32-lane parts (512 -> 16
    warps either way), so NVIDIA and Moore Threads are unchanged. The clamp
    matters where a tile is wider than the device can staff: the tile stays the
    same width and each thread simply covers more of it.
    """
    warp, maxt = _launch_geometry()
    return max(1, min(block_size // warp, maxt // warp))


def _vendor_tle_enabled() -> bool:
    """Does this backend actually support TLE, per its own VendorDescriptor?

    `has_triton_tle()` only proves the Python module imports. It does not prove
    the backend can LOWER tle.gpu.alloc. MetaX C550 is exactly that case: every
    tle symbol resolves, but compilation dies with

        'triton._C.libtriton.ir.builder' object has no attribute
        'make_swizzled_shared_encoding_attr'

    which took both ops from "slow" to "cannot run at all", when the non-TLE
    fallback would have worked fine.

    The VendorDescriptor already carries `tle_enabled`, and it is already correct
    -- nvidia/mthreads/enflame declare True, everyone else defaults False. It was
    simply never read by anything. Reading it here makes the non-TLE path the
    default for any backend that has not declared TLE support, which is the safe
    direction: that path is plain Triton over global scratch and works everywhere.

    FLAGGEMS_FORCE_TLE=1 overrides, so a vendor can test whether its TLE works
    without editing the descriptor.
    """
    override = os.environ.get("FLAGGEMS_FORCE_TLE")
    if override is not None:
        return override.lower() not in {"0", "false", "off", "no"}
    try:
        return bool(getattr(runtime.device.info, "tle_enabled", False))
    except Exception:  # noqa: BLE001 - never let detection break the import
        return False


if has_triton_tle(3, 6, 0) and _vendor_tle_enabled():
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


logger = logging.getLogger(__name__)

# Start of shared implementation code for top_k_per_row_decode and top_k_per_row_prefill

SORTING_ALGORITHM_THRESHOLD = 12288
SPLIT_WORK_THRESHOLD = 200 * 1000
NUM_THREADS_PER_BLOCK = 512
MULTIPLE_BLOCKS_PER_ROW_CONFIG = 10
NUM_THREADS_PER_BLOCK_MERGE = 1024


NUM_FILNAL_ITEMS = 2048
NUM_BINS = 2048
RADIX_BITS_FINAL = 8
RADIX_SIZE_FINAL = 1 << RADIX_BITS_FINAL
RADIX_FINAL_PREFILL_VOCAB_THRESHOLD = 65536


def _use_radix_final_for_prefill(vocab_size):
    return vocab_size >= RADIX_FINAL_PREFILL_VOCAB_THRESHOLD


def _tl_assume_supported() -> bool:
    """Can this backend round-trip `llvm.intr.assume`?

    `tl.assume` is a pure optimisation hint -- dropping it changes no result.
    The Ascend backend writes its IR to a file and parses it back, and its build
    has no custom assembly form for the op, so the round trip fails at
    ConvertLinalgRToBinary with

        error: custom op 'llvm.intr.assume' has no custom assembly form

    There is nothing to feature-detect: the symbol exists and traces fine, and
    only the backend's own serialisation rejects it. So this is keyed off the
    vendor and defaults to ON, leaving every already-validated backend emitting
    exactly what it emitted before.

    FLAGGEMS_TL_ASSUME=0/1 overrides, so a vendor can retest without editing
    this list.
    """
    override = os.environ.get("FLAGGEMS_TL_ASSUME")
    if override is not None:
        return override.lower() not in {"0", "false", "off", "no"}
    try:
        return getattr(runtime.device.info, "vendor_name", "") not in ("ascend",)
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return True


HAS_TL_ASSUME = _tl_assume_supported()


if HAS_TL_ASSUME:

    @triton.jit
    def _assume(cond):
        tl.assume(cond)

else:

    @triton.jit
    def _assume(cond):
        pass


def _atomic_return_reliable() -> bool:
    """Are this backend's per-lane atomic return values trustworthy?

    On Ascend they are not. Ten-line repro (tools/triton_smoke.py probes 12-14):
    512 lanes each add 1 to one counter, the counter correctly ends at 512, and
    the RETURNED values hold only 65 distinct numbers instead of 0..511. Used as
    store addresses those collide, and a masked store with duplicate lane
    addresses is silently dropped there -- exactly the shape of the failure: the
    histogram, whose accumulation ignores its return, comes out complete, while
    the output buffer gets 7 of 64 entries, every one of them valid.

    Not feature-detectable: the call compiles and the count is right, only the
    returned values are wrong. Vendor-keyed, defaulting to ON so every backend
    where this operator is already validated keeps the atomic path.
    FLAGGEMS_ATOMIC_RETURN=0/1 overrides for retesting.
    """
    override = os.environ.get("FLAGGEMS_ATOMIC_RETURN")
    if override is not None:
        return override.lower() not in {"0", "false", "off", "no"}
    try:
        return getattr(runtime.device.info, "vendor_name", "") not in ("ascend",)
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return True


HAS_ATOMIC_RETURN = _atomic_return_reliable()


if HAS_ATOMIC_RETURN:

    @triton.jit
    def _compact_pos(cnt_scalar_ptr, cnt_bcast_ptrs, ones, take):
        return tl.atomic_add(
            cnt_bcast_ptrs, ones, mask=take, sem="relaxed", scope="cta"
        )

else:

    @triton.jit
    def _compact_pos(cnt_scalar_ptr, cnt_bcast_ptrs, ones, take):
        """Same destinations, from a scan instead of the atomic's return.

        Each counter here is per-row and the grid is one program per row, so
        there is no cross-program contention to serialise -- the atomic was only
        ever providing unique offsets within a tile and a running base across
        tiles. An exclusive prefix sum gives the first, a read-modify-write of
        the scalar gives the second.

        Measured 5.3x slower than the atomic on Moore Threads, which is why this
        is gated rather than adopted: correct everywhere, worth paying for only
        where the atomic cannot be trusted.
        """
        t = take.to(tl.int32)
        if len(t.shape) == 2:
            # The vectorised tiles are [BLOCK_SIZE, VEC]. Reducing over axis 0
            # alone leaves a [VEC] block, and storing that through a scalar
            # pointer is
            #   'Value argument cannot be block type if pointer argument is not
            #    a block'
            # So do it in two levels: a prefix down each column, plus the total
            # of all columns to its left. That numbers the tile column-major,
            # which is an order, and an order is all uniqueness needs.
            col_tot = tl.sum(t, axis=0)
            col_excl = tl.cumsum(col_tot, axis=0) - col_tot
            excl = (tl.cumsum(t, axis=0) - t) + col_excl[None, :]
            total = tl.sum(col_tot, axis=0)
        else:
            excl = tl.cumsum(t, axis=0) - t
            total = tl.sum(t, axis=0)
        base = tl.load(cnt_scalar_ptr)
        tl.store(cnt_scalar_ptr, base + total)
        return base + excl


# tl.reduce_or does not exist in every Triton build. It is absent from the
# Ascend backend's 3.2.0, where its use below made both operators fail to
# compile at all:
#
#     AttributeError: module 'triton.language' has no attribute 'reduce_or'
#
# The call is a block-wide "did any lane find it", the same thing vLLM's CUDA
# does with __syncthreads_or(foundThreshold). A max over the mask as int32 says
# exactly that and is available everywhere.
#
# Defined conditionally rather than replaced outright so that any build which
# HAS reduce_or keeps emitting it: this is then a no-op on NVIDIA and Moore
# Threads, where the operator is already validated, and only changes backends
# that could not run at all.
#
# Same shape as the capability flags already used elsewhere in this repo --
# IS_GATHER_SUPPORTED in FLA/gdn2_native/chunk_intra.py, the
# make_tensor_descriptor probes in FLA/triton_ops_helper.py -- and the same
# family of problem as PR #686, which fixed an Ascend import failure caused by
# triton.language.math losing `pow`. That one was the math/libdevice module and
# was fixed by picking the right one per Triton version; this one is a core
# language builtin, so tl_extra_shim does not apply.
HAS_REDUCE_OR = hasattr(tl, "reduce_or")

if HAS_REDUCE_OR:

    @triton.jit
    def _block_any(mask):
        return tl.reduce_or(mask, axis=0)

else:

    @triton.jit
    def _block_any(mask):
        return tl.max(mask.to(tl.int32), axis=0) != 0


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _extract_bin_idx(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        # int32, not uint32. `mapped` is uint16 and the shift leaves 11 bits, so
        # the value range is 0..2047 either way -- but the Ascend backend cannot
        # lower the cast this produced:
        #
        #   'hivm.hir.vcast' op currently don't support cast
        #   uint32_t_to_uint64_t_rintmode
        #
        # and every consumer treats the result as an index or compares it after
        # an explicit .to(tl.int32) anyway, so the unsigned type bought nothing.
        bin_idx = (mapped >> 5).to(tl.int32)
    else:
        bits = _convert_to_uint32(x)
        # Every branch lands in int32, matching STEP 0 above. The masks leave 11
        # or 10 bits, so the range is 0..2047 and the signed type holds it
        # exactly. The uint32 form is what produced
        #
        #   'hivm.hir.vcast' op currently don't support cast
        #   uint32_t_to_uint64_t_rintmode
        #
        # on Ascend: the result is used as `s_histogram_ptr + bin_idx`, and
        # promoting an unsigned 32-bit offset to the 64-bit one pointer
        # arithmetic wants is the cast that backend cannot lower. Signed
        # promotes fine. The bit patterns are identical either way.
        if STEP == 1:
            bin_idx = (bits >> 21).to(tl.int32)
        elif STEP == 2:
            bin_idx = ((bits >> 10) & 0x7FF).to(tl.int32)
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = (bits & 0x3FF).to(tl.int32)
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _convert_to_trt_uint16_hi11(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    return (mapped >> 5).to(tl.int32)


@triton.jit
def _distribute_to_bins(
    logits,
    in_range,
    ones,
    logit_pattern,
    s_histogram_ptr,
    STEP: tl.constexpr,
):
    bin_idx, is_partial_match = _extract_bin_idx(
        logits,
        in_range,
        logit_pattern,
        STEP=STEP,
    )
    tl.atomic_add(
        s_histogram_ptr + bin_idx,
        ones,
        mask=is_partial_match,
        sem="relaxed",
        scope="cta",
    )


@triton.jit
def _process_bins(
    logits,
    in_range,
    ones,
    offs,  # row_start based
    found_topk_values_ptrs,
    final_cnt_ptrs,
    s_found_topk_values_ptr,
    s_final_cnt_ptr,
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
    out_pos_lt = _compact_pos(
        s_found_topk_values_ptr, found_topk_values_ptrs, ones, take_lt
    )
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
            final_pos = _compact_pos(
                s_final_cnt_ptr, final_cnt_ptrs, ones, take_eq_final
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
        # At STEP 3 every taken lane has bin_idx == threshold_bin_idx, so the
        # per-lane counter address is a single address and the scalar form is
        # exact.
        out_pos_eq = _compact_pos(
            s_histogram_ptr + threshold_bin_idx,
            s_histogram_ptr + bin_idx,
            ones,
            take_eq,
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


@triton.jit
def _process_histogram_step(
    logits_ptr,
    row_start,
    row_end,
    stride1,
    vocab_size,
    skip_elems,
    indices_ptr,
    logit_pattern,
    threshold_bin_idx,
    assume_aligned,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if STEP == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_histogram_ptr + clear_bins, 0)
    tl.debug_barrier()

    if STEP == 2:
        logit_pattern = (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 21
    elif STEP == 3:
        logit_pattern |= (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 10

    if assume_aligned:
        n_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            _distribute_to_bins(
                x,
                True,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
    elif stride1 == 1:
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            _distribute_to_bins(
                x_vec,
                True,
                ones_vec_2d,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            _distribute_to_bins(
                x,
                True,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + offs, mask=in_range, other=float("-inf")
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            _distribute_to_bins(
                x,
                in_range,
                ones,
                logit_pattern,
                s_histogram_ptr,
                STEP=STEP,
            )
    last_value = tl.load(s_found_topk_values_ptr)
    tl.debug_barrier()

    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
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
            found_round = _block_any(threshold_mask)
            threshold_found = found_round
            last_value = total_sum

    tl.debug_barrier()
    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
    use_final = final_bin_size <= NUM_FINAL_ITEMS
    write_directly = ((STEP == 0) & (final_bin_size <= NUM_FINAL_ITEMS)) | (STEP >= 1)

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if assume_aligned:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        n_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            _process_bins(
                x_vec,
                True,
                ones_vec_2d,
                offs,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            _process_bins(
                x,
                True,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    elif stride1 == 1:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            _process_bins(
                x_vec,
                True,
                ones_vec_2d,
                offs + skip_elems,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            _process_bins(
                x,
                True,
                ones,
                offs + skip_elems,
                found_ptrs,
                final_cnt_ptrs,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + offs, mask=in_range, other=float("-inf")
            )
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            _process_bins(
                x,
                in_range,
                ones,
                offs + skip_elems,
                found_ptrs,
                final_cnt_ptrs,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
                final_cnt_ptrs,
                s_found_topk_values_ptr,
                s_final_cnt_ptr,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    tl.debug_barrier()
    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx


@triton.jit
def _final_select_radix(
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL
    RADIX_MASK_FINAL: tl.constexpr = RADIX_SIZE_FINAL - 1
    DIGIT_START: tl.constexpr = 32 - RADIX_BITS_FINAL

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    bins = tl.arange(0, RADIX_SIZE_FINAL)

    s_radix_counts = tle.gpu.alloc(
        [RADIX_SIZE_FINAL],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0,))
    radix_count_vec_ptr = s_radix_count_ptr + bins
    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
    remain = tl.minimum(TOPK - base_idx, final_cnt)
    tl.debug_barrier()

    if remain > 0:
        desired = tl.zeros((), dtype=tl.uint32)
        desired_mask = tl.zeros((), dtype=tl.uint32)
        k_to_find = remain + 1

        for digit_pos in tl.static_range(DIGIT_START, -1, -RADIX_BITS_FINAL):
            if k_to_find > 1:
                tl.store(s_radix_count_ptr + lane, 0, mask=lane < RADIX_SIZE_FINAL)
                tl.debug_barrier()

                cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
                for t in tl.range(0, cnt_tiles):
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    x = tl.load(
                        s_final_logits_ptr + pos,
                        mask=valid,
                        other=0,
                    )
                    key = _convert_to_uint32(x)
                    matches = (key & desired_mask) == desired
                    digit = ((key >> digit_pos) & RADIX_MASK_FINAL).to(tl.int32)
                    take = valid & matches
                    tl.atomic_add(
                        s_radix_count_ptr + digit,
                        ones,
                        mask=take,
                        sem="relaxed",
                        scope="cta",
                    )

                tl.debug_barrier()
                counts = tl.load(radix_count_vec_ptr)
                prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)
                next_prefix_sum = prefix_sum + counts
                threshold_mask = (prefix_sum < k_to_find) & (
                    next_prefix_sum >= k_to_find
                )
                threshold_init = tl.full((), RADIX_SIZE_FINAL, dtype=tl.int32)
                threshold_bin = tl.min(
                    tl.where(threshold_mask, bins, threshold_init), axis=0
                ).to(tl.int32)
                threshold_bin = tl.where(
                    threshold_bin == RADIX_SIZE_FINAL,
                    RADIX_SIZE_FINAL - 1,
                    threshold_bin,
                )
                counts_lt = tl.max(
                    tl.where(bins == threshold_bin, prefix_sum, 0), axis=0
                ).to(tl.int32)

                desired = desired | (threshold_bin.to(tl.uint32) << digit_pos)
                desired_mask = desired_mask | (
                    tl.full((), RADIX_MASK_FINAL, dtype=tl.uint32) << digit_pos
                )
                k_to_find = k_to_find - counts_lt

        thr_key = desired
        found_ptrs = s_found_topk_values_ptr + zeros
        cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
        for t in tl.range(0, cnt_tiles):
            pos = t * BLOCK_SIZE + lane
            valid = pos < final_cnt
            idx = tl.load(s_histogram_ptr + pos, mask=valid, other=0)
            x = tl.load(
                s_final_logits_ptr + pos,
                mask=valid,
                other=0,
            )
            key = _convert_to_uint32(x)
            take_lt = valid & (key < thr_key)
            out_pos_gt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_gt,
                idx,
                mask=take_lt & (out_pos_gt < TOPK),
            )
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    s_out_logits_ptr + out_pos_gt,
                    x,
                    mask=take_lt & (out_pos_gt < TOPK),
                )

        tl.debug_barrier()
        cur = tl.load(s_found_topk_values_ptr)
        if cur < TOPK:
            for t in tl.range(0, cnt_tiles):
                cur = tl.load(s_found_topk_values_ptr)
                if cur < TOPK:
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    idx = tl.load(s_histogram_ptr + pos, mask=valid, other=0)
                    x = tl.load(
                        s_final_logits_ptr + pos,
                        mask=valid,
                        other=0,
                    )
                    key = _convert_to_uint32(x)
                    take_eq = valid & (key == thr_key)
                    out_pos_eq = tl.atomic_add(
                        found_ptrs,
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_ptr + out_pos_eq,
                        idx,
                        mask=take_eq & (out_pos_eq < TOPK),
                    )
                    if MULTIPLE_BLOCKS_PER_ROW:
                        tl.store(
                            s_out_logits_ptr + out_pos_eq,
                            x,
                            mask=take_eq & (out_pos_eq < TOPK),
                        )
        tl.debug_barrier()


@triton.jit
def _top_k_per_row_job(
    logits_ptr,
    out_indices_ptr,
    row_start,
    row_end,
    stride1,
    vocab_size,
    skip_elems,
    out_logits_ptr,
    indices_ptr,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HAS_TLE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    assume_aligned = (
        (row_start == 0)
        & (row_end == vocab_size)
        & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    if assume_aligned:
        _assume(row_start == 0)
        _assume(row_end == vocab_size)
        _assume(stride1 == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
    elif stride1 == 1:
        _assume(stride1 == 1)

    lane = tl.arange(0, BLOCK_SIZE)
    row_len = row_end - row_start
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    out_indices_ptr + pos,
                    (pos + row_start).to(tl.int32),
                    mask=take_row,
                )
                logits = tl.load(logits_ptr + pos + row_start, mask=take_row)
                tl.store(out_logits_ptr + pos, logits, mask=take_row)
            else:
                tl.store(
                    out_indices_ptr + pos,
                    pos.to(tl.int32),
                    mask=take_row,
                )
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_indices_ptr + pos, -1, mask=take_pad)
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(out_logits_ptr + pos, float("-inf"), mask=take_pad)
    else:
        # An early `return` inside this branch is what the Ascend
        # backend's TritonToLinalgIncubated pass aborts on:
        #   UseDefLists.h:198 'Cannot destroy a value that still
        #   has uses!'  with OperandType = BlockOperand.
        # An if/else is the same computation without the extra
        # block terminator, so the guard is inverted instead.
        tl.store(s_final_cnt_ptr, 0)
        tl.store(s_found_topk_values_ptr, 0)
        tl.debug_barrier()
        logit_pattern = tl.zeros((), dtype=tl.uint32)
        continue_to_next_step = tl.full((), True, dtype=tl.int1)
        threshold_bin_idx = tl.full((), -1, dtype=tl.int32)
        for step_idx in tl.static_range(0, 4):
            if continue_to_next_step:
                (
                    continue_to_next_step,
                    logit_pattern,
                    threshold_bin_idx,
                ) = _process_histogram_step(
                    logits_ptr,
                    row_start,
                    row_end,
                    stride1,
                    vocab_size,
                    skip_elems,
                    indices_ptr,
                    logit_pattern,
                    threshold_bin_idx,
                    assume_aligned,
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_threshold_bin_idx_ptr,
                    s_final_bin_size_ptr,
                    s_found_topk_values_ptr,
                    s_out_indices_ptr,
                    s_out_logits_ptr,
                    STEP=step_idx,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    HAS_TLE=HAS_TLE,
                    MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                    MERGE_BLOCKS=MERGE_BLOCKS,
                )

        if not continue_to_next_step:
            if USE_RADIX_FINAL and HAS_TLE:
                _final_select_radix(
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_found_topk_values_ptr,
                    s_out_indices_ptr,
                    s_out_logits_ptr,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                )
            else:
                base_idx = tl.load(s_found_topk_values_ptr)
                # Guard against stale/oversized counts to avoid out-of-bounds accesses
                # in the shared-memory final buffers.
                final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
                sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
                for sort_chunk in tl.range(0, sort_chunks):
                    pos = sort_chunk * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    logit_i = tl.load(
                        s_final_logits_ptr + pos,
                        mask=valid,
                        other=0,
                    )
                    out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                    for j in tl.range(0, final_cnt):
                        logit_j = tl.load(s_final_logits_ptr + j)
                        better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                        out_rank = out_rank + (valid & better).to(tl.int32)
                    dst_pos = base_idx + out_rank
                    take = valid & (dst_pos < TOPK)
                    idx_i = tl.load(
                        s_histogram_ptr + pos,
                        mask=take,
                        other=0,
                    )
                    tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
                    if MULTIPLE_BLOCKS_PER_ROW:
                        tl.store(s_out_logits_ptr + dst_pos, logit_i, mask=take)
                tl.debug_barrier()

        # out_indices_ptr is identical to s_out_indices_ptr for non-tle
        if HAS_TLE:
            flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
            for flush_chunk in tl.static_range(flush_chunks):
                pos = flush_chunk * BLOCK_SIZE + lane
                mask = pos < TOPK
                out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
                tl.store(out_indices_ptr + pos, out_vals, mask=mask)
                if MULTIPLE_BLOCKS_PER_ROW:
                    split_logits = tl.load(
                        s_out_logits_ptr + pos, mask=mask, other=float("-inf")
                    )
                    tl.store(out_logits_ptr + pos, split_logits, mask=mask)


# End of shared implementation code for top_k_per_row_decode and top_k_per_row_prefill


@triton.jit
def tle_top_k_per_row_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    NUM_FILNAL_ITEMS: tl.constexpr = 2048
    NUM_BINS: tl.constexpr = 2048
    VEC: tl.constexpr = 4

    row_id = tl.program_id(0) + ROW_OFFSET
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    # used for histogram, indices in final sort and exclude prefix_sum
    s_histogram = tle.gpu.alloc(
        [NUM_BINS],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_logits = tle.gpu.alloc(
        [NUM_FILNAL_ITEMS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPKP],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_histogram_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_final_logits_ptr = tle.gpu.local_ptr(s_final_logits, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))

    _top_k_per_row_job(
        logits_ptr,
        out_indices_ptr,
        row_start,
        row_end,
        stride1,
        vocab_size,
        skip_elems,
        None,
        None,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_out_indices_ptr,
        None,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=True,
        MULTIPLE_BLOCKS_PER_ROW=False,
        MERGE_BLOCKS=False,
    )


@triton.jit
def non_tle_top_k_per_row_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_BINS: tl.constexpr = 2048
    NUM_FILNAL_ITEMS: tl.constexpr = 2048

    row_id = tl.program_id(0) + ROW_OFFSET
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    s_histogram_ptr += row_id * NUM_BINS
    s_final_logits_ptr += row_id * NUM_FILNAL_ITEMS
    s_final_cnt_ptr += row_id
    s_threshold_bin_idx_ptr += row_id
    s_final_bin_size_ptr += row_id
    s_found_topk_values_ptr += row_id

    _top_k_per_row_job(
        logits_ptr,
        out_indices_ptr,
        row_start,
        row_end,
        stride1,
        vocab_size,
        skip_elems,
        None,
        None,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        out_indices_ptr,
        None,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=False,
        HAS_TLE=False,
        MULTIPLE_BLOCKS_PER_ROW=False,
        MERGE_BLOCKS=False,
    )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Selects top_k indices from a single row of logits using radix-based
    selection. Only valid elements within [row_start, row_end) are considered.

    Args:
        logits: [num_rows, vocab_size] float32 tensor.
        row_starts: [num_rows] int32 — start of valid range per row (inclusive).
        row_ends: [num_rows] int32 — end of valid range per row (exclusive).
        indices: [num_rows, top_k] int32 — output buffer, filled with 0-based indices
                 relative to row_starts[i]. Caller pre-allocates this.
        num_rows: number of rows.
        stride0: logits.stride(0), typically == vocab_size for contiguous tensor.
        stride1: logits.stride(1), typically == 1 for contiguous tensor.
        top_k: number of top elements per row.
    """
    logger.debug("GEMS TOP_K_PER_ROW_PREFILL")

    vocab_size = logits.shape[1]
    assert num_rows == logits.shape[0]
    if HAS_TLE:
        topkp = triton.next_power_of_2(top_k)
        use_radix_final = _use_radix_final_for_prefill(vocab_size)
        num_insert_sort_blocks = (
            0 if use_radix_final else min(num_rows, SORTING_ALGORITHM_THRESHOLD)
        )
        if num_insert_sort_blocks > 0:
            tle_top_k_per_row_prefill[(num_insert_sort_blocks,)](
                logits,
                indices,
                row_starts,
                row_ends,
                stride0,
                stride1,
                vocab_size,
                TOPK=top_k,
                TOPKP=topkp,
                BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
                USE_RADIX_FINAL=False,
                ROW_OFFSET=0,
                num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
            )
        if num_rows > num_insert_sort_blocks:
            num_radix_sort_blocks = num_rows - num_insert_sort_blocks
            tle_top_k_per_row_prefill[(num_radix_sort_blocks,)](
                logits,
                indices,
                row_starts,
                row_ends,
                stride0,
                stride1,
                vocab_size,
                TOPK=top_k,
                TOPKP=topkp,
                BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
                USE_RADIX_FINAL=True,
                ROW_OFFSET=num_insert_sort_blocks,
                num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
            )
    else:
        # based on tle version
        device = logits.device
        s_histogram_ptr = torch.empty(
            (num_rows, NUM_BINS), device=device, dtype=torch.int32
        )
        s_final_logits_ptr = torch.empty(
            (num_rows, NUM_FILNAL_ITEMS), device=device, dtype=torch.float32
        )
        s_final_cnt_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
        s_threshold_bin_idx_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_final_bin_size_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        s_found_topk_values_ptr = torch.empty(
            (num_rows,), device=device, dtype=torch.int32
        )
        non_tle_top_k_per_row_prefill[(num_rows,)](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            stride1,
            vocab_size,
            s_histogram_ptr,
            s_final_logits_ptr,
            s_final_cnt_ptr,
            s_threshold_bin_idx_ptr,
            s_final_bin_size_ptr,
            s_found_topk_values_ptr,
            TOPK=top_k,
            BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
            ROW_OFFSET=0,
            num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
        )
