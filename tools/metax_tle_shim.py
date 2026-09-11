"""Put top_k_per_row on its TLE path on MetaX -- tools only, no src/ change.

Measured on the C550 with the -D__MCTLE__ FlagTree build (see
tools/build_flagtree_metax.sh and tools/metax_tle_force_decode.py):

  cumsum     tle.cumsum needs builder.create_exclusive_cumsum, absent on
             metax. Replaced by tl.cumsum(x) - x and tl.sum(x): tle.cumsum
             IS (exclusive prefix, total), per its docstring.
  local_ptr  the plugin's __MCTLE__ block widens vec for UNMASKED shared
             loads from pointer alignment alone, unclamped by elements per
             thread, and asserts at < 4 elements/thread. Adding pid >> 31
             (always 0, divisibility 1) makes the alignment unprovable. It
             must be a BUILTIN: a jit shim taking the buffer needs
             builder.get_memdesc_type, also absent on metax.
  radix      USE_RADIX_FINAL runs only under TLE and returns duplicate
             indices on MetaX whenever it runs (bisected: radix off ->
             every case CORRECT). Off by default here.

Both shims go in as a types.ModuleType bound to the generic modules' `tle`
global -- @jit code may only reach module-typed globals.

    import tools.metax_tle_shim as s; s.install()      # before first call
"""

import os
import types
from importlib import import_module

import triton
import triton.language as tl


@triton.jit
def _cumsum_shim(x, axis: tl.constexpr = 0, reverse: tl.constexpr = False):
    tl.static_assert(not reverse, "cumsum shim: reverse=True not implemented")
    return tl.cumsum(x, axis=axis) - x, tl.sum(x, axis=axis)


def install(radix_final=False, opaque=True):
    """Returns (ok, message). Requires FLAGGEMS_FORCE_TLE=1 set before the
    generic modules were first imported, since HAS_TLE is fixed at import."""
    dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    if not (dec.HAS_TLE and pre.HAS_TLE):
        return False, (f"TLE off (decode={dec.HAS_TLE} prefill={pre.HAS_TLE}); "
                       f"FLAGGEMS_FORCE_TLE={os.environ.get('FLAGGEMS_FORCE_TLE')}")
    real = dec.tle.gpu
    gpu = real
    if opaque:
        @tl.core.builtin
        def _local_ptr_opaque(buffer, indices=None, _semantic=None, _generator=None):
            p = real.local_ptr(buffer, indices, _semantic=_semantic, _generator=_generator)
            zero = tl.program_id(0, _semantic=_semantic).__rshift__(31, _semantic=_semantic)
            return p.__add__(zero, _semantic=_semantic)

        gpu = types.ModuleType("tle_gpu_metax_shim")
        for k, v in vars(real).items():
            if not k.startswith("__"):
                setattr(gpu, k, v)
        gpu.local_ptr = _local_ptr_opaque
    shim = types.ModuleType("tle_metax_shim")
    shim.gpu = gpu
    shim.cumsum = _cumsum_shim
    dec.tle = shim
    pre.tle = shim
    if not radix_final:
        dec.SORTING_ALGORITHM_THRESHOLD = 1 << 40
        pre._use_radix_final_for_prefill = lambda vocab_size: False
    # threads: Triton's limit on the C550 is 512 threads per block ("Hardware
    # limit: 512"), but _launch_geometry reads torch's max_threads_per_block,
    # which says more -- so the TLE-only multi-block MERGE launch at
    # NUM_THREADS_PER_BLOCK_MERGE=1024 asked for 16 warps and died with
    # OutOfResources on every vocab >= SPLIT_WORK_THRESHOLD shape that the
    # MetaX split does not take (rows >= SMs). Shrink the TILE, not just the
    # warps: 1024 lanes on 8 warps is 2 elements/thread, which is exactly
    # where the masked single-address smem atomic writes wrong byte offsets.
    for m in (dec, pre):
        if hasattr(m, "NUM_THREADS_PER_BLOCK_MERGE"):
            m.NUM_THREADS_PER_BLOCK_MERGE = 512
        if hasattr(m, "_LAUNCH_GEOMETRY"):
            m._LAUNCH_GEOMETRY = (64, 512)
    return True, (f"TLE shims installed (opaque={opaque} radix_final={radix_final} "
                  f"merge_threads=512)")
