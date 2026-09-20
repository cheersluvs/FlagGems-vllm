"""Read-only BW1000 codegen audit of the shipped non-TLE prefill kernel.

Compile/launch the exact production kernel at the production geometry, then
report metadata and a compact target-assembly opcode census. No timing or
speedup claim is made; this is a gate for subsequent layout/ISA experiments.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from importlib import import_module

from hygon_prefill_audit import Plan, SHAPES, emit, inputs


SHAPE_IDS = (0, 2, 3, 6)
INTEREST = (
    "atomic", "barrier", "waitcnt", "scratch", "buffer_load",
    "buffer_store", "global_load", "global_store", "flat_load",
    "flat_store", "ds_", "cvt_f16", "cvt_f32", "perm", "bpermute",
)


def census(asm):
    opcodes = Counter()
    examples = []
    for line in asm.splitlines():
        code = line.split("//", 1)[0].strip()
        if not code or code.startswith((".", ";", "#")):
            continue
        opcode = code.split(None, 1)[0]
        if not opcode.startswith(("s_", "v_", "ds_", "buffer_", "global_", "flat_", "scratch_")):
            continue
        opcodes[opcode] += 1
        if len(examples) < 24 and any(word in opcode for word in INTEREST):
            examples.append(code[:180])
    return opcodes, examples


def worker(shape_id):
    import torch

    shape = SHAPES[shape_id]
    rows, vocab, k, stride0 = shape
    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * k
    mod = ov._dense_carry if dense else ov._sparse
    if mod is None or mod.HAS_TLE:
        raise RuntimeError("Expected shipped non-TLE Hygon path")
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    tensors = inputs(rows, vocab, stride0, k, 42)
    plan = Plan(mod, tensors, k, block, warps)
    compiled = mod.non_tle_top_k_per_row_prefill.run(
        *plan.args, TOPK=k, BLOCK_SIZE=block, ROW_OFFSET=0,
        num_warps=warps, grid=(rows,), warmup=False,
    )
    torch.cuda.synchronize()
    if compiled is None:
        raise RuntimeError("JIT did not return a CompiledKernel")
    asm_map = getattr(compiled, "asm", {})
    keys = list(asm_map.keys())
    target_key = next((key for key in ("amdgcn", "hsaco", "ptx") if key in asm_map), None)
    metadata = getattr(compiled, "metadata", None)
    fields = dict(
        shape=shape, shape_id=shape_id, dense=dense,
        geometry=[block, warps], asm_keys=keys, target_key=target_key,
        registers=getattr(compiled, "n_regs", None),
        spills=getattr(compiled, "n_spills", None),
        shared_bytes=getattr(metadata, "shared", None),
        metadata_warps=getattr(metadata, "num_warps", None),
        kernel_name=getattr(compiled, "name", None),
    )
    if target_key is None or not isinstance(asm_map[target_key], str):
        emit("codegen", **fields, note="No textual target assembly available")
        return
    asm = asm_map[target_key]
    counts, examples = census(asm)
    emit(
        "codegen", **fields,
        target_sha256=hashlib.sha256(asm.encode()).hexdigest(),
        target_bytes=len(asm.encode()),
        opcode_total=sum(counts.values()),
        interesting={op: n for op, n in sorted(counts.items())
                     if any(word in op for word in INTEREST)},
        top_opcodes=counts.most_common(25),
        first_interesting_lines=examples,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=SHAPE_IDS)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    failures = []
    for shape_id in SHAPE_IDS:
        emit("codegen_worker_start", shape_id=shape_id, shape=SHAPES[shape_id])
        try:
            done = subprocess.run(
                [sys.executable, "-u", __file__, "--worker", str(shape_id)],
                timeout=1800, check=False,
            )
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = 124
        emit("codegen_worker_exit", shape_id=shape_id, code=code)
        if code:
            failures.append(shape_id)
    emit("codegen_suite_summary", failures=failures)
    if failures:
        raise RuntimeError(f"Codegen workers failed: {failures}")


if __name__ == "__main__":
    main()
