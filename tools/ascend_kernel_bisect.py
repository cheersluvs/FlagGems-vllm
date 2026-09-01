#!/usr/bin/env python3
"""Which piece of the REAL operator crashes TritonToLinalgIncubated?

Hand-written probes reproducing the operator's constructs all compile -- twelve
of them, including the scan loop written out verbatim. So the trigger needs more
context than a probe reconstructs, and re-deriving it by hand is guesswork.

This calls the operator's OWN shared @triton.jit functions from thin wrapper
kernels instead, one piece at a time:

    _final_select_radix          the smallest, called once at the end
    _process_histogram_step      one refinement step: distribute, scan, compact
    _top_k_per_row_job           all four steps plus the final select
    (whole kernel)               non_tle_top_k_per_row_prefill, known to abort

Whichever is the smallest that aborts is the reproducer to send upstream, and it
is built from the real code rather than from my reconstruction of it.

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=src:$PYTHONPATH python tools/ascend_kernel_bisect.py

Each piece is a separate process writing to files, in its own process group:
an MLIR assertion aborts, and a compiler grandchild holding an inherited pipe
once made a crash look like a hang for a whole round.
"""

import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

PRELUDE = '''
import torch
import triton
import triton.language as tl
from importlib import import_module

import flaggems_vllm

# import_module: ops/__init__ re-exports the FUNCTION under this name.
M = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
DEV = flaggems_vllm.device
NR, VOCAB, TOPK, BLOCK = 1, 2048, 64, 128

torch.manual_seed(0)
lg = torch.randn((NR, VOCAB), dtype=torch.float32, device=DEV)
idx = torch.empty((NR, TOPK), dtype=torch.int32, device=DEV)
st = torch.zeros((NR,), dtype=torch.int32, device=DEV)
en = torch.full((NR,), VOCAB, dtype=torch.int32, device=DEV)
hist = torch.zeros((NR, M.NUM_BINS), device=DEV, dtype=torch.int32)
fin = torch.zeros((NR, M.NUM_FILNAL_ITEMS), device=DEV, dtype=torch.float32)
cnt = torch.zeros((NR,), device=DEV, dtype=torch.int32)
thr = torch.zeros((NR,), device=DEV, dtype=torch.int32)
bsz = torch.zeros((NR,), device=DEV, dtype=torch.int32)
fnd = torch.zeros((NR,), device=DEV, dtype=torch.int32)
'''

PIECES = [
    ("A. _final_select_radix 单独", '''
@triton.jit
def w(hist_p, fin_p, cnt_p, fnd_p, out_p,
      TOPK: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    M_final_select_radix(hist_p, fin_p, cnt_p, fnd_p, out_p, None,
                         TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE,
                         MULTIPLE_BLOCKS_PER_ROW=False)

w[(NR,)](hist, fin, cnt, fnd, idx, TOPK=TOPK, BLOCK_SIZE=BLOCK, num_warps=1)
'''),
    ("B. _process_histogram_step 单步", '''
@triton.jit
def w(lg_p, out_p, hist_p, fin_p, cnt_p, thr_p, bsz_p, fnd_p,
      TOPK: tl.constexpr, BLOCK_SIZE: tl.constexpr, VOCAB: tl.constexpr):
    M_process_histogram_step(
        lg_p, 0, VOCAB, 1, VOCAB, 0, None, 0, -1, False,
        hist_p, fin_p, cnt_p, thr_p, bsz_p, fnd_p, out_p, None,
        STEP=0, TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE, HAS_TLE=False,
        MULTIPLE_BLOCKS_PER_ROW=False, MERGE_BLOCKS=False)

w[(NR,)](lg, idx, hist, fin, cnt, thr, bsz, fnd,
         TOPK=TOPK, BLOCK_SIZE=BLOCK, VOCAB=VOCAB, num_warps=1)
'''),
    ("C. _top_k_per_row_job 全部四步", '''
@triton.jit
def w(lg_p, out_p, hist_p, fin_p, cnt_p, thr_p, bsz_p, fnd_p,
      TOPK: tl.constexpr, BLOCK_SIZE: tl.constexpr, VOCAB: tl.constexpr):
    M_top_k_per_row_job(
        lg_p, out_p, 0, VOCAB, 1, VOCAB, 0, None, None,
        hist_p, fin_p, cnt_p, thr_p, bsz_p, fnd_p, out_p, None,
        TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE, USE_RADIX_FINAL=False,
        HAS_TLE=False, MULTIPLE_BLOCKS_PER_ROW=False, MERGE_BLOCKS=False)

w[(NR,)](lg, idx, hist, fin, cnt, thr, bsz, fnd,
         TOPK=TOPK, BLOCK_SIZE=BLOCK, VOCAB=VOCAB, num_warps=1)
'''),
    ("D. 整个 non_tle_top_k_per_row_prefill（已知崩）", '''
M.non_tle_top_k_per_row_prefill[(NR,)](
    lg, idx, st, en, lg.stride(0), lg.stride(1), VOCAB,
    hist, fin, cnt, thr, bsz, fnd,
    TOPK=TOPK, BLOCK_SIZE=BLOCK, ROW_OFFSET=0, num_warps=1)
'''),
]

# The wrappers call the real jit functions, which must be visible as plain
# globals in the generated module -- Triton resolves them by name at trace time.
BIND = '''
M_final_select_radix = M._final_select_radix
M_process_histogram_step = M._process_histogram_step
M_top_k_per_row_job = M._top_k_per_row_job
'''


def main():
    print("=" * 80)
    print("  真算子分段编译：最小的会崩的那一段就是要提交的复现")
    print("=" * 80)
    print(f"  {'分段':<44}结果\n")
    env = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="asc_kb_")
    for i, (name, body) in enumerate(PIECES):
        path = os.path.join(tmp, f"piece_{i}.py")
        with open(path, "w") as f:
            f.write(PRELUDE + BIND + body + '\nprint("COMPILED")\n')
        op, ep = path + ".out", path + ".err"
        print(f"  {name:<44}...", end="", flush=True)
        t0 = time.time()
        with open(op, "w") as fo, open(ep, "w") as fe:
            proc = subprocess.Popen([sys.executable, path], stdout=fo, stderr=fe,
                                    env=env, start_new_session=True)
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
                print(f"\r  {name:<44}超时", flush=True)
                continue
        dt = time.time() - t0
        out = pathlib.Path(op).read_text().strip()
        err = pathlib.Path(ep).read_text().strip()
        if proc.returncode == 0 and "COMPILED" in out:
            verdict = "编译通过"
        else:
            lines = [ln for ln in err.splitlines() if ln.strip()]
            why = lines[-1] if lines else f"exit={proc.returncode}"
            if proc.returncode < 0:
                why = f"信号 {-proc.returncode}  {why}"
            verdict = "FAIL  " + why
        print(f"\r  {name:<44}{verdict}   [{dt:.0f}s]", flush=True)
        print(f"        源文件: {path}", flush=True)

    print()
    print("  读法")
    print("    A 就崩          => final select 一段即可复现，最小")
    print("    A 过、B 崩      => 一个精化步就够，仍然很小")
    print("    A B 过、C 崩    => 需要四步叠加，是规模/组合问题")
    print("    A B C 全过、D 崩 => 触发点在 non_tle 包装层而非 job 本身")
    print("    印出的源文件可以直接连同 MLIR reproducer 一起交给 FlagTree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
