#!/usr/bin/env python3
"""Which parameter makes the operator crash this compiler?

The construct smoke test passes every single thing top_k_per_row uses --
elementwise, tl.max/tl.sum, a 2048-wide tl.cumsum, masked global atomics, a
static_range loop, a data-dependent guard inside one, masked stores and a
barrier inside that guard. So the MLIR abort

    UseDefLists.h:198 'use_empty() && "Cannot destroy a value that still has
    uses!"'  ->  Aborted (core dumped)

comes from the combination or the scale, not from any one construct.

This drives the non-TLE prefill kernel DIRECTLY, one parameter off the baseline
at a time, so a crash names a parameter. num_warps is first because its value on
this card is fabricated: get_device_properties exposes only name and
total_memory, so _launch_geometry falls back to a hardcoded (32, 1024) and
_num_warps(512) returns 16 -- a warp count for a part that has no warps.

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=src:$PYTHONPATH python tools/ascend_param_sweep.py

Each case is its own process, written to a real file: an MLIR assertion aborts
and would take the rest of the run with it, and Triton's JIT needs to be able to
read the kernel back with inspect.getsource(), which a `python -c` string does
not allow. Both were learned the hard way here.
"""

import os
import subprocess
import sys
import tempfile
import time

BASE = dict(num_rows=1, vocab=20000, top_k=1024, block=512, warps=16)

# One change off the baseline each. num_warps first: it is the value this card
# cannot actually report.
CASES = [
    ("基线 (block=512 warps=16 k=1024 vocab=20000)", {}),
    ("num_warps=1", dict(warps=1)),
    ("num_warps=2", dict(warps=2)),
    ("num_warps=4", dict(warps=4)),
    ("num_warps=8", dict(warps=8)),
    ("BLOCK_SIZE=128 (warps=4)", dict(block=128, warps=4)),
    ("BLOCK_SIZE=256 (warps=8)", dict(block=256, warps=8)),
    ("TOPK=64", dict(top_k=64)),
    ("TOPK=256", dict(top_k=256)),
    ("vocab=2048", dict(vocab=2048)),
    ("vocab=2048 且 TOPK=64", dict(vocab=2048, top_k=64)),
    ("最小组合 block=128 warps=1 k=64 vocab=2048",
     dict(block=128, warps=1, top_k=64, vocab=2048)),
]

TEMPLATE = '''
import torch
from importlib import import_module

import flaggems_vllm

# import_module, NOT `from flaggems_vllm.ops import top_k_per_row_prefill as M`:
# ops/__init__.py re-exports the FUNCTION under that name, so the latter binds a
# function and every M.NUM_BINS is an AttributeError.
M = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

DEV = flaggems_vllm.device
num_rows, vocab, top_k = {num_rows}, {vocab}, {top_k}
BLOCK, WARPS = {block}, {warps}

torch.manual_seed(0)
logits = torch.randn((num_rows, vocab), dtype=torch.float32, device=DEV)
indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
starts = torch.zeros((num_rows,), dtype=torch.int32, device=DEV)
ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)

hist = torch.empty((num_rows, M.NUM_BINS), device=DEV, dtype=torch.int32)
fin = torch.empty((num_rows, M.NUM_FILNAL_ITEMS), device=DEV, dtype=torch.float32)
cnt = torch.empty((num_rows,), device=DEV, dtype=torch.int32)
thr = torch.empty((num_rows,), device=DEV, dtype=torch.int32)
bsz = torch.empty((num_rows,), device=DEV, dtype=torch.int32)
fnd = torch.empty((num_rows,), device=DEV, dtype=torch.int32)

M.non_tle_top_k_per_row_prefill[(num_rows,)](
    logits, indices, starts, ends, logits.stride(0), logits.stride(1), vocab,
    hist, fin, cnt, thr, bsz, fnd,
    TOPK=top_k, BLOCK_SIZE=BLOCK, ROW_OFFSET=0, num_warps=WARPS,
)
flaggems_vllm.runtime.torch_device_fn.synchronize()

k = min(top_k, vocab)
got = indices[0, :k].to(torch.int64)
if int(got.min()) < 0 or int(got.max()) >= vocab:
    print("COMPILED_BUT_BAD_INDICES")
else:
    a = torch.sort(logits[0][got]).values
    b = torch.sort(torch.topk(logits[0], k, largest=True, sorted=False).values).values
    print("OK" if torch.equal(a, b) else "COMPILED_BUT_WRONG")
'''


def main():
    print("=" * 84)
    print("  昇腾参数扫描：哪个参数让编译器崩")
    print("=" * 84)
    print(f"  {'配置':<48}结果\n")
    env = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="asc_sweep_")
    first_ok = None
    for i, (name, over) in enumerate(CASES):
        cfg = dict(BASE)
        cfg.update(over)
        path = os.path.join(tmp, f"case_{i}.py")
        with open(path, "w") as f:
            f.write(TEMPLATE.format(**cfg))
        # Announce BEFORE running. Each case pays a fresh CANN init plus a
        # bisheng compile of a large kernel, so a case can take minutes and a
        # silent terminal is indistinguishable from a hang.
        print(f"  {name:<48}...", end="", flush=True)
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, env=env, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"\r  {name:<48}超时 (>600s)", flush=True)
            continue
        dt = time.time() - t0
        out = (r.stdout or "").strip()
        if r.returncode == 0 and "OK" in out:
            verdict = "OK  编译且结果正确"
            if first_ok is None:
                first_ok = name
        elif "COMPILED_BUT" in out:
            verdict = "编译过了但结果不对: " + out.splitlines()[-1]
        else:
            lines = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()]
            # NOT truncated. Three times in this bring-up a clipped diagnostic
            # hid the answer -- a NameError cut at 36 chars, a find piped
            # through head -3, a traceback flushed above a tail. Long is fine.
            why = lines[-1] if lines else f"exit={r.returncode}"
            if r.returncode < 0:
                why = f"信号 {-r.returncode} abort  {why}"
            verdict = "FAIL  " + why
        print(f"\r  {name:<48}{verdict}   [{dt:.0f}s]", flush=True)

    print()
    print("  读法")
    print("    某个参数一改就通过  => 崩溃由它决定，那是可绕过的")
    print("    只有最小组合通过    => 是规模问题，逐项放大找边界")
    print("    全部崩              => 与参数无关，得在内核体里二分")
    print("    「编译过了但结果不对」比崩溃更危险：昇腾有静默返回错数据的先例，")
    print("      所以每个通过的用例都对了一遍 torch.topk，而不只看有没有异常")
    if first_ok:
        print(f"\n  第一个完全通过的配置: {first_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
