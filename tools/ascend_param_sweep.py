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
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

BASE = dict(num_rows=1, vocab=20000, top_k=1024, block=512, warps=16,
            watchdog=180)

# One change off the baseline each. num_warps first: it is the value this card
# cannot actually report.
# Cut down while the hang is being located: the full grid is pointless if the
# first case never finishes. Restore the rest once a case is known to complete.
# Correctness, not compilability: the kernel now compiles and runs, and gives
# the wrong answer at vocab=20000. `assume_aligned` requires
# vocab_size % BLOCK_SIZE == 0, and 20000 % 512 = 32 -- so that launch took the
# UNALIGNED path, which is where this backend has a recorded defect of masked
# loads with a runtime row offset silently returning wrong data. These pair
# aligned and unaligned vocabularies at the same size to test exactly that.
CASES = [
    ("vocab=2048  对齐 (2048%512=0)   k=64", dict(vocab=2048, top_k=64)),
    ("vocab=2080  非对齐 (2080%512=32) k=64", dict(vocab=2080, top_k=64)),
    ("vocab=20480 对齐                 k=64", dict(vocab=20480, top_k=64)),
    ("vocab=20000 非对齐               k=64", dict(vocab=20000, top_k=64)),
    ("vocab=20480 对齐                 k=1024", dict(vocab=20480, top_k=1024)),
    ("vocab=20000 非对齐               k=1024", dict(vocab=20000, top_k=1024)),
    ("vocab=512   对齐 且 = BLOCK      k=64", dict(vocab=512, top_k=64)),
    ("num_rows=4  vocab=2048 对齐      k=64", dict(num_rows=4, vocab=2048, top_k=64)),
]


TEMPLATE = '''
import faulthandler
# A hung case is useless without a stack. The Ascend frontend runs its MLIR
# passes IN PROCESS -- the abort we first saw came from one -- so a case that
# stops making progress leaves no bisheng child and nothing in the Triton
# cache, and there is otherwise no way to tell a slow pass from a spin.
faulthandler.dump_traceback_later({watchdog}, exit=True)

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
    if torch.equal(a, b):
        print("OK")
    else:
        same = int((a == b).sum())
        print(f"COMPILED_BUT_WRONG  {k - same}/{k} 个值不符")
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
        # To FILES, not pipes, and in its own process group.
        #
        # capture_output=True waits for the pipes to close, and Triton spawns a
        # compiler child that inherits them -- so when the in-process watchdog
        # killed our probe at 180s the grandchild kept the pipes open, run()
        # blocked to the outer timeout, and TimeoutExpired discarded exactly the
        # stack dump the watchdog had just written. Thirteen cases reported
        # "outer timeout" and not one showed where it was stuck.
        op = os.path.join(tmp, f"case_{i}.out")
        ep = os.path.join(tmp, f"case_{i}.err")
        timed_out = False
        with open(op, "w") as fo, open(ep, "w") as fe:
            proc = subprocess.Popen([sys.executable, path], stdout=fo, stderr=fe,
                                    env=env, start_new_session=True)
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
        dt = time.time() - t0
        rc = proc.returncode
        out = pathlib.Path(op).read_text().strip()
        err = pathlib.Path(ep).read_text().strip()
        if timed_out:
            print(f"\r  {name:<48}外层超时 (>300s)   [{dt:.0f}s]", flush=True)
            frames = [ln for ln in err.splitlines() if ln.strip().startswith("File ")]
            for fr in frames[-10:]:
                print(f"        {fr.strip()}", flush=True)
            continue
        r = None
        if rc == 0 and "OK" in out:
            verdict = "OK  编译且结果正确"
            if first_ok is None:
                first_ok = name
        elif "COMPILED_BUT" in out:
            verdict = "编译过了但结果不对: " + out.splitlines()[-1]
        else:
            if "Timeout (0:0" in err or "dump_traceback_later" in err:
                # the watchdog fired: show where it was stuck, not just that it was
                frames = [ln for ln in err.splitlines()
                          if ln.strip().startswith("File ")]
                print(f"\r  {name:<48}卡死 (>{cfg['watchdog']}s)   [{dt:.0f}s]",
                      flush=True)
                for fr in frames[-8:]:
                    print(f"        {fr.strip()}", flush=True)
                continue
            lines = [ln for ln in err.splitlines() if ln.strip()]
            # NOT truncated. Three times in this bring-up a clipped diagnostic
            # hid the answer -- a NameError cut at 36 chars, a find piped
            # through head -3, a traceback flushed above a tail. Long is fine.
            why = lines[-1] if lines else f"exit={rc}"
            if rc is not None and rc < 0:
                why = f"信号 {-rc} abort  {why}"
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
