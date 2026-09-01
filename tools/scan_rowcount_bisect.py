#!/usr/bin/env python3
"""Where does the scan compaction start disagreeing with torch?

With FLAGGEMS_ATOMIC_RETURN=0 the suite is 18/20 on Moore Threads, failing only
test_top_k_per_row_prefill_variable_lengths at num_rows=16383 for both
vocabularies. The same test at num_rows=4 passes and the atomic path passes
everything, so the scan is wrong somewhere only a large variable-length batch
reaches. Two candidates, and a sweep of num_rows separates them:

  * the host splits at SORTING_ALGORITHM_THRESHOLD = 12288, issuing a SECOND
    launch with USE_RADIX_FINAL=True for the rows above it
  * row_ends is drawn from [top_k, vocab], so a row of length exactly top_k --
    which takes the short-row branch -- is near-certain at 16383 rows and about
    0.1% likely at 4

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/scan_rowcount_bisect.py

Every configuration runs in its OWN process with the gate set in the
environment. Reloading the module to flip it does not work: the top-level
binding still points at the old module's function, Triton has cached the
compiled kernels, and `import a.b as c` yields the re-exported FUNCTION rather
than the module, so importlib.reload cannot even find it. Both mistakes were
made here first.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

VOCAB, TOPK = 4095, 512
SIZES = [4, 64, 1024, 8192, 12287, 12288, 12289, 13000, 16383]

CASE = '''
import torch
from importlib import import_module
import flaggems_vllm

# import_module, not `import ... as M`: ops/__init__ re-exports the FUNCTION
# under this name.
M = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
DEV = flaggems_vllm.device
VOCAB, TOPK, NROWS = {vocab}, {topk}, {nrows}

torch.manual_seed(123)
logits = torch.randn(NROWS, VOCAB, device=DEV, dtype=torch.float32)
starts = torch.zeros(NROWS, dtype=torch.int32, device=DEV)
ends = torch.randint(TOPK, VOCAB + 1, (NROWS,), dtype=torch.int32, device=DEV)
got = torch.full((NROWS, TOPK), -1, dtype=torch.int32, device=DEV)

M.top_k_per_row_prefill(logits, starts, ends, got, NROWS,
                        logits.stride(0), logits.stride(1), TOPK)
flaggems_vllm.runtime.torch_device_fn.synchronize()

n_eq = int((ends - starts == TOPK).sum())
bad_row, why = -1, ""
for i in range(NROWS):
    s, e = int(starts[i]), int(ends[i])
    k = min(TOPK, e - s)
    g = got[i][got[i] >= 0].to(torch.int64)
    if g.numel() != k:
        bad_row, why = i, "数量 " + str(int(g.numel())) + " 应为 " + str(k)
        break
    a = torch.sort(logits[i, s:e][g]).values
    b = torch.sort(torch.topk(logits[i, s:e], k, largest=True,
                              sorted=False).values).values
    if not torch.equal(a, b):
        bad_row, why = i, "值不符  行长=" + str(e - s)
        break
print("RESULT " + ("OK" if bad_row < 0 else "FAIL") + " " + str(bad_row)
      + " " + str(n_eq) + " " + why)
'''


def run(nrows, atomic, tmp, env):
    path = os.path.join(tmp, f"r{nrows}_{int(atomic)}.py")
    with open(path, "w") as f:
        f.write(CASE.format(vocab=VOCAB, topk=TOPK, nrows=nrows))
    e = dict(env)
    e["FLAGGEMS_ATOMIC_RETURN"] = "1" if atomic else "0"
    r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       env=e, timeout=1200)
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("RESULT "):
            p = ln.split(" ", 4)
            return p[1], int(p[2]), int(p[3]), (p[4] if len(p) > 4 else "")
    tail = [x for x in (r.stderr or "").splitlines() if x.strip()]
    return "ERR", -1, -1, (tail[-1] if tail else f"exit={r.returncode}")


def main():
    print("=" * 88)
    print("  scan 压缩：行数从哪里开始出错")
    print("=" * 88)
    print(f"  vocab={VOCAB}  top_k={TOPK}  行长随机取 [{TOPK}, {VOCAB}]")
    print("  拆分阈值 SORTING_ALGORITHM_THRESHOLD = 12288\n")
    print(f"  {'num_rows':>9}{'恰好=top_k 行数':>16}{'原子':>7}{'scan':>7}   scan 首个出错")
    env = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="scan_bisect_")
    for n in SIZES:
        a, _, n_eq, a_why = run(n, True, tmp, env)
        s, s_row, _, s_why = run(n, False, tmp, env)
        note = "" if s == "OK" else (f"第 {s_row} 行 {s_why}" if s == "FAIL" else s_why)
        if a != "OK":
            note = f"原子也失败: {a_why}  <- 探针本身有问题" + note
        print(f"  {n:>9}{n_eq:>16}{a:>7}{s:>7}   {note}", flush=True)
    print("\n  读法")
    print("    scan 从 12289 起错、12287 对   => 是两次启动 / radix-final 那条路")
    print("    scan 从「恰好=top_k 行数」首次非零起错 => 是短行分支")
    print("    都对不上                       => 另有原因，继续在这两点之间二分")
    print("    原子那一列必须全 OK；不全 OK 说明是探针的问题，不是 scan 的")
    return 0


if __name__ == "__main__":
    sys.exit(main())
