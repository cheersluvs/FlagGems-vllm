"""Is chunking the eager baseline safe to time? Equivalence, then its overhead.

Uses eager_chunked.py with its thresholds forced low, at shapes the baseline also
completes whole, so the chunked and unchunked forms can be compared directly:

  1. equivalence at 32768 x 64: q (count of rel > 1e-2, max rel diff) and k_cache
     (differing bytes) between whole and 4-chunk runs on identical inputs
  2. overhead at 32768 x 128 and 65536 x 128: whole (twice, A/A floor) against
     2 and 4 chunks, plain triton.testing.do_bench median, order rotated, 3 rounds
  3. the chunking the re-benchmark will actually use, at 98304 x 128 and
     131072 x 128, one child process per shape: does it complete, and how long

If 2 shows the added launches cost far less than the floor, chunked numbers for
the two shapes the baseline could not reach are comparable with the rest.

    REPO=$PWD PYTHONPATH=$PWD/src:$PYTHONPATH python3 bench-scripts/probe_eager_chunk.py
"""

import json
import os
import statistics
import subprocess
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W, V, HEAD_BYTES, EPS = 512, 64, 584, 1e-6


def build(n, h, dev):
    import torch
    torch.manual_seed(0)
    q = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(n, dtype=torch.int64, device=dev)
    inv = 1.0 / (10000.0 ** (torch.arange(0, V, 2, dtype=torch.float32, device=dev) / V))
    t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
    f = torch.einsum("i,j->ij", t, inv)
    cs = torch.cat((f.cos(), f.sin()), dim=-1)
    slot = torch.arange(n, dtype=torch.int64, device=dev)
    kc = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8, device=dev)
    return q, kv, kc, slot, pos, cs


def child(n, h):
    import torch
    import triton
    import flaggems_vllm
    import eager_chunked as ec
    fn = flaggems_vllm.runtime.torch_device_fn
    q, kv, kc, slot, pos, cs = build(n, h, flaggems_vllm.device)
    call = lambda: ec.eager_chunked(q, kv, kc, slot, pos, cs, EPS, 64)
    call()
    fn.synchronize()
    ms = triton.testing.do_bench(call, warmup=100, rep=2000, return_mode="median")
    step = max(1, ec.CHUNK_ROWS // h)
    print("ROW " + json.dumps({"n": n, "h": h, "ms": ms, "chunks": -(-n // step)}), flush=True)


def main():
    import torch
    import triton
    import flaggems_vllm
    import eager_chunked as ec
    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device

    print("=" * 78 + "\n1. equivalence, 32768 x 64: whole vs 4 chunks\n" + "=" * 78)
    q, kv, kc, slot, pos, cs = build(32768, 64, dev)
    qa, kca = q.clone(), kc.clone()
    ec.eager(qa, kv.clone(), kca, slot, pos, cs, EPS, 64)
    qb, kcb = q.clone(), kc.clone()
    ec.eager_chunked(qb, kv.clone(), kcb, slot, pos, cs, EPS, 64, max_rows=0, chunk_rows=8192 * 64)
    fn.synchronize()
    a, b = qa.float(), qb.float()
    rel = ((a - b).abs() / a.abs().clamp(min=1e-6))
    print("  q: differing {}  rel>1e-2 {}  max rel {:.3e}".format(
        int((qa != qb).sum()), int((rel > 1e-2).sum()), float(rel.max())))
    print("  k_cache differing bytes: {}".format(int((kca != kcb).sum())))
    del q, kv, kc, qa, qb, kca, kcb, a, b, rel
    fn.empty_cache()

    print("\n" + "=" * 78 + "\n2. overhead: whole (A/A) vs 2 and 4 chunks, ms, median of 3 rounds\n" + "=" * 78)
    for n, h in ((32768, 128), (65536, 128)):
        q, kv, kc, slot, pos, cs = build(n, h, dev)
        rows = n * h
        slots = {
            "whole": lambda: ec.eager(q, kv, kc, slot, pos, cs, EPS, 64),
            "2 chunks": lambda: ec.eager_chunked(q, kv, kc, slot, pos, cs, EPS, 64, max_rows=0, chunk_rows=rows // 2),
            "4 chunks": lambda: ec.eager_chunked(q, kv, kc, slot, pos, cs, EPS, 64, max_rows=0, chunk_rows=rows // 4),
        }
        slots["whole2"] = slots["whole"]
        names = list(slots)
        per = {k: [] for k in names}
        for r in range(3):
            for k in names[r % 4:] + names[:r % 4]:
                per[k].append(triton.testing.do_bench(slots[k], warmup=100, rep=3000, return_mode="median"))
        m = {k: statistics.median(v) for k, v in per.items()}
        print("\n  {}x{}".format(n, h))
        for k in names:
            print("    {:<9} {:>10.2f} ms  {:>+7.2%} vs whole   rounds {}".format(
                k, m[k], m[k] / m["whole"] - 1, " ".join("{:.1f}".format(x) for x in per[k])))
        del q, kv, kc, cs
        fn.empty_cache()

    print("\n" + "=" * 78 + "\n3. the chunking the re-benchmark uses, one process per shape\n" + "=" * 78)
    for n, h in ((98304, 128), (131072, 128)):
        log = "/tmp/eager_chunk_{}x{}.log".format(n, h)
        with open(log, "w") as f:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), str(n), str(h)],
                               stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        row = None
        for line in open(log, errors="replace"):
            if line.startswith("ROW "):
                row = json.loads(line[4:])
        if row:
            print("  {}x{}: completed, {} chunks, {:.1f} ms".format(n, h, row["chunks"], row["ms"]))
        else:
            tail = [x.strip() for x in open(log, errors="replace").read().splitlines() if x.strip()][-1:]
            print("  {}x{}: FAILED (exit {}): {}".format(n, h, p.returncode, (tail[0] if tail else "")[:90]))
    print("\n[RESULT] EAGER_CHUNK_DONE")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3:
            child(int(sys.argv[1]), int(sys.argv[2]))
        else:
            main()
    except Exception:
        traceback.print_exc()
        print("\n[RESULT] FAILED")
    sys.stdout.flush()
