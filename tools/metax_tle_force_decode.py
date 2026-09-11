"""Force top_k_per_row_decode onto its TLE path on MetaX; print the ROOT cause.

pytest shows only the outermost CompilationError -- the call site of
_top_k_per_row_job -- while the real failure sits several frames in (the run
also printed "at 165:43" and "at 80:16"). Triton chains them through
__cause__/__context__, so walk the chain and print every link, innermost last.

If it compiles, check the answer against torch.topk: the TLE path is where
masked smem atomics hand out write positions, so "runs" is not "correct".

    FLAGGEMS_FORCE_TLE is set here; run with the mctle venv:
    PYTHONPATH=src:$PYTHONPATH /data/wuyuqing/workspace/mctle-test/bin/python \
        tools/metax_tle_force_decode.py
"""

import os
import sys
from importlib import import_module

os.environ["FLAGGEMS_FORCE_TLE"] = "1"

import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
print(f"HAS_TLE={dec.HAS_TLE}")
if not dec.HAS_TLE:
    print("!! TLE did not switch on -- wrong venv, or tle failed to import")
    sys.exit(3)


def chain(e):
    seen, links = set(), []
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        links.append(e)
        e = e.__cause__ or e.__context__
    return links


SHAPES = ((1, 262144, 512), (64, 129280, 512), (8, 32768, 2048))
for B, V, K in SHAPES:
    print(f"\n=== B={B} V={V} K={K}")
    torch.manual_seed(0)
    logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((B,), V, dtype=torch.int32, device="cuda")
    idx = torch.zeros(B, K, dtype=torch.int32, device="cuda")
    try:
        flaggems_vllm.top_k_per_row_decode(logits, 1, seq_lens, idx, B,
                                           logits.stride(0), logits.stride(1), K)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        links = chain(e)
        print(f"  FAILED -- {len(links)} chained exception(s), innermost last:")
        for i, x in enumerate(links):
            lines = str(x).strip().splitlines()
            print(f"  --- [{i}] {type(x).__module__}.{type(x).__name__}")
            for ln in lines[-14:]:
                print(f"      {ln}")
        break                       # later shapes would fail the same way
    got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
    want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
    dup = sum(int(idx[r].unique().numel()) != K for r in range(B))
    oob = int(((idx < 0) | (idx >= V)).sum())
    ok = torch.allclose(got, want) and dup == 0 and oob == 0
    print(f"  {'CORRECT' if ok else 'WRONG'}  rows_with_dup_indices={dup}  "
          f"out_of_range={oob}  max|diff|={float((got - want).abs().max()):.3g}")
