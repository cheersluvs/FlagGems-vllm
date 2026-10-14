# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Candidate source reduction. GPU reproduction is not yet verified."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def async_dot(Q, K, Out, STAGES: tl.constexpr, POLICY: tl.constexpr):
    n = tl.arange(0, 64)
    d = tl.arange(0, 32)
    query = tl.load(Q + n[None, :] * 32 + d[:, None])
    for tile in tl.range(0, 16, num_stages=STAGES):
        rows = tile * 64 + tl.arange(0, 64)
        keys = tl.load(K + rows[:, None] * 32 + d[None, :], eviction_policy=POLICY)
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        tl.store(Out + rows[:, None] * 64 + n[None, :], scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--policy", choices=("evict_first", "none"), default="evict_first"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="flagtree-cp-async-source."))
    output.mkdir(parents=True, exist_ok=True)
    print("Reports:", output, flush=True)
    print("torch:", torch.__version__, torch.__file__)
    print("triton:", triton.__version__, triton.__file__)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    print("device:", torch.cuda.get_device_name(), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    q = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1024, 32), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((1024, 64), device="cuda", dtype=torch.float32)
    expected = k.float() @ q.float().T
    policy = "" if args.policy == "none" else args.policy
    options = {"STAGES": args.stages, "POLICY": policy, "num_warps": 4}
    state = {"stages": args.stages, "policy": args.policy, "passed": False}
    try:
        torch.cuda.synchronize()
        state["phase"] = "compile"
        compiled = async_dot.warmup(q, k, out, grid=(1,), **options)
        for kind in ("ttgir", "llir", "ptx", "cubin"):
            data = compiled.asm.get(kind)
            if data is not None:
                (output / ("kernel." + kind)).write_bytes(
                    data if isinstance(data, bytes) else data.encode()
                )
        ptx = compiled.asm["ptx"]
        state["async_cache_hints"] = sum(
            "cp.async." in line and ".L2::cache_hint" in line
            for line in ptx.splitlines()
        )
        print("async cache-hint sites:", state["async_cache_hints"], flush=True)
        state["phase"] = "launch"
        async_dot[(1,)](q, k, out, **options)
        state["phase"] = "synchronize"
        torch.cuda.synchronize()
        state["phase"] = "validate"
        torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-3)
        state.update(phase="passed", passed=True)
        print("PASS; this candidate did not reproduce the failure", flush=True)
        return 0
    except Exception as exc:
        state["error"] = repr(exc)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
