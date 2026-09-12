"""Does the MetaX TLE override switch on, and is the answer right either way?

Three checks the production override needs, in one run:

  1. which build this is, and what top_k_per_row_tle's self-test decided
  2. both generic modules' HAS_TLE after the first call (the switch is lazy)
  3. the answers themselves, against torch.topk, on both operators

then the two operators' functional tests in the same interpreter.

Run it TWICE -- once per environment -- because the interesting part is that
the verdict differs and both are correct:

    PY=/data/wuyuqing/workspace/mctle-test/bin/python \
        tools/vendor_probe.sh tools/metax_tle_override_check.py metax_override_mctle
    PY=/opt/conda/bin/python \
        tools/vendor_probe.sh tools/metax_tle_override_check.py metax_override_stock

On the mctle build expect on=True / 'self-test passed'; on a stock wheel
expect on=False with a reason, and the same tests passing on the non-TLE path.
FLAGGEMS_METAX_TLE=0 forces it off. --no-tests skips the pytest part.
"""

import sys
from importlib import import_module

import torch

import flaggems_vllm

RUN_TESTS = "--no-tests" not in sys.argv


def line(k, v):
    print(f"  {k:<22} {v}")


print("=== build")
line("python", sys.executable)
try:
    from triton._C import libtriton

    line("libtriton", libtriton.__file__)
    line(
        "swizzled binding",
        hasattr(libtriton.ir.builder, "make_swizzled_shared_encoding_attr"),
    )
except Exception as e:  # noqa: BLE001
    line("libtriton", f"unavailable: {e}")
try:
    line(
        "enable_mctle",
        getattr(import_module("triton.backends.metax.compiler"), "enable_mctle", None),
    )
except Exception as e:  # noqa: BLE001
    line("enable_mctle", f"unavailable: {type(e).__name__}")

dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
tle_mod = import_module("flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_tle")
line("HAS_TLE before", f"decode={dec.HAS_TLE} prefill={pre.HAS_TLE}")
line("status before", tle_mod.status())

print("\n=== first calls (this is what runs the self-test)")
torch.manual_seed(0)
dev = "cuda"
bad = []

B, V, K = 8, 65536, 512
logits = torch.randn(B, V, dtype=torch.float32, device=dev)
seq_lens = torch.full((B,), V, dtype=torch.int32, device=dev)
idx = torch.zeros(B, K, dtype=torch.int32, device=dev)
flaggems_vllm.top_k_per_row_decode(
    logits, 1, seq_lens, idx, B, logits.stride(0), logits.stride(1), K
)
torch.cuda.synchronize()
got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
ok_dec = torch.allclose(got, want) and int(((idx < 0) | (idx >= V)).sum()) == 0
bad += [] if ok_dec else ["decode"]

starts = torch.zeros(B, dtype=torch.int32, device=dev)
ends = torch.full((B,), V, dtype=torch.int32, device=dev)
idx2 = torch.zeros(B, K, dtype=torch.int32, device=dev)
flaggems_vllm.top_k_per_row_prefill(
    logits, starts, ends, idx2, B, logits.stride(0), logits.stride(1), K
)
torch.cuda.synchronize()
got2 = logits.gather(1, idx2.long().clamp(0, V - 1)).sort(dim=1).values
ok_pre = torch.allclose(got2, want) and int(((idx2 < 0) | (idx2 >= V)).sum()) == 0
bad += [] if ok_pre else ["prefill"]

line("status after", tle_mod.status())
line("HAS_TLE after", f"decode={dec.HAS_TLE} prefill={pre.HAS_TLE}")
line(
    "vs torch.topk",
    f"decode={'OK' if ok_dec else 'WRONG'} prefill={'OK' if ok_pre else 'WRONG'}",
)
on = tle_mod.status().get("on")
line(
    "verdict",
    f"TLE {'ON' if on else 'OFF'}, answers {'correct' if not bad else 'WRONG: ' + ','.join(bad)}",
)

if not RUN_TESTS:
    sys.exit(0 if not bad else 1)

print("\n=== functional tests, same interpreter", flush=True)
import pytest  # noqa: E402

rc = pytest.main(
    [
        "tests/test_top_k_per_row_prefill.py",
        "tests/test_top_k_per_row_decode.py",
        "-q",
        "-rf",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
)
print(
    f"\n=== TLE {'ON' if on else 'OFF'} | pytest rc={rc} | "
    f"spot checks {'ok' if not bad else 'WRONG'}"
)
sys.exit(rc or (1 if bad else 0))
