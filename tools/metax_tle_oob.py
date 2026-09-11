"""Which input makes the TLE decode kernel fault on MetaX?

The shimmed-TLE test run hit a memory violation in tle_top_k_per_row_decode
on test_topk_greater_than_row_len: B=1, V=262144, top_k=512, seq_len=496.
That row goes through the MetaX split: 8 chunks of 32768, and with only 496
valid elements seven chunks are EMPTY -- their candidates are -1 / -inf
padding, so the merge stage sees 4096 candidates of which ~3584 are an
identical -inf. The kernel runs twice (per-chunk, then merge); the trap does
not say which.

A tie group larger than NUM_FINAL_ITEMS (2048) is the suspect: in the TLE
kernel s_final_logits / s_histogram are SHARED memory of fixed size, where
the non-TLE path writes global scratch. Each case below isolates one
ingredient, each in its own process (a memory violation disables the
runtime for the rest of the process).

    PYTHONPATH=src:$PYTHONPATH /data/wuyuqing/workspace/mctle-test/bin/python \
        tools/metax_tle_oob.py
"""

import os
import subprocess
import sys

CASES = {
    "test_as_is":        "B=1 V=262144 K=512 seq_len=496 (split on)",
    "split_off":         "same, FLAGGEMS_METAX_TOPK_SPLIT=0",
    "merge_like":        "1x4096 K=512: 512 randn + 3584 -inf",
    "all_equal_4096":    "1x4096 K=512: every value 1.0",
    "tie_3000":          "1x8192 K=512: 3000 copies of the max, rest randn",
    "tie_1500":          "1x8192 K=512: 1500 copies of the max (< 2048)",
    "neg_inf_only_tail": "1x8192 K=512: 600 randn + 7592 -inf",
}

if len(sys.argv) == 1:
    for name, desc in CASES.items():
        env = dict(os.environ)
        if name == "split_off":
            env["FLAGGEMS_METAX_TOPK_SPLIT"] = "0"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), name],
                           capture_output=True, text=True, timeout=600, env=env)
        out = (r.stdout + r.stderr).splitlines()
        res = next((l for l in out if l.startswith("RESULT")), None)
        if res is None:
            trap = next((l for l in out if "memory violation" in l.lower()
                         or "illegal" in l.lower()), "")
            kern = next((l for l in out if "kernelName" in l), "")
            res = (f"RESULT {name}: FAULT (exit {r.returncode}) "
                   f"{kern.split('kernelName:')[-1].split(',')[0].strip()} "
                   f"| {trap.strip()[:90]}")
        print(f"{res}\n    {desc}")
    sys.exit(0)

os.environ["FLAGGEMS_FORCE_TLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402
import metax_tle_shim  # noqa: E402

ok, msg = metax_tle_shim.install()
if not ok:
    print(f"RESULT {sys.argv[1]}: SKIP {msg}")
    sys.exit(3)

name = sys.argv[1]
torch.manual_seed(0)
dev = "cuda"
K = 512
if name in ("test_as_is", "split_off"):
    V, seq = 262144, 496
    logits = torch.randn(1, V, device=dev)
elif name == "merge_like":
    V = seq = 4096
    logits = torch.full((1, V), float("-inf"), device=dev)
    logits[0, :512] = torch.randn(512, device=dev)
elif name == "all_equal_4096":
    V = seq = 4096
    logits = torch.ones(1, V, device=dev)
elif name in ("tie_3000", "tie_1500"):
    V = seq = 8192
    n = 3000 if name == "tie_3000" else 1500
    logits = torch.randn(1, V, device=dev)
    logits[0, torch.randperm(V, device=dev)[:n]] = 10.0
else:
    V = seq = 8192
    logits = torch.full((1, V), float("-inf"), device=dev)
    logits[0, :600] = torch.randn(600, device=dev)

seq_lens = torch.full((1,), seq, dtype=torch.int32, device=dev)
idx = torch.zeros(1, K, dtype=torch.int32, device=dev)
flaggems_vllm.top_k_per_row_decode(logits, 1, seq_lens, idx, 1,
                                   logits.stride(0), logits.stride(1), K)
torch.cuda.synchronize()

valid = logits[0, :seq]
kk = min(K, seq)
want = torch.topk(valid, kk).values.sort().values
sel = idx[0, :kk].long()
inb = bool(((sel >= 0) & (sel < seq)).all())
got = valid[sel.clamp(0, seq - 1)].sort().values
pad_ok = bool((idx[0, kk:] == -1).all()) if kk < K else True
ok = inb and pad_ok and torch.equal(got, want)
print(f"RESULT {name}: {'CORRECT' if ok else 'WRONG'}  in_bounds={inb} "
      f"pad_ok={pad_ok} unique={int(sel.unique().numel())}/{kk}")
