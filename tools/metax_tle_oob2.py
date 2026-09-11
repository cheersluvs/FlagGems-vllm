"""Pin the short-row failure of the shimmed TLE decode to one stage.

Round 1 (metax_tle_oob.py): ties are innocent (all-equal, 3000-way tie,
-inf padding all CORRECT). Only rows SHORTER than top_k fail, two ways:

    split on   memory violation in tle_top_k_per_row_decode
    split off  WRONG, 2 distinct indices, some out of range -- the generic
               multi-block + merge path (vocab >= 200k), each of its 10
               blocks ~49 long, i.e. every block on the row_len <= TOPK
               branch, then a merge over mostly padding

and with the split on, the merge's padding is finfo.min (from
_gather_candidates), not -inf -- round 1 only tried -inf.

Every case here calls the GENERIC op directly (no MetaX override), so each
stage is exercised alone. One process per case: a memory violation disables
the runtime for the rest of the process.

Check, for seq < K: output as a set is exactly {0..seq-1} plus K-seq of -1.
For seq >= K: the selected values equal torch.topk's.

    PYTHONPATH=src:$PYTHONPATH /data/wuyuqing/workspace/mctle-test/bin/python \
        tools/metax_tle_oob2.py
"""

import os
import subprocess
import sys

CASES = {
    "stage1_8rows":   "stage 1 exactly as the split issues it: 8x32768 view, lens [496,0x7]",
    "stage1_496":     "1x32768, seq 496 (short row, single-block path)",
    "stage1_0":       "1x32768, seq 0 (empty row)",
    "merge_finfo":    "1x4096, 496 randn + 3600 finfo.min (stage-2 input)",
    "merge_neginf":   "1x4096, 496 randn + 3600 -inf (same, -inf padding)",
    "mb_496":         "1x262144, seq 496  (multi-block path, 10 blocks of ~49)",
    "mb_600":         "1x262144, seq 600  (row > K, every block < K)",
    "mb_5200":        "1x262144, seq 5200 (every block > K)",
    "mb_full":        "1x262144, seq 262144",
}

if len(sys.argv) == 1:
    for name, desc in CASES.items():
        r = subprocess.run([sys.executable, os.path.abspath(__file__), name],
                           capture_output=True, text=True, timeout=600)
        out = (r.stdout + r.stderr).splitlines()
        res = next((l for l in out if l.startswith("RESULT")), None)
        if res is None:
            kern = next((l for l in out if "kernelName" in l), "")
            err = next((l for l in reversed(out) if "Error" in l), out[-1] if out else "")
            res = (f"RESULT {name}: FAULT (exit {r.returncode}) "
                   f"{kern.split('kernelName:')[-1].split(',')[0].strip()} | {err.strip()[:100]}")
        print(f"{res}\n    {desc}")
    sys.exit(0)

os.environ["FLAGGEMS_FORCE_TLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

import torch  # noqa: E402

import flaggems_vllm  # noqa: E402,F401
import metax_tle_shim  # noqa: E402

ok, msg = metax_tle_shim.install()
if not ok:
    print(f"RESULT {sys.argv[1]}: SKIP {msg}")
    sys.exit(3)
gen = import_module("flaggems_vllm.ops.top_k_per_row_decode")

name = sys.argv[1]
torch.manual_seed(0)
dev = "cuda"
K = 512


def check_row(values, seq, out):
    """values: the row's logits (>= seq long); out: [K] int32."""
    out = out.cpu().long()
    if seq < K:
        want = sorted(list(range(seq)) + [-1] * (K - seq))
        return sorted(out.tolist()) == want, f"distinct={out.unique().numel()}"
    inb = bool(((out >= 0) & (out < seq)).all())
    if not inb:
        return False, "out_of_range"
    got = values[:seq].cpu()[out].sort().values
    ref = torch.topk(values[:seq].cpu(), K).values.sort().values
    return torch.equal(got, ref), f"distinct={out.unique().numel()}"


if name == "stage1_8rows":
    base = torch.randn(1, 262144, device=dev)
    view = base.as_strided((8, 32768), (32768, 1))
    lens = torch.tensor([496] + [0] * 7, dtype=torch.int32, device=dev)
    out = torch.zeros(8, K, dtype=torch.int32, device=dev)
    gen.top_k_per_row_decode(view, 1, lens, out, 8, 32768, 1, K)
    torch.cuda.synchronize()
    oks = [check_row(view[r], int(lens[r]), out[r]) for r in range(8)]
    ok = all(o for o, _ in oks)
    detail = " ".join(f"r{r}:{'ok' if o else 'BAD'}" for r, (o, _) in enumerate(oks))
else:
    if name.startswith("stage1"):
        V, seq = 32768, (496 if name == "stage1_496" else 0)
        logits = torch.randn(1, V, device=dev)
    elif name.startswith("merge"):
        V, seq = 4096, 4096
        fill = torch.finfo(torch.float32).min if name == "merge_finfo" else float("-inf")
        logits = torch.full((1, V), fill, device=dev)
        logits[0, :496] = torch.randn(496, device=dev)
    else:
        V = 262144
        seq = {"mb_496": 496, "mb_600": 600, "mb_5200": 5200, "mb_full": V}[name]
        logits = torch.randn(1, V, device=dev)
    lens = torch.full((1,), seq, dtype=torch.int32, device=dev)
    out = torch.zeros(1, K, dtype=torch.int32, device=dev)
    gen.top_k_per_row_decode(logits, 1, lens, out, 1, logits.stride(0), 1, K)
    torch.cuda.synchronize()
    ok, detail = check_row(logits[0], seq, out[0])
    if not ok:
        detail += f" first={out[0, :8].tolist()}"
print(f"RESULT {name}: {'CORRECT' if ok else 'WRONG'}  {detail}")
