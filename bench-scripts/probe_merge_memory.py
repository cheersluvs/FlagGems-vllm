"""Does the merged (one-launch) operator leave more device memory behind?

THE QUESTION, narrowly. Merging the two launches is correct -- `--quick` is
52 passed / 8 skipped, identical to the two-launch version, and the merge probe
was bit-exact on every shape it covered. But on the FULL shape set the merged
version fails more tests: 13 vs 20 in a back-to-back pair sharing one cache
state, against a baseline that itself wobbles between 12 and 13.

Every one of those failures is an OOM inside the TEST ORACLE --

    rmsnorm_no_weight_f32: variance = xf.pow(2).mean(dim=-1, keepdim=True)
    NPU out of memory. Tried to allocate 16.00 GiB (60.96 total;
    32.46 allocated; 12.89 free; 40.01 reserved by PyTorch)

-- a torch reference building a float32 copy of a 16 GiB bfloat16 q. The
operator is not even on the stack. So the operator's own arithmetic is not in
question; what is in question is how much memory is FREE by the time the oracle
runs, and whether the merged operator is what consumed it.

WHAT WOULD EXPLAIN IT, and what would not. The merged kernel allocates no user
tensors, so a naive reading says its footprint is identical. Two things did
change that could plausibly move device memory, and they point in OPPOSITE
directions, which is exactly why this needs measuring rather than reasoning:

  * `num_tokens` and `num_tokens_insert` are no longer `tl.constexpr`, so shapes
    no longer force a recompile. That should leave FEWER kernel binaries
    resident, not more -- and it is visibly true in wall clock, 66s against
    146s for the same suite.
  * the grid is now one range covering both kinds of work, so a launch is wider
    and the chunk count differs (11 chunks at 131072x128 where the two-launch
    form issued 8 + 2). If anything is allocated per launch or per program,
    that is where it would show.

So: measure reserved and allocated bytes around each version at matched shapes,
with the same allocator state, alternating so drift lands on both.

Reports `memory_reserved` as well as `memory_allocated`. Reserved is the number
that matters here -- the caching allocator holds freed blocks, and the OOM
message says so itself: 32.46 GiB allocated against 40.01 GiB reserved. A
version that fragments worse can OOM while "using" the same amount.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import traceback

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")
OLD_REV = os.environ.get("OLD_REV", "5f165b4")   # two launches
NEW_REV = os.environ.get("NEW_REV", "4ec6ce4")   # one launch
REL = (
    "src/flaggems_vllm/runtime/backend/_ascend/fused/"
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py"
)

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

HEAD_DIM, ROPE_DIM, CACHE_BLOCK = 512, 64, 64
TOKEN_DATA_BYTES, SCALE_BYTES = 576, 8
GB = 1 << 30


def load(rev, name):
    """Both revisions of the override are self-contained, so both can be loaded
    side by side. Assert it rather than assume: if either imported the package,
    this would silently measure the installed version twice."""
    src = subprocess.check_output(
        ["git", "-C", REPO, "show", "{}:{}".format(rev, REL)]
    ).decode()
    assert "flaggems_vllm" not in src, "{} is not self-contained".format(rev)
    path = "/tmp/memprobe_{}.py".format(name)
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("memprobe_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert


def launches(rev_mod_name, n, h, insert):
    """How many kernel launches each version issues, from the host logic."""
    cap = 65535
    hp = 1
    c = min(32, h)
    while hp * 2 <= c and h % (hp * 2) == 0:
        hp *= 2
    q_programs = n * (h // hp)
    if rev_mod_name == "two":
        return (
            (q_programs + cap - 1) // cap + (insert + cap - 1) // cap
        )
    return (q_programs + insert + cap - 1) // cap


def make(n, h):
    nb = (n + CACHE_BLOCK - 1) // CACHE_BLOCK + 1
    bb = CACHE_BLOCK * (TOKEN_DATA_BYTES + SCALE_BYTES)
    torch.manual_seed(0)
    # Build bf16 without an fp32 temporary: torch.randn(dtype=bfloat16) makes
    # one on this backend, which is itself an OOM at these sizes.
    q = torch.empty(n, h, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    seed = torch.randn(1 << 20, dtype=torch.float32).to(torch.bfloat16).npu()
    flat = q.view(-1)
    for off in range(0, flat.numel(), seed.numel()):
        end = min(off + seed.numel(), flat.numel())
        flat[off:end].copy_(seed[: end - off])
    kv = torch.empty(n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    kv.view(-1)[:].copy_(seed[: n * HEAD_DIM] if n * HEAD_DIM <= seed.numel()
                         else seed.repeat((n * HEAD_DIM) // seed.numel() + 1)
                         [: n * HEAD_DIM])
    slot = torch.arange(n, dtype=torch.int64, device="npu")
    pos = torch.arange(n, dtype=torch.int64, device="npu")
    cs = torch.randn(max(4096, n), ROPE_DIM, dtype=torch.float32).npu()
    cache = torch.zeros(nb, bb, dtype=torch.uint8, device="npu")
    del seed
    return q, kv, cache, slot, pos, cs


def measure(fn, tag, n, h, calls=3):
    torch.npu.empty_cache()
    torch.npu.synchronize()
    base_alloc = torch.npu.memory_allocated()
    base_res = torch.npu.memory_reserved()

    q, kv, c, sl, po, cs = make(n, h)
    torch.npu.synchronize()
    after_inputs_res = torch.npu.memory_reserved()

    torch.npu.reset_peak_memory_stats()
    for _ in range(calls):
        fn(q, kv, c, sl, po, cs, 1e-6, CACHE_BLOCK)
    torch.npu.synchronize()

    peak_alloc = torch.npu.max_memory_allocated()
    end_res = torch.npu.memory_reserved()
    inputs = after_inputs_res - base_res

    del q, kv, c, sl, po, cs
    torch.npu.synchronize()
    # What survives the tensors going away is what would starve the oracle.
    leftover_res = torch.npu.memory_reserved() - base_res
    leftover_alloc = torch.npu.memory_allocated() - base_alloc

    print(
        "  {:<6} {:>7} {:>5} {:>9} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            tag, n, h, launches(tag, n, h, n),
            inputs / GB, (peak_alloc - base_alloc) / GB,
            leftover_res / GB, leftover_alloc / GB,
        )
    )
    torch.npu.empty_cache()
    return leftover_res


def main():
    two = load(OLD_REV, "two")
    one = load(NEW_REV, "one")
    print("two launches = {}   one launch = {}\n".format(OLD_REV, NEW_REV))
    print("  {:<6} {:>7} {:>5} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
        "ver", "tokens", "heads", "launches", "inputs GB", "peak GB",
        "left res", "left alloc"))

    # Alternate the two versions at each shape so allocator drift lands on both.
    for n, h in ((4096, 64), (8192, 128), (32768, 64), (65536, 128)):
        try:
            for tag, fn in (("two", two), ("one", one)):
                measure(fn, tag, n, h)
        except RuntimeError as e:
            print("  {:>7} {:>5}  skipped: {}".format(
                n, h, str(e).splitlines()[0][:60]))
            torch.npu.empty_cache()
        print()

    print("[RESULT] MEMPROBE_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] MEMPROBE_FAILED")
sys.stdout.flush()
