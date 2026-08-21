"""Can the tiled Q body and the KV body live in ONE kernel without blowing UB?

WHY THIS IS THE ONLY OPEN QUESTION. The operator currently issues two launches
(:466 and :481) and a launch costs ~450 us on this card, so its floor is ~0.95 ms
across the whole 1-1024 token range -- most of the benchmark's shapes and all of
decode. One launch would halve that.

The structural half of the question is already answered, and not by reasoning:
the SHIPPED KV kernel still carries a full Q branch at :302, dead at runtime
because the host builds `pid` so that `is_kv` is always true, but compiled all
the same. A Q branch and a KV branch coexist in one kernel on this card today.

What is NOT answered is UB. That dead branch is the OLD per-unit Q body -- a 1-D
[512] block, about 2 KB. The merged kernel needs the TILED body, an [H, 512]
float32 tile, and H is already at the edge: H=64 fails with

    ub overflow, requires 3215360 bits while 1572864 bits available

392.5 KB against 192 KB, three times the tile itself, because intermediates are
live together and multi-buffering asks for more again. H=32 fits alone. Whether
it still fits beside the KV body is the measurement.

`num_stages` is swept with H because it IS the multi-buffering knob: the shipped
KV kernel passes num_stages=2 and the Q kernel leaves it at the default, and a
merged kernel can only have one value.

Each variant runs in a FRESH SUBPROCESS. A UB overflow is a clean compile error,
but the neighbouring failures on this backend (ttir_to_linalg, the UB allocator
meeting an scf.if) abort with SIGABRT, and an aborted compile leaves ~8 TBE
workers holding a pipe's write end -- so output goes to a FILE and the whole
process group is swept afterwards. One probe once hung 900s on that.

The bodies are taken verbatim from the shipped kernels and the encoder is
IMPORTED from the shipped module rather than copied, so this measures production
code. What is dropped is only what a merged kernel would genuinely not have: the
dead Q branch inside the KV kernel, and the bounds guards that an exact grid
makes unreachable.

FIRST RUN MEASURED NOTHING. All five variants died identically, in 19 seconds,
on a mistake of mine rather than anything about this card:

    Mismatched type for even_blk between then block (<[32, 32], fp32>)
    and else block (<[32], fp32>)

Triton folds a name assigned in both arms of an `if` into one SSA value and
requires one type at the join, and both branches here bound `even_blk`,
`odd_blk`, `new_even_blk`, `new_odd_blk` at different shapes. Nothing
Ascend-specific -- it would fail the same way on NVIDIA. The shipped KV kernel
escapes it only because its dead Q branch is the OLD per-unit body, whose
`qkv_blk_rope` happens to be [32, 2] on both sides.

The KV branch is now fully prefixed, including names that do agree today, and
the fix is checked by walking the AST of both arms and asserting the sets of
bound names are disjoint -- a check that costs nothing and would have caught
this before a round trip to the box.
"""

import os
import signal
import subprocess
import sys
import time

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")
TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "600"))

# (name, H, num_stages, early_returns)
#
# Ordered so the most informative come first: H32/ns1 is the outcome that needs
# no compromise, H16 the likely fallback, and the last one separates "a `return`
# inside a branch" from "UB is full" if the first three all abort.
VARIANTS = [
    ("H32_ns1", 32, 1, True),
    ("H16_ns1", 16, 1, True),
    ("H32_ns2", 32, 2, True),
    ("H8_ns1", 8, 1, True),
    ("H32_ns1_noret", 32, 1, False),
]

CHILD = r'''
import os, sys, time, traceback
REPO = {repo!r}
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from importlib import import_module

SHIP = import_module(
    "flaggems_vllm.runtime.backend._ascend.fused"
    ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
)
# Imported, not copied: the encoder is the hot path and the thing most likely to
# drift if this probe kept its own transcription of it.
_f32_to_e4m3_bits = SHIP._f32_to_e4m3_bits
_ue8m0_scale = SHIP._ue8m0_scale
shipped_op = SHIP.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

H_CONST = {h}
NUM_STAGES = {ns}


@triton.jit
def merged_kernel(
    q, kv, k_cache, k_cache_bf16, slot_mapping, position_ids, cos_sin_cache,
    eps, cache_block_size: tl.constexpr, num_heads, kv_block_stride,
    pid_offset, q_programs, tiles_per_token, H: tl.constexpr,
):
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM
    FP8_MAX: tl.constexpr = 448.0

    # One grid over both kinds of work. Programs [0, q_programs) are Q tiles and
    # the rest are one KV unit each, so a chunk may straddle the boundary
    # harmlessly -- every program decides for itself from its global id, which
    # is why this needs no "merge below 65535, split above" dispatch rule.
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    if pid < q_programs:
        token_idx = pid // tiles_per_token
        head_base = (pid % tiles_per_token) * H
        rows = token_idx * num_heads + head_base + tl.arange(0, H)

        col = tl.arange(0, HEAD_DIM)
        blk = tl.load(q + rows[:, None] * HEAD_DIM + col[None, :]).to(tl.float32)

        variance = tl.sum(blk * blk, axis=1) / HEAD_DIM
        rsqrt = tl.rsqrt(variance + eps)
        blk = blk * rsqrt[:, None]
        tl.store(
            q + rows[:, None] * HEAD_DIM + col[None, :],
            blk.to(tl.bfloat16),
            mask=col[None, :] < NOPE_DIM,
        )

        position_id = tl.load(position_ids + token_idx)
        half = tl.arange(0, HALF_ROPE_DIM)
        cos_blk = tl.load(cos_sin_cache + position_id * ROPE_DIM + half)
        sin_blk = tl.load(
            cos_sin_cache + position_id * ROPE_DIM + HALF_ROPE_DIM + half
        )
        pair_off = (
            rows[:, None, None] * HEAD_DIM
            + NOPE_DIM
            + half[None, :, None] * 2
            + tl.arange(0, 2)[None, None, :]
        )
        pair = tl.load(q + pair_off).to(tl.float32)
        even_blk, odd_blk = tl.split(pair)
        even_blk = even_blk * rsqrt[:, None]
        odd_blk = odd_blk * rsqrt[:, None]
        new_even_blk = even_blk * cos_blk[None, :] - odd_blk * sin_blk[None, :]
        new_odd_blk = even_blk * sin_blk[None, :] + odd_blk * cos_blk[None, :]
        tl.store(q + pair_off, tl.join(new_even_blk, new_odd_blk).to(tl.bfloat16))
    else:
        # Every name here is prefixed so that NOTHING is shared with the Q
        # branch. Triton unifies a name assigned in both arms of an `if` into
        # one SSA value and requires the types to match, so `even_blk` as
        # [H, 32] above and [32] here is a compile error at the join point --
        # which is what the first run of this probe hit, five times over, before
        # it could say anything about UB at all. Names that DO happen to agree
        # today (token_idx, cos_blk) are renamed too: relying on that is relying
        # on the two branches never being edited independently.
        kv_token = pid - q_programs
        kv_base = kv + kv_token * HEAD_DIM
        offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
        offset_quant = tl.arange(0, QUANT_BLOCK)
        offset_pair = offset_half_rope[:, None] * 2 + tl.arange(0, 2)[None, :]

        qkv_blk_rope = tl.load(kv_base + NOPE_DIM + offset_pair).to(tl.float32)
        kv_position = tl.load(position_ids + kv_token)
        cs_base = cos_sin_cache + kv_position * ROPE_DIM
        kv_cos = tl.load(cs_base + offset_half_rope)
        kv_sin = tl.load(cs_base + offset_half_rope + HALF_ROPE_DIM)
        kv_even, kv_odd = tl.split(qkv_blk_rope)
        kv_new_even = kv_even * kv_cos - kv_odd * kv_sin
        kv_new_odd = kv_even * kv_sin + kv_odd * kv_cos
        qkv_blk_rope = tl.join(kv_new_even, kv_new_odd).to(tl.bfloat16)

        kv_slot = tl.load(slot_mapping + kv_token)
{kv_tail}


def merged_op(q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, eps,
              cache_block_size):
    num_tokens, num_heads, _ = q.shape
    num_tokens_insert = slot_mapping.shape[0]
    k_cache_bf16 = k_cache.view(torch.bfloat16)
    H = H_CONST
    assert num_heads % H == 0
    tiles_per_token = num_heads // H
    q_programs = num_tokens * tiles_per_token
    total = q_programs + num_tokens_insert
    for pid_offset in range(0, total, SHIP.MAX_PROGRAMS_PER_LAUNCH):
        grid = min(SHIP.MAX_PROGRAMS_PER_LAUNCH, total - pid_offset)
        merged_kernel[(grid,)](
            q, kv, k_cache, k_cache_bf16, slot_mapping, position_ids,
            cos_sin_cache, eps, cache_block_size, num_heads,
            k_cache.stride(0), pid_offset, q_programs, tiles_per_token, H,
            num_warps=1, num_stages=NUM_STAGES,
        )


HEAD_DIM, ROPE_DIM, CACHE_BLOCK = 512, 64, 64
TOKEN_DATA_BYTES, SCALE_BYTES = 576, 8


def make(n, h, negatives=False):
    nb = (n + CACHE_BLOCK - 1) // CACHE_BLOCK + 1
    bb = CACHE_BLOCK * (TOKEN_DATA_BYTES + SCALE_BYTES)
    torch.manual_seed(0)
    q = torch.randn(n, h, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    kv = torch.randn(n, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    slot = torch.arange(n, dtype=torch.int64, device="npu")
    if negatives:
        # The padding path. It is the only `if` left in the KV branch, so a
        # variant that restructures those returns has to be checked on it.
        slot[::3] = -1
    pos = torch.arange(n, dtype=torch.int64, device="npu")
    cs = torch.randn(max(4096, n), ROPE_DIM, dtype=torch.float32).npu()
    cache = torch.zeros(nb, bb, dtype=torch.uint8, device="npu")
    return q, kv, cache, slot, pos, cs


def check(n, h, negatives):
    q1, kv, c1, sl, po, cs = make(n, h, negatives)
    q2, c2 = q1.clone(), c1.clone()
    shipped_op(q1, kv, c1, sl, po, cs, 1e-6, CACHE_BLOCK)
    merged_op(q2, kv, c2, sl, po, cs, 1e-6, CACHE_BLOCK)
    torch.npu.synchronize()
    dq = int((q1.cpu() != q2.cpu()).sum())
    dc = int((c1.cpu() != c2.cpu()).sum())
    tag = " (mixed -1 slots)" if negatives else ""
    print("  {{}}x{{}}{{}}: q differs {{}}, k_cache differs {{}}".format(
        n, h, tag, dq, dc))
    del q1, q2, kv, c1, c2
    torch.npu.empty_cache()
    # The two compute q identically -- same tile, same order -- so unlike the
    # before/after comparison this one must be EXACT, not merely close.
    return dq == 0 and dc == 0


def timeit(fn, args, iters=20):
    for _ in range(3):
        fn(*args)
    torch.npu.synchronize()
    best = None
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        torch.npu.synchronize()
        t = (time.perf_counter() - t0) / iters
        best = t if best is None else min(best, t)
    return best * 1e3


def main():
    print("H={{}} num_stages={{}}".format(H_CONST, NUM_STAGES))
    ok = True
    for n, h, neg in ((64, 64, False), (64, 64, True), (1024, 64, False),
                      (17, 128, False)):
        if h % H_CONST != 0:
            continue
        ok = check(n, h, neg) and ok
    if not ok:
        print("[RESULT] COMPILED_BUT_WRONG")
        return

    print()
    print("  {{:>7}} {{:>6}} {{:>12}} {{:>12}} {{:>10}}".format(
        "tokens", "heads", "two ms", "one ms", "gain"))
    for n, h in ((1, 64), (64, 64), (256, 64), (1024, 128), (2048, 64)):
        if h % H_CONST != 0:
            continue
        q, kv, c, sl, po, cs = make(n, h)
        rest = (kv, c, sl, po, cs, 1e-6, CACHE_BLOCK)
        t2 = timeit(shipped_op, (q,) + rest)
        t1 = timeit(merged_op, (q,) + rest)
        print("  {{:>7}} {{:>6}} {{:>12.4f}} {{:>12.4f}} {{:>9.2f}}x".format(
            n, h, t2, t1, t2 / t1))
        del q, kv, c
        torch.npu.empty_cache()
    print("\n[RESULT] OK")


try:
    main()
except Exception as e:
    traceback.print_exc()
    msg = str(e)
    if "ub overflow" in msg or "bits available" in msg:
        print("\n[RESULT] UB_OVERFLOW")
    else:
        print("\n[RESULT] FAILED")
sys.stdout.flush()
'''

KV_TAIL_RETURN = """        if kv_slot < 0:
            return
        block_idx = kv_slot // cache_block_size
        pos_in_block = kv_slot % cache_block_size
        block_base = k_cache + block_idx * kv_block_stride
        token_fp8_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
        token_bf16_idx = (
            block_idx * kv_block_stride + pos_in_block * TOKEN_DATA_BYTES + NOPE_DIM
        ) // 2
        token_scale_ptr = (
            block_base
            + cache_block_size * TOKEN_DATA_BYTES
            + pos_in_block * SCALE_BYTES_PER_TOKEN
        )
        tl.store(k_cache_bf16 + token_bf16_idx + offset_pair, qkv_blk_rope)
        gidx = tl.arange(0, 8)
        keep_group = gidx < NUM_QUANT_BLOCKS
        kv_quant_blk = tl.load(
            kv_base + gidx[:, None] * QUANT_BLOCK + offset_quant[None, :]
        ).to(tl.float32)
        block_max = tl.maximum(tl.max(tl.abs(kv_quant_blk), axis=1), 1e-4)
        scale, scale_code = _ue8m0_scale(block_max / FP8_MAX)
        x_scaled = kv_quant_blk / scale[:, None]
        x_uint8 = _f32_to_e4m3_bits(x_scaled)
        tl.store(
            token_fp8_ptr + gidx[:, None] * QUANT_BLOCK + offset_quant[None, :],
            x_uint8,
            mask=keep_group[:, None],
        )
        tl.store(token_scale_ptr + gidx, scale_code.to(tl.uint8), mask=keep_group)
        tl.store(token_scale_ptr + NUM_QUANT_BLOCKS, tl.zeros((), dtype=tl.uint8))"""

# Same work under `if kv_slot >= 0:` instead of an early return. This is the
# variant that separates "a return inside a branch" from "UB is full" -- but it
# is not free: it adds a second nested scf.if, and the UB allocator meeting an
# scf.if is a known failure on this backend.
KV_TAIL_NORET = "        if kv_slot >= 0:\n" + "\n".join(
    "    " + line for line in KV_TAIL_RETURN.splitlines()[2:]
)


def assert_branches_disjoint(src):
    """No name may be bound in both arms of the kernel's `if`.

    Triton unifies such a name into one SSA value and demands one type at the
    join, so a shared name at two shapes is a compile error -- the one that cost
    this probe its entire first run. Checking it here is free and local; finding
    it on the box costs a round trip. It asserts disjointness rather than
    matching types, because two branches that agree today can be edited apart
    tomorrow.
    """
    import ast

    body = src.split("if pid < q_programs:")[1].split("\ndef merged_op")[0]
    q_arm, kv_arm = body.split("    else:\n")

    def bound(text):
        tree = ast.parse("if 1:\n" + text)
        return {
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }

    shared = bound(q_arm) & bound(kv_arm)
    assert not shared, "bound in both branches, will not compile: {}".format(
        sorted(shared)
    )


def run(name, h, ns, early_returns):
    src = CHILD.format(
        repo=REPO,
        h=h,
        ns=ns,
        kv_tail=KV_TAIL_RETURN if early_returns else KV_TAIL_NORET,
    )
    assert_branches_disjoint(src)
    path = "/tmp/probe_merge_{}.py".format(name)
    log = path + ".log"
    with open(path, "w") as f:
        f.write(src)

    timed_out = False
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(
            [sys.executable, path],
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = None
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if timed_out:
        rc = p.wait()
    elapsed = time.time() - t0

    with open(log) as f:
        out = f.read()
    verdict = "NO_RESULT_LINE"
    for line in out.splitlines():
        if line.startswith("[RESULT]"):
            verdict = line[len("[RESULT] "):]
    if timed_out:
        verdict = "TIMED_OUT after {}s".format(TIMEOUT)
    elif rc == -6:
        verdict = "ABORTED (SIGABRT -- compiler assertion, not a UB message)"
    elif rc is not None and rc < 0:
        verdict = "KILLED by signal {}".format(-rc)

    print("=" * 74)
    print("### {}  (H={}, num_stages={}, early_returns={})".format(
        name, h, ns, early_returns))
    print("### returncode={}  elapsed={:.1f}s  VERDICT: {}".format(
        rc, elapsed, verdict))
    lines = out.splitlines()
    # Print it whole when short, both ends otherwise. Never grep: filtering has
    # hidden the real error on this box three times.
    if len(lines) <= 60:
        print("\n".join(lines))
    else:
        print("\n".join(lines[:30]))
        print("   ... {} lines elided ...".format(len(lines) - 55))
        print("\n".join(lines[-25:]))
    print()
    return verdict


def main():
    print("Merging the two launches: does the tiled Q body fit beside KV in UB?")
    print("log files: /tmp/probe_merge_<name>.py.log\n")
    results = []
    for name, h, ns, ret in VARIANTS:
        results.append((name, run(name, h, ns, ret)))
    print("=" * 74)
    print("SUMMARY\n")
    for name, v in results:
        print("  {:<16} {}".format(name, v))
    print("\n[RESULT] MERGE_PROBE_DONE")


main()
sys.stdout.flush()
