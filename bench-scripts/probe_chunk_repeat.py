"""Is the chunk-boundary defect deterministic? Everything downstream depends on it.

WHAT IS ESTABLISHED, and is not re-tested here.

  * Splitting the same work across launches corrupts the FIRST PROGRAM of every
    launch after the first, RoPE region only, 2048 elements, NoPE and k_cache
    clean. Predicted bad tokens matched observed on 8 of 8 configurations.
  * The bad program really does rotate -- solved cos^2+sin^2 = 0.9997 -- so the
    rotation arithmetic and the RMS scale are not at fault.
  * The cos/sin it uses are genuine table rows. A null control settled this:
    random angle vectors the table never held match at 2.78-3.00, while the
    observed matches sit at 1.1e-2 to 2.2e-2, two hundred times closer.
  * Those rows are small integer multiples of the token index -- 4x, 5x, 6x of
    6144, and 5x of 4096.

WHAT IS NOT ESTABLISHED, AND WHY IT DECIDES THE FIX. Two runs with the same
seed, the same cap and the same table size returned DIFFERENT multiples --
[30720, 36864] once and [24576, 30720] the next. Either the defect is
non-deterministic, or something outside the declared inputs differs between
runs. Those need opposite fixes:

  deterministic  -> one address expression is wrong; fix the arithmetic
  varying        -> a race; no index correction can help, and the suspect is the
                    masked NoPE store that precedes the RoPE re-load inside the
                    same program, on a backend where a masked store with
                    duplicate lane addresses is silently dropped and a load mask
                    with a runtime row offset silently returns wrong data. That
                    ordering needs a barrier, not a new index.

Guessing between those two is exactly the move that has already been refuted
four times on this operator, so it is measured instead.

THE TEST. One process, one allocation of every input, the same cap run five
times, compared against each other rather than against a reference. Identical
five times over is evidence of determinism at this shape; any disagreement ends
the question immediately. num_warps is not varied -- the override pins it at 1
-- but the launch count is, because that is the knob already known to matter.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_repeat.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, NOPE_DIM, ROPE_DIM, HALF = 512, 448, 64, 32
HEAD_BYTES, EPS = 584, 1e-6
NUM_TOKENS, NUM_HEADS, BLOCK = 8192, 64, 64
CS_ROWS = 8 * NUM_TOKENS
REPEATS = 5


def main():
    import flaggems_vllm

    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused"
        ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    impl = mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize
    real_cap = mod.MAX_PROGRAMS_PER_LAUNCH

    torch.manual_seed(0)
    q0 = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32,
                                          device=dev) / ROPE_DIM))
    t = torch.arange(CS_ROWS, dtype=torch.float32, device=dev)
    fr = torch.einsum("i,j->ij", t, inv)
    cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
    nb = (NUM_TOKENS + BLOCK - 1) // BLOCK + 1
    slot = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    kc = torch.zeros(nb, BLOCK * HEAD_BYTES, dtype=torch.uint8, device=dev)

    def run(cap):
        qq, kk = q0.clone(), kc.clone()
        mod.MAX_PROGRAMS_PER_LAUNCH = cap
        try:
            impl(qq, kv, kk, slot, pos, cs, EPS, BLOCK)
            sync()
        finally:
            mod.MAX_PROGRAMS_PER_LAUNCH = real_cap
        return qq

    print("=" * 76)
    print("Same inputs, same cap, {} times. Compared against each other.".format(REPEATS))
    print("=" * 76)

    verdicts = {}
    for cap in (1 << 30, 12289, 8193, 4096):
        outs = [run(cap) for _ in range(REPEATS)]
        base = outs[0]
        print("\n  cap = {}".format("1<<30 (single launch)" if cap > 1 << 20 else cap))
        stable = True
        for i, o in enumerate(outs[1:], start=2):
            nd = int((base != o).sum())
            stable = stable and nd == 0
            print("    run 1 vs run {}:  differing elements {}".format(i, nd))
        # where the runs disagree, if they do
        if not stable:
            acc = torch.zeros_like(base, dtype=torch.bool)
            for o in outs[1:]:
                acc |= (base != o)
            idx = acc.nonzero()
            print("    run-to-run disagreement at tokens {}, dims {}..{}".format
                  (sorted(set(idx[:, 0].tolist()))[:6],
                   int(idx[:, 2].min()), int(idx[:, 2].max())))
        verdicts[cap] = stable
        print("    -> {}".format("identical every time" if stable
                                 else "NOT REPEATABLE"))
        del outs, base
        flaggems_vllm.runtime.torch_device_fn.empty_cache()

    multi = [c for c in verdicts if c <= (1 << 20)]
    print("\n" + "=" * 76)
    if verdicts[1 << 30] and all(verdicts[c] for c in multi):
        print("DETERMINISTIC at every cap. The corruption is reproducible, so it is")
        print("an address expression, not a race. The differing multiples seen")
        print("between the two earlier probes came from something else those runs")
        print("did not hold fixed -- find that before trusting either number, but")
        print("fix the arithmetic, not the ordering.")
        print("\n[RESULT] DETERMINISTIC")
    elif not verdicts[1 << 30]:
        print("EVEN THE SINGLE LAUNCH IS NOT REPEATABLE. The chunk boundary is not")
        print("the whole story and may not be the cause at all -- an unrepeatable")
        print("baseline cannot support any of the per-boundary conclusions. Settle")
        print("this before anything else.")
        print("\n[RESULT] BASELINE_UNSTABLE")
    else:
        print("NOT REPEATABLE when chunked, repeatable when not. That is a race,")
        print("and no index correction can fix it. The suspect is the masked NoPE")
        print("store that precedes the RoPE re-load in the same program: this")
        print("backend silently drops a masked store with duplicate lane addresses")
        print("and silently returns wrong data for a load mask with a runtime row")
        print("offset. Try tl.debug_barrier between the store and the re-load, or")
        print("keep the pairs in registers and never re-read q.")
        print("\n[RESULT] RACE_WHEN_CHUNKED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
