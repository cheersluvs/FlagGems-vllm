"""Is the Ascend override's multi-launch chunking wrong? Decide it in one run.

THE OBSERVATION. Every shape the suite passes needs ONE launch; every shape it
fails needs more than one. heads_per_program is 32, so 8192 tokens at 64 heads
is 24576 programs (one launch, passes) while 32768 is 98304 (two launches,
fails, greatest relative difference 1800 at the last token).

Perfect correlation is not proof. The failing shapes are also the biggest, so
in the suite's own shape list size and chunk-count are confounded. This
separates them.

THE EXPERIMENT. Take a shape that PASSES today and change nothing about it
except the chunk cap. Same input, same kernel, same launch config -- only
MAX_PROGRAMS_PER_LAUNCH moves, which pushes identical work through two or more
launches instead of one. The single-launch result is its own reference, so no
oracle is involved and nothing about torch, the test's fp8 encoder or host
memory can enter the answer.

  cap huge  -> one launch  -> the reference
  cap small -> N launches  -> compare against it

Differ, and chunking is wrong while size is irrelevant. Agree at every cap, and
chunking is exonerated and the large-shape failures are something else -- which
is worth as much, because it kills a plausible story instead of leaving it
standing.

WHAT THIS DELIBERATELY DOES NOT DO. It proposes no mechanism. Four proposed
mechanisms for this operator were refuted by measurement in one sitting; the
rule that came out of that is to measure the split first and explain second.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunked_launch.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES = 512, 64, 584


def main():
    import flaggems_vllm

    # import_module, NOT `from ... import X as mod`. The package's __init__
    # re-exports the FUNCTION under the module's own name, so the `from` form
    # binds a callable and every module attribute below -- the chunk cap this
    # probe exists to move -- raises AttributeError.
    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused"
        ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    impl = mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize
    real_cap = mod.MAX_PROGRAMS_PER_LAUNCH
    print("module MAX_PROGRAMS_PER_LAUNCH = {}".format(real_cap))

    def make(num_tokens, num_heads, block_size=64):
        torch.manual_seed(0)
        q = torch.randn(num_tokens, num_heads, HEAD_DIM,
                        dtype=torch.bfloat16, device=dev)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(num_tokens, dtype=torch.int64, device=dev)
        inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32,
                                              device=dev) / ROPE_DIM))
        t = torch.arange(max(4096, num_tokens), dtype=torch.float32, device=dev)
        fr = torch.einsum("i,j->ij", t, inv)
        cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
        nb = (num_tokens + block_size - 1) // block_size + 1
        slot = torch.arange(num_tokens, dtype=torch.int64, device=dev)
        kc = torch.zeros(nb, block_size * HEAD_BYTES, dtype=torch.uint8, device=dev)
        return q, kv, kc, slot, pos, cs, block_size

    def run(inp, cap):
        q, kv, kc, slot, pos, cs, bs = inp
        q2, kc2 = q.clone(), kc.clone()
        mod.MAX_PROGRAMS_PER_LAUNCH = cap
        try:
            impl(q2, kv, kc2, slot, pos, cs, 1e-6, bs)
            sync()
        finally:
            mod.MAX_PROGRAMS_PER_LAUNCH = real_cap
        return q2, kc2

    print("\n" + "=" * 80)
    print("  Same shape, same input. ONLY the chunk cap changes.")
    print("=" * 80)
    hdr = "  {:>7} {:>5} {:>10} {:>9} {:>11} {:>13} {:>11}"
    print(hdr.format("tokens", "heads", "cap", "launches", "q differs",
                     "max rel diff", "kc differs"))

    clean = True
    for num_tokens, num_heads in ((8192, 64), (8192, 128)):
        inp = make(num_tokens, num_heads)
        hpp = mod.q_heads_per_program(num_heads)
        total = num_tokens * (num_heads // hpp) + num_tokens
        ref_q, ref_kc = run(inp, 1 << 30)
        print(hdr.format(num_tokens, num_heads, "1<<30", 1, 0, 0.0, 0))
        for cap in (total // 2 + 1, total // 3 + 1, 4096, 1024):
            got_q, got_kc = run(inp, cap)
            n_launch = -(-total // cap)
            d = ref_q != got_q
            nd = int(d.sum())
            rel = 0.0
            if nd:
                a, b = ref_q.float(), got_q.float()
                rel = float(((a - b).abs() / a.abs().clamp(min=1e-30))[d].max())
            ndc = int((ref_kc != got_kc).sum())
            print(hdr.format(num_tokens, num_heads, cap, n_launch, nd, rel, ndc))
            if nd:
                tok = d.nonzero()[:, 0]
                uniq = torch.unique(tok)
                print("          differing tokens: first {}, last {}, distinct {}"
                      " of {}".format(int(uniq.min()), int(uniq.max()),
                                      int(uniq.numel()), num_tokens))
            clean = clean and nd == 0 and ndc == 0
        del inp, ref_q, ref_kc
        flaggems_vllm.runtime.torch_device_fn.empty_cache()

    print()
    if clean:
        print("CHUNKING IS EXONERATED -- identical output at every cap, including")
        print("caps that force four or more launches. The large-shape failures are")
        print("something else, and chunking should not be blamed for them.")
        print("\n[RESULT] CHUNKING_CLEAN")
    else:
        print("CHUNKING IS WRONG -- the same work split across launches gives a")
        print("different answer, at a shape the suite currently passes. The")
        print("variable is the number of launches, not the size.")
        print("\n[RESULT] CHUNKING_BROKEN")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
