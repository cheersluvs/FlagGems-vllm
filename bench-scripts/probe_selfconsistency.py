"""Run the operator twice on identical input, over the whole test shape grid.

WHY THIS, AND WHY IT REPLACES WHAT CAME BEFORE. Two mistakes have to be undone.

  1. Single-launch stability was measured at ONE shape, 8192 x 64, five runs
     identical -- and then treated as a property of the single-launch path. It
     was not. It showed that shape is self-consistent, never that it is correct.
  2. `18 failed -> 12 failed` was read as improvement, from one pytest run each,
     on a failure already known to be non-deterministic. For a flaky failure a
     single run is not a measurement; a RATE is. The remaining set is ragged --
     32768 and 65536 pass at 64 heads while 98304 and 131072 fail, and a
     17-token case now fails although it needs one launch either way -- and a
     rerun of the twelve gave eleven failures and one pass, flipping again.

So this measures the thing that actually decides what to fix, and separates two
problems that the pytest output cannot tell apart:

  the two runs DIFFER   -> the operator is non-deterministic. That is the defect,
                           it is independent of any oracle, and it is what a fix
                           has to remove.
  the two runs AGREE, and the suite still fails at that shape
                        -> the operator is deterministic and disagrees with the
                           reference. Different problem entirely: tolerance, the
                           oracle, or a stable miscomputation. Nothing about
                           launches or races applies to it.

No oracle, no torch reference, no fp8 encoder on the host, so nothing outside
the kernel can enter the answer -- and it runs in seconds per shape rather than
minutes, which is what makes repeating it affordable.

Every shape in the benchmark grid, three runs each, reporting how many of the
three agree and where they differ.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_selfconsistency.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
RUNS = 3
SHAPES = [(n, h, b) for n in (17, 1024, 8192, 32768, 65536, 98304, 131072)
          for h in (64, 128) for b in (16, 64)]


def main():
    import flaggems_vllm

    impl = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize

    print("  {:>7} {:>5} {:>5} {:>12} {:>12} {:>26}".format(
        "tokens", "heads", "blk", "q differs", "kc differs", "where"))
    print("  " + "-" * 74)

    nondet, det, failed = [], [], []
    for n, h, b in SHAPES:
        try:
            torch.manual_seed(0)
            q0 = torch.randn(n, h, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            pos = torch.arange(n, dtype=torch.int64, device=dev)
            inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2,
                                                  dtype=torch.float32,
                                                  device=dev) / ROPE_DIM))
            t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
            fr = torch.einsum("i,j->ij", t, inv)
            cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
            nb = (n + b - 1) // b + 1
            slot = torch.arange(n, dtype=torch.int64, device=dev)
            kc0 = torch.zeros(nb, b * HEAD_BYTES, dtype=torch.uint8, device=dev)

            outs = []
            for _ in range(RUNS):
                q, kc = q0.clone(), kc0.clone()
                impl(q, kv, kc, slot, pos, cs, EPS, b)
                sync()
                outs.append((q, kc))

            dq = max(int((outs[0][0] != o[0]).sum()) for o in outs[1:])
            dc = max(int((outs[0][1] != o[1]).sum()) for o in outs[1:])
            where = ""
            if dq:
                # reduce before nonzero; the full index list wants gigabytes
                m = outs[0][0] != outs[1][0]
                tok = m.any(2).any(1).nonzero().flatten()
                dim = m.any(0).any(0).nonzero().flatten()
                where = "tok {}..{} ({}), dim {}..{}".format(
                    int(tok.min()), int(tok.max()), int(tok.numel()),
                    int(dim.min()), int(dim.max()))
            print("  {:>7} {:>5} {:>5} {:>12} {:>12} {:>26}".format(
                n, h, b, dq, dc, where))
            (nondet if (dq or dc) else det).append((n, h, b))
            del outs, q0, kv, kc0, cs
        except Exception as e:
            print("  {:>7} {:>5} {:>5}   {}".format(
                n, h, b, str(e).splitlines()[0][:52]))
            failed.append((n, h, b))
        flaggems_vllm.runtime.torch_device_fn.empty_cache()

    print("\n  self-consistent: {} shapes   non-deterministic: {}   errored: {}"
          .format(len(det), len(nondet), len(failed)))
    print()
    if nondet:
        print("THE OPERATOR IS NON-DETERMINISTIC at {} of {} shapes:".format(
            len(nondet), len(SHAPES)))
        for s in nondet[:12]:
            print("    {} tokens, {} heads, block {}".format(*s))
        print("That is the defect, and it is oracle-independent. Note which")
        print("shapes: if they are not the multi-launch ones, the chunk-boundary")
        print("story was wrong and the 2-D grid change addressed the wrong thing.")
        print("\n[RESULT] NONDETERMINISTIC")
    else:
        print("THE OPERATOR IS SELF-CONSISTENT EVERYWHERE, three runs per shape.")
        print("So the remaining suite failures are NOT non-determinism -- the")
        print("kernel computes the same answer every time and that answer")
        print("disagrees with the reference. Stop looking at launches and races;")
        print("compare against the oracle at one small failing shape instead,")
        print("and check the oracle too. 17 tokens failing is the place to start,")
        print("because nothing about size or launch count can be involved there.")
        print("\n[RESULT] SELF_CONSISTENT")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
