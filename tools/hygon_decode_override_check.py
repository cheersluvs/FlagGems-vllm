"""Validate the Hygon decode override's CACHED (direct-launch) path.

The functional tests call each shape once, which exercises only a plan's first
call -- the JIT launches that compile. Every later call of the same shape goes
through cached CompiledKernels launched directly, a different code path that
the tests never reach. So here each (rows, seq_len) case is called three times
with FRESH logits and seq_lens, and every call's answer is checked against
torch.topk on the row's valid range.

Each case is also run on rounded logits. The override picks its threshold from
a sample and falls back, on the device, for a row that admits too few or too
many candidates; random normals never leave that range, so only an input with
few distinct values exercises the fallback at all.

    tools/vendor_probe.sh tools/hygon_decode_override_check.py hygon_decode_override_check
"""

import sys
from importlib import import_module

import torch

import flaggems_vllm

V, K = 262144, 512
ROWS = (1, 4, 8, 16, 24, 32, 56, 79, 496)


def check(logits, seq_lens, idx):
    bad = 0
    for r in range(logits.shape[0]):
        n = int(seq_lens[r])
        kk = min(K, n)
        want = torch.topk(logits[r, :n], kk).values.sort().values
        sel = idx[r, :K]
        valid = sel[(sel >= 0) & (sel < n)].long()
        got = logits[r, valid].sort().values
        pad_ok = int((sel < 0).sum()) == K - kk
        if not (valid.numel() == kk and torch.equal(got, want) and pad_ok):
            bad += 1
    return bad


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    total_bad = 0
    print(f"vocab {V}, top_k {K}; each case called 3x with fresh inputs\n")
    print(f"  {'rows':>4} {'seq_len':<13} {'split':>5}  call1 call2 call3")
    for rows in ROWS:
        for label, seq in (
            ("full", V),
            ("partial", V // 3 + 7),
            ("short", 496),
            ("full tied", V),
            ("partial tied", V // 3 + 7),
        ):
            split = ov._split_factor(rows, V, K)
            cells = []
            for call in range(3):
                torch.manual_seed(1000 * rows + call)
                logits = torch.randn(rows, V, dtype=torch.float32, device=dev)
                if label.endswith("tied"):
                    logits = (logits * 4).round() / 4
                seq_lens = torch.full((rows,), seq, dtype=torch.int32, device=dev)
                if label.startswith("partial"):
                    seq_lens -= torch.arange(rows, dtype=torch.int32, device=dev)
                idx = torch.full((rows, K), -9, dtype=torch.int32, device=dev)
                flaggems_vllm.top_k_per_row_decode(
                    logits, 1, seq_lens, idx, rows, V, 1, K
                )
                torch.cuda.synchronize()
                bad = check(logits, seq_lens, idx)
                total_bad += bad
                cells.append(" ok " if bad == 0 else f"{bad:>3}!")
            print(f"  {rows:>4} {label:<13} {split:>5}  " + "  ".join(cells))
    stages = ("prepare", "select", "tail")
    direct = sum(
        1 for p in ov._PLANS.values() for st in stages if getattr(p, st).runner
    )
    print(f"\n  plans cached: {len(ov._PLANS)}; direct runners: {direct}")
    print(f"  {'ALL CORRECT' if total_bad == 0 else f'{total_bad} WRONG ROWS'}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
