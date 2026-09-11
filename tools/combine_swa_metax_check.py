"""On-device exactness check for PR #753.

For every case this compares four implementations element-for-element:

  gems      flaggems_vllm.combine_topk_swa_indices (the top-level entry, as the
            PR's test and benchmark bind it)
  fallback  the PR's vectorized torch baseline, extracted VERBATIM from the
            benchmark file with ast -- this tests exactly what the PR ships,
            not a copy of it
  vllm      the vLLM op, when importable and the window is a power of 2 (its
            kernel does tl.arange(0, WINDOW_SIZE), so it cannot take others)
  oracle    a CPU loop transcribed from the reference in tests/

base.Benchmark compares its baseline against the gems output, so the fallback
must be bit-exact for the benchmark to report SUCCESS; this checks that on the
real device, where repeat_interleave / where / clamp all lower through the
vendor's torch rather than CUDA's.
"""
import ast
import os
import sys

import torch

import flaggems_vllm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(
    ROOT, "benchmark", "test_deepseek_v4_attention_combine_topk_swa_indices.py"
)
DEV = flaggems_vllm.device


def load_fallback():
    tree = ast.parse(open(BENCH).read())

    def wanted(n):
        if isinstance(n, ast.FunctionDef):
            return n.name == "_torch_combine_topk_swa_indices"
        return isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TOPK_ALIGNMENT" for t in n.targets
        )

    keep = [n for n in tree.body if wanted(n)]
    if len(keep) != 2:
        sys.exit(f"!! expected _TOPK_ALIGNMENT + the fallback in {BENCH}, got {len(keep)}")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), BENCH, "exec"), ns)
    return ns["_torch_combine_topk_swa_indices"], ns["_TOPK_ALIGNMENT"]


FALLBACK, ALIGN = load_fallback()
GEMS = flaggems_vllm.combine_topk_swa_indices
try:
    from vllm.v1.attention.ops.deepseek_v4_ops import combine_topk_swa_indices as VLLM
except Exception:
    VLLM = None


def oracle(ti, qsl, seq_lens, gather_lens, window, compress, topk, M, N):
    """Per-row transcription of the loop in tests/ (row slices, not elements,
    so 4096-token shapes take seconds rather than minutes)."""
    nt = ti.shape[0]
    ct = (topk + window + ALIGN - 1) // ALIGN * ALIGN
    out = torch.full((nt, ct), -1, dtype=torch.int32)
    lens = torch.empty(nt, dtype=torch.int32)
    q0 = int(qsl[0])
    for b in range(seq_lens.numel()):
        start, end = int(qsl[b]) - q0, int(qsl[b + 1]) - q0
        seq, gat = int(seq_lens[b]), int(gather_lens[b])
        start_pos, gstart = seq - (end - start), seq - gat
        for t in range(start, end):
            pos = start_pos + (t - start)
            tk = min((pos + 1) // compress, topk)
            sw = min(pos + 1, window)
            if tk:
                out[t, :tk] = ti[t, :tk] + M * b
            out[t, tk : tk + sw] = torch.arange(sw, dtype=torch.int32) + (
                M * b + N + pos - sw + 1 - gstart
            )
            lens[t] = tk + sw
    return out, lens


# (query_lens, seq_lens, gather_lens, topk, window, compress, M, N)
CASES = [
    ("bench1", ([3, 2], [6, 4], [4, 3], 4, 4, 2, 20, 8)),
    ("bench2", ([128], [512], [256], 32, 128, 4, 42240, 40960)),
    ("bench3", ([512, 256], [2048, 1024], [1024, 512], 64, 256, 4, 45056, 40960)),
    ("bench4", ([4096], [4096], [4096], 128, 256, 4, 45056, 40960)),
    ("bench5", ([1024, 1024], [8192, 4096], [2048, 1024], 128, 256, 4, 45056, 40960)),
    ("bench6", ([128], [4096], [512], 32, 256, 128, 5632, 1280)),
    ("bench7", ([4096], [4096], [4096], 128, 256, 128, 8448, 1280)),
    ("edge:topk_len=0", ([4], [4], [4], 8, 4, 4, 64, 16)),
    ("edge:window=100", ([64], [256], [128], 16, 100, 4, 1024, 512)),
    ("edge:topk=24", ([64], [256], [256], 24, 32, 4, 1024, 512)),
    ("edge:gather<<seq", ([32], [4096], [64], 32, 64, 4, 8192, 4096)),
    ("edge:pad 2/128", ([256], [1024], [512], 1, 1, 4, 2048, 2048)),
    ("edge:3req,130+2", ([3, 50, 7], [10, 400, 64], [10, 300, 32], 130, 2, 4, 4096, 2048)),
]


def make(spec, gen):
    ql, sv, gv, topk, win, comp, M, N = spec
    qs = [0]
    for q in ql:
        qs.append(qs[-1] + q)
    ti = torch.randint(-1, max(N, 1), (sum(ql), topk), generator=gen, dtype=torch.int32)
    cpu = (
        ti,
        torch.tensor(qs, dtype=torch.int32),
        torch.tensor(sv, dtype=torch.int32),
        torch.tensor(gv, dtype=torch.int32),
        win, comp, topk, M, N,
    )
    on_dev = tuple(x.to(DEV) if torch.is_tensor(x) else x for x in cpu)
    return cpu, on_dev


def diff(got, ref):
    (c, l), (rc, rl) = got, ref
    c, l = c.cpu(), l.cpu()
    if c.shape != rc.shape:
        return f"shape {tuple(c.shape)} vs {tuple(rc.shape)}"
    if not torch.equal(l, rl):
        i = int((l != rl).nonzero()[0])
        return f"lens[{i}] {int(l[i])} vs {int(rl[i])}"
    if not torch.equal(c, rc):
        r, k = (c != rc).nonzero()[0].tolist()
        return f"combined[{r},{k}] {int(c[r, k])} vs {int(rc[r, k])}"
    return None


def main():
    gen = torch.Generator().manual_seed(0)
    print(f"device {DEV}; vLLM op {'importable' if VLLM else 'unavailable'}; fallback from {BENCH}")
    print(f"{'case':<18}{'tokens':>7} {'gems':<6}{'fallback':<10}{'vllm':<6} detail")
    bad = 0
    for name, spec in CASES:
        cpu, dev = make(spec, gen)
        ref = oracle(*cpu)
        res, notes = {}, []
        impls = [("gems", GEMS), ("fallback", FALLBACK)]
        win = spec[4]
        if VLLM is not None and win & (win - 1) == 0:
            impls.append(("vllm", VLLM))
        for label, fn in impls:
            try:
                d = diff(fn(*dev), ref)
            except Exception as e:  # report and keep going
                d = f"raised {type(e).__name__}: {e}"
            res[label] = "ok" if d is None else "FAIL"
            if d is not None:
                notes.append(f"{label}: {d}")
        if VLLM is not None and "vllm" not in res:
            res["vllm"] = "n/a"
        bad += sum(v == "FAIL" for v in res.values())
        print(f"{name:<18}{sum(spec[0]):>7} {res['gems']:<6}{res['fallback']:<10}"
              f"{res.get('vllm', '-'):<6} {'; '.join(notes)}")
    print(f"\n== {'ALL EXACT' if bad == 0 else f'{bad} FAILURE(S)'} "
          f"over {len(CASES)} cases ==")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
