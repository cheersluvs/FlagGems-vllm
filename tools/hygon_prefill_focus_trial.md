# BW1000 gaps-v1 focused follow-up

`hygon_prefill_gaps_v1` completed all 182 workers, but two geometry workers
failed exact-value validation at `(1280, 16385)` and `(2560, 16385)`, both
configuration 0. At `(64, 129280)`, identical `[512, 8]` source/configuration
plans measured roughly 1.29x apart, so the apparent 1.36x gain from
`[1024, 16]` cannot yet be promoted.

The focus probe makes no production changes. It uses the same source digest,
input contract, exact torch.topk oracle, output guards, and profiler timing as
the original audit. Each worker is a separate process, with a 1200-second
timeout. The wrapper commits and pushes the report even if a worker fails.

Run on an idle BW1000 checkout of `codex/hygon-prefill-audit` after pulling the
focus-probe commit:

```bash
tools/hygon_prefill_next_run.sh focus hygon_prefill_focus_v1
```

The six workers are:

1. Four correctness workers: two failing shapes × geometry configurations 0
   and 1. Each checks adversarial cases, then repeats exact-value validation
   with seeds 42 and 43 four times for both default and candidate launches.
   `focus_mismatch` records the first bad row, wrong values, range, and
   refinement counters; failure does not suppress the other workers.
2. Two timing workers at `(64, 129280, 1024, padded-row stride)`, testing
   `[512, 8]` against itself and against `[1024, 16]`. Each seed measures the
   same plan under two labels, separate same-config plans, then the same plans
   after swapping their scratch buffers. The geometry candidate is compared
   against both reference workspaces. Ratios greater than 1 favor the
   candidate, but only after the same-config ratios are close to 1.

Interpretation: a same-object ratio materially different from 1 indicates a
timing method problem; a separate-plan ratio that changes sign after swapping
scratch suggests placement sensitivity; persistent mismatch in the default
launch is a correctness issue independent of geometry. Do not change production
dispatch until these checks and a separate ordinary benchmark pass.
