# BW1000 sparse algorithm pivot: block-private histogram cost gate

The bitonic whole-row trial passed 96 checks but was 4.5–17.7× slower than
the default carry route (`reports/hygon_prefill_bitonic_v1.txt`). Stop that
branch. This probe asks whether a different **hierarchical** first stage can
fit inside the 64×129280 shape's budget. It is *not* a complete top-k operator
and its timing is not an end-to-end speedup.

## G1: project truth

| Field | Frozen value |
| --- | --- |
| Current op | Hygon `top_k_per_row_prefill`, 64×129280/k1024, default single-CTA sparse path vs vLLM C++ baseline |
| Editable / read-only | Only `tools/hygon_prefill_private_hist*` and the trial runner; all `src/`, existing tests, benchmark and reports read-only |
| Build | Python syntax, shell syntax and pre-commit locally; Triton JIT on BW1000 |
| Validation | Private 256-bin counts match live-row lengths on all rows and exact CPU reference bins on four rows; tied, constant, partial, short, special and padded-row cases before timing |
| Benchmark | First-stage CUDA kernel time paired against vLLM's full prefill kernel on identical normal inputs, A-B-B-A/reverse, three rounds × two seeds; report **stage fraction**, not an operator speedup |
| Active set | fp32/int32, column-contiguous padded rows; B1024/2048/4096 and 4–16 warps on 64×129280/k1024 |
| Aggregation | Per-config median stage microseconds, baseline microseconds, both-seed fractions and raw samples |
| Timeout / profiler | 1200 s per worker, 14400 s wrapper; exact one CUDA kernel event per call; `hy-smi` before/after |
| Autotune / fallback | Offline Hygon cost gate only. Production stays untouched; no torch compute fallback |

## G2: contract

| Field | Frozen value |
| --- | --- |
| Input | fp32 logits, int32 starts/ends, `stride1=1`, padded `stride0>=vocab`, valid `[start,end)` ranges |
| Output of this stage | Scratch `[rows,chunks,256]` int32 histogram; each live element counted exactly once. No top-k indices are produced |
| Key | Shipped STEP-0 float16 ordered key narrowed from 11 to 8 bits (`mapped >> 8`); exact key agreement checked on GPU/CPU |
| Boundaries | Empty, short, partial, ties, infinities, signed zero and padded rows checked; NaNs/other dtype outside scope |
| Torch use | Only probe setup, reference, validation, profiler and vLLM comparison; never production computation |

## G3: path and decision

| Path | Trigger | Mechanism / risk |
| --- | --- | --- |
| Sparse first stage | 64×129280 | Many CTAs, each reads one chunk and emits a **private** 256-bin histogram to its own scratch slice. No cross-CTA atomic contention and no per-element global atomic |
| Future exact path | Only if this cost gate passes | Reduce chunk histograms, locate the threshold bin, collect candidates, refine exact values; bins larger than capacity must trigger refinement/fallback, never truncation |
| Old paths | All other shapes | Unchanged defaults; old row split and stage split repeated radix or used cross-CTA per-element atomics and are not this algorithm |

The vLLM C++ baseline is around 109 µs and the current sparse kernel around
262 µs in recent full-operator reports. If this **first stage alone** costs
~50 µs or more, leaving enough budget for reduction, collection and exact
refinement to beat vLLM becomes doubtful. This is a screening threshold, not
a proven performance bound. The report also records coarse threshold-bin
sizes; any >2048 makes a direct fixed-capacity final step invalid.

Remote invocation after fetching this probe commit:

```bash
tools/hygon_prefill_next_run.sh private_hist hygon_prefill_private_hist_v1
```
