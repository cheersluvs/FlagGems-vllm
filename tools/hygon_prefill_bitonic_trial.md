# BW1000 algorithm pivot: on-chip bitonic threshold

This is a measurement-only feasibility test. The default Hygon implementation
and the vLLM baseline remain untouched. The previous geometry sweep found no
four-row gain; rank8+carry added only ~1.5% to the seven-shape geometric mean.

## G1: project truth

| Field | Frozen value |
| --- | --- |
| Current op | `flaggems_vllm.top_k_per_row_prefill` on BW1000; vLLM 0.18.1 C++ op is the baseline |
| Editable / read-only | Only this trial document and `tools/hygon_prefill_bitonic*`; `src/`, existing tests/benchmark and prior reports are read-only |
| Build | Python syntax and pre-commit on Mac, Triton 3.6 JIT on BW1000; no package install |
| Validation | Exact `torch.topk` value multiset, unique relative indices, `-1` short-row padding, output guards; full normal seeds 42/43 plus tied/constant/partial/short/special/padded-row before timing |
| Benchmark | Per-kernel CUDA profiler events, paired A-B-B-A/reverse, three rounds × two seeds; candidate/default device-us ratio >1 is improvement |
| Active set | First feasibility gate: four dense benchmark shapes, fp32 logits, int32 bounds/output, column-contiguous with row padding; B2048/4096/8192, 4–16 warps |
| Aggregation | Per-shape, per-warp median and both-seed medians; no cherry-picked minima and no seven-shape speedup claim yet |
| Timeout / profiler | 1200 s per worker, 10800 s wrapper; `hy-smi` before/after; exactly one kernel event per call |
| Autotune | Offline Hygon experiment only; no production/NVIDIA tuning changes |
| Fallback | No production change; ordinary carry route remains the only shipped path; no torch compute fallback |

## G2: contract

| Field | Frozen value |
| --- | --- |
| Reference | Top-k value multiset for each `[row_start,row_end)` interval; output indices relative to row start, arbitrary order |
| Device/dtype | BW1000; fp32 logits, int32 metadata/output, forward only |
| Layout | `stride1=1`; `stride0>=vocab`, including padded rows |
| Boundary | `row_len<=k` emits every valid relative index then `-1`; ties/infinities/signed zero/partial spans checked |
| Unsupported | NaN, other dtypes, non-column-contiguous input remain outside this trial |
| Torch usage | Probe inputs/oracle/profiler only, never production computation |

## G3: algorithmic paths

| Path | Trigger | Mechanism / risk |
| --- | --- | --- |
| On-chip dense candidate | `row_len>k`, vocab up to 5115 in this first gate | One CTA loads the row once, `tl.topk` obtains the kth value, strict-better mask plus deterministic equal-value quota, `tl.cumsum` writes indices. No 2048-bin global histogram or second row read. Risk: register pressure and bitonic network cost |
| Short/empty | `row_len<=k` | Direct relative indices plus `-1`, no sort |
| Shipped dense comparison | Existing carry route at production geometry | Same inputs, paired single-kernel timing and exact checks |
| Sparse wide future path | 4×8193/16385 and 64×129280 | Not measured in this gate: whole-row bitonic size/occupancy is likely poor. If dense gate passes, test chunked exact candidate selection with an overflow certificate and fallback |
| Unsupported | NaN, other dtype/layout | No promise expansion |

Stop this branch if the complete candidate is slower than carry on the dense
shapes. The more ambitious sparse design would be hierarchical: per-chunk
bitonic top-C, merge candidates, certify that no discarded value beats the
global kth threshold, and use the old exact path on certification failure.
That design is a hypothesis, not a measured win; a fixed C without the
certificate/fallback is not exact and must not be shipped.

Remote invocation after fetching the trial commit:

```bash
tools/hygon_prefill_next_run.sh bitonic hygon_prefill_bitonic_v1
```
