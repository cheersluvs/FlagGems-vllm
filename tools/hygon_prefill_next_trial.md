# BW1000 follow-up: four-row launch, then rank8 × carry

This is a **measurement-only** trial on top of the default-carry production
commit and `reports/hygon_prefill_carry_v2.txt`. No production source, public
dispatch, functional test, existing benchmark, or vLLM baseline is modified.
Run the two stages separately; inspect the launch report before starting combo.

## G1: project truth

| Field | Frozen value |
| --- | --- |
| Current op | Hygon `flaggems_vllm.top_k_per_row_prefill` |
| Editable / read-only | Only `tools/hygon_prefill_next*` and audit source-construction tools; `src/`, existing test and benchmark, vLLM baseline and prior reports are read-only |
| Build | Local Python syntax, source-construction tests, shell syntax; Triton JIT on BW1000; no package install |
| Validation | Exact `torch.topk` value multiset, unique relative indices, `-1` padding, output guards, public route; normal seeds 42/43 and tied/constant/partial/short/special/padded-row cases, **before** timing each candidate |
| Benchmark | Torch profiler CUDA kernel events, exactly one `non_tle_top_k_per_row_prefill` event per call; median microseconds; A-B-B-A and reverse order over four rounds and two seeds; ratios >1 favor candidate |
| Active set | Launch: `(4,8193,512)` and `(4,16385,512)`, seven candidate geometries vs shipped B512/w8. Combo: four dense benchmark shapes, four factorial arms, using production geometry |
| Aggregation | Per-shape median of eight paired ratios, range and raw per-call samples retained; no winner selected from unvalidated or failed workers; no seven-shape geomean claim before the ordinary benchmark |
| Timeout / profiler | 1200 s per shape/config worker, 14400 s wrapper. `hy-smi` before/after. Device-only profiler events, not host launch time |
| Autotune | Exempt: Hygon-only offline probe, no production tuning or NVIDIA config |
| Fallback | Production stays at B512/w8 sparse and current carry dense. No torch compute fallback is added |

## G2: input/output contract

| Field | Contract |
| --- | --- |
| Reference | Exact `torch.topk` value multiset over the live `[row_start,row_end)` interval; indices are relative to row start |
| Device / dtype | BW1000, fp32 logits; int32 starts, ends and output; forward-only |
| Layout | Column-contiguous `stride1=1`; row padding via `stride0>=vocab` is valid. Column stride 2 is outside the current implementation contract |
| Output | Caller-owned `[rows,k]` int32, unique valid indices and `-1` for short/empty intervals; no aliasing or promotion change |
| Semantics | Ties, constants, infinities, signed zeros, partial/short ranges checked; NaN and non-fp32 remain unestablished/unsupported |
| Torch use | Only probe inputs, oracle, checking, and profiler; not production computation |

## G3: paths and decision gates

| Path | Trigger | Measurement and risk |
| --- | --- | --- |
| Four-row sparse | `vocab>10*k`, 4 rows | Directly launch the shipped sparse kernel with B256/w2,w4,w8; B512/w4,w16; B1024/w8,w16, each paired with B512/w8. Separate subprocesses contain JIT faults. Old `num_warps=1` wrong answers and B1024/w2,w4 faults are excluded |
| Dense factorial | `vocab<=10*k` on four dense benchmark shapes | Compare control→rank8, control→carry, carry→rank8_carry. The carry source must exactly equal the default production source. `rank8_carry` differs from carry only in the final rank job |
| General/short/tied | Existing kernel paths | Adversarial checks use the same block/warp setting; 5 rows for four-row candidates to include short lengths 0,1,k−1,k,k+1 |
| Unsupported | Other dtype, NaN, non-column-contiguous | No coverage expansion or fallback |

The launch stage is a diagnosis of the two below-1.0 sparse four-row shapes,
not permission to change geometry from a single noisy result. A candidate must
pass every correctness case, improve both seeds and both shapes by more than
~2%, and then survive a separate ordinary seven-shape kernel benchmark before
promotion. For rank8+carry, report the **incremental** gain against carry and
its interaction with rank8 alone. A shape gate is needed if any dense shape
regresses; the previous rank8-alone audit had a large loss on 4100×1025.

On the Hygon worktree after fetching the new commit:

```bash
tools/hygon_prefill_next_run.sh launch hygon_prefill_launch_device_v1
# Inspect the pushed launch report before the next stage.
tools/hygon_prefill_next_run.sh combo hygon_prefill_combo_device_v1
```

The runner always commits/pushes its report, including a
failed or timed-out probe; a nonzero probe exit does not mean the report was
lost. Run only on an idle HCU and set `HIP_VISIBLE_DEVICES` if necessary.
