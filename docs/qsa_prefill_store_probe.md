# QSA bulk/tail output-store probe (2026-10-23)

Pre-coding board. Report d2464dd78225d8439f8d1c5a4e9ce970104a5b3a
contains75 verified files; both long captures passed and all six scorer cubins
have matching PTX/TTGIR/TTIR/LLIR/SASS. Across24 launches store-sector bytes /
useful output bytes is~4. BF16 and FP8 retained candidate emit scalar stores.
This is an L1 transaction observation, not DRAM amplification or a4x speedup.

| Project truth | Frozen scope |
| --- | --- |
| Editable | Private scorer copy changing only output-store masks/helper; fixed-choice diagnostic, CPU tests and wrapper |
| Read-only | Production, original combined scorer, native reference, configs, timer, GPU guards and tolerances |
| Hypothesis | Arbitrary per-row visible lengths obstruct vector-store masks; split full groups of4 from the0–3 tail elements |
| Active set | BF16 n1/q16384/p102400 and FP8 long100k; four real512MiB-budget chunks each; exact original combined choices incl FP8 P2 |
| Baseline | Retained query-stage-joint scorer, not the small-gain BF16 query-split candidate |
| Benchmark | Existing --mode kernel timer,2 alternating rounds; score, unchanged TopK and sequential pipeline; same buffers and paired A/A on score/pipeline |
| Profiler | No NCU rerun initially; save actual compiled PTX/TTGIR and static store instruction widths before/after; assess resources and paired timings |
| Runtime | Existing FlagTree /opt/flagtree, pinned ptxas13.4, CUDAcompat and vLLM0.31 virtualenv; require captured compiler identity |
| Validation | All-row small BF16/FP8 mask/canary tests plus sampled real-case reference and TopK checks immediately before/after timing |
| Autotune | Exempt private one-hypothesis probe: exact saved per-chunk choices, no candidate expansion or retuning |
| Timeout |1800s per case; existing exclusive GPU guards,80C admission,120s cooldown |

| Contract | Preserved |
| --- | --- |
| Math | Same full-D dot, head ReLU and reduction order; stores retain invalid-page -inf |
| Output | Write each valid visible element exactly once; bulk and tail masks disjoint, union equals original; leave masked elements and canaries untouched |
| Alignment | Bulk path only if output is contiguous FP32, base16B aligned, row width divisible by4; exact original JIT otherwise |
| Layout | Query/cache/table strides, chunk offsets, output row pitch and memory budget unchanged |
| TopK | Same persistent_topk function and common-score TopK callable; no TopK optimization |

| Path | Decision |
| --- | --- |
| Aligned real chunks | Private bulk/tail scorer, same tile/query/stages/warps |
| Odd width/misaligned output | Exact original scorer; validate compiled-binary fallback identity |
| Small tests | Visible residues0–3, zero, boundary clipping, request/chunk tails, padded input strides, nonzero query base and output canaries |
| Evidence | Check emitted vector instructions; if scalar remains, do not claim vectorization; if slower despite vectorization, reject |
| Integration | Deferred until same-run probe advantage survives regression; production untouched |

The alignment hint follows [Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.multiple_of.html).
Vectorization is a hypothesis to verify in actual generated code, not guaranteed
by a source hint. Optional static instruction counts are not hardware counters.

## Execution and local checks

```bash
export QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python
bash tools/run_qsa_prefill_store_probe_h800.sh \
  --baseline-report /tmp/qsa-profiling-h800.Ji2cwS \
  --flagtree /opt/flagtree --rounds 2
```

The wrapper uses the existing verified CUDA13.4 compatibility bootstrap, prints
all stage timings and paired A/A ranges plus actual static store instruction
forms and register/shared/spill resources, and uploads reports even on failure.
PTX and TTGIR come from actual compiled callables. Do not compare profiler replay
durations with these timings. The new source uses standard Triton operations;
no NVIDIA-specific instruction/API is added by the bulk/tail change.

Local validation:30 CPU checks pass across this probe, the preceding query
probe, fixed-query plumbing and the shared score benchmark timer. Checks include
AST equivalence of all scoring math, exact output-mask partition, strict original
fixed choices, alignment routing without changing launch arguments, unchanged
TopK/common buffers, alternating paired timing and stopping on numerical failure.
Pre-commit and shell syntax pass. Both actual archived fixture metadata records
load successfully. GPU compilation/correctness/performance remain pending;
scalar stores may remain or extra tail instructions may outweigh vector gains.

No production changes or formal30-case acceptance claim are made here. If the
probe works, the next step is measured regression coverage before integration.

Both workers explicitly select from scored-prefill; the default workload suite
is upstream-only and omits FP8. A regression check exercises the FP8 worker.
