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

## Rank-hint fix: report86d59bbf (2026-10-23)

| Project truth | Fix scope |
| --- | --- |
| Failure | First BF16 boundary compilation; no performance samples or new correctness pass |
| Editable | One hint in the private store helper, its pinned SHA and CPU regression checks |
| Read-only | Production/baseline, math, masks, fixed choices, timer and guards |
| Contract | `visible[None,:]` and `limit` have rank2; hints require one entry per dimension |
| Path | Use `(1,4)` for the rank2 rounded limit; keep aligned/fallback routes unchanged |
| Validation | Execute the actual helper AST on rank-aware CPU tensors, check masks/output/canaries; rerun affected local checks |

The former scalar `4` became a one-entry hint and failed rank checking. This is
an error in the diagnostic helper, not a compiler compatibility finding. GPU
compilation and both paired performance cases still require a rerun.

## Paired result4642c871 and alignment-only follow-up

47 report files verified against Git blobs. Both workers passed6 all-row small
checks and pre/post sampled long-case validations; same buffers/TopK, frozen
kernels and common logits checked.44 sampled GPU guards passed. This is a
comparison against retained combined, not a fresh native-vLLM comparison.

| Case | Stage | Original us | Bulk/tail us | Speedup |
| --- | --- | ---: | ---: | ---: |
| BF16 long | score |1734.720|2010.664|0.862760|
| BF16 long | TopK |1478.880|1479.160|0.999811|
| BF16 long | pipeline |3228.280|3477.648|0.928294|
| FP8 long | score |1272.480|1442.352|0.882226|
| FP8 long | TopK |1472.296|1472.664|0.999750|
| FP8 long | pipeline |2779.712|2952.864|0.941361|

All actual kernels remained scalar st.global.b32; static store sites32->64.
BF16 registers234->255, spills0->16, shared81920 unchanged; FP8 registers154->144,
spills0, shared49152 unchanged. Static sites are not dynamic store counts.
Paired score regression is much larger than the sampled A/A envelope
(~0.9909–1.0040 BF16;~0.9976–0.9999 FP8). Reject this exact variant; no production
integration or30-case expansion. This does not disprove all output-store optimizations.

| Project truth | Next isolated step |
| --- | --- |
| Evidence | Actual TTGIR full limit is [1,R] with tt.divisibility=[1,4]; output [key,query] store order0, sizePerThread=[4,1] |
| Inference | Hint1 on continuous key dimension can override inferred divisibility and obstruct mask vectorization |
| Editable | Hint(4,4), pinned SHA, optional compile-only path and tests; no other mask/math/tile changes |
| Contract | Every rounded-limit element is a multiple of4, so4 is valid along both axes; unaligned output still uses exact original JIT |
| Validation | Actual-helper rank/value checks; long-case score checks and small canaries; export actual PTX/TTGIR/resources |
| Benchmark | Compile-only path does not call timer or measure_scores, run TopK stage, or claim speedup; all previous timings remain historical |
| Path | Only if vector stores appear and resources are acceptable, consider a separate paired timing run |

The override behavior is documented in the [upstream v3.6 AxisInfo implementation](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Analysis/AxisInfo.cpp#L1021).
This is supporting source evidence; the exact FlagTree outcome must be verified
in newly emitted PTX/TTGIR, not assumed from upstream source alone.

Compile-only also intercepts the public observer's initial TopK call (records
its arguments but does not execute selection); normal paired benchmarks retain
the observer and all TopK calls. Boundary and sampled score checks still launch
real scorer kernels. Thus compile-only is not a GPU-free/offline command.

Local follow-up validation:31 CPU checks pass, including a compile-only worker
that fails if a performance timer or TopK/pipeline callable is used. Pre-commit
and shell syntax pass. The corrected hint's GPU output remains unverified.
