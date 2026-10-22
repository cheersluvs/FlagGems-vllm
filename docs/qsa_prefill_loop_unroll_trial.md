# Private score-loop unroll trial

Pre-coding scope,2026-10-22. The audited joint-stage candidate report e861c93c
has pipeline/native mean1.183323 but five cases below0.98. Hypothesis: unroll
adjacent key iterations by two to expose load/dot scheduling opportunities,
without changing arithmetic within an iteration or query/key reuse semantics.
This is a portable Triton loop transform; performance remains backend dependent.

| Project truth | Frozen decision |
| --- | --- |
| Op/editable | Private loop-unroll kernel/factory/tests/runner,explicit benchmark routing and source archive,accuracy selector and this document |
| Read-only | Canonical production,combined and stage-joint factories/kernels/configs,TopK,workloads,pinned reference,timers,guards,runtime profile |
| Build | Python/shell syntax,CPU regression tests,changed-file pre-commit; H800 JIT through existing FlagTree runner |
| Validation | Existing40 canonical iteration nodes +22 candidate nodes; every34/20 dtype-pruned config,both depths/query modes,page/chunk boundaries; full before integration |
| Benchmark | Frozen30 scored-prefill cases,kernel mode,2 rounds,200ms warmup/100ms repetition; native score/TopK/pipeline plus same-run stage-joint score/pipeline |
| Acceptance | Arithmetic mean native/pipeline >1.1,each >=1 or proposed near-one floor0.98; retain all stage data,do not optimize TopK |
| Timeout/profiler | Existing1800s case/7200s accuracy/900s cooldown; compiled resource/PTX metadata,no local CUDA profiler |
| Autotune | Reuse exact stage-joint34 YAML choices,key,strategy/pruner; fixed unroll2 is the sole algorithmic alternative,not a new tunable parameter; no new config group |
| Fallback | No torch compute fallback; unsupported behavior unchanged |

| Contract | Decision |
| --- | --- |
| Reference | Existing pinned native vLLM QSA source7a877ae; not a standalone torch API |
| Input | Existing BF16/FP8,H/D,page and query/cache/table strides,request offsets,visible bounds,invalid pages |
| Output | Existing FP32 logits,unchanged in-place index output and alias contract |
| Arithmetic | Full-D FP32 dot accumulation,ReLU before head sum,unscaled; no reassociation between keys |
| Boundaries | Dynamic loop bounds/tails and all masks unchanged; tl.range handles remaining single iteration |
| Autograd/unsupported | Forward-only; existing unsupported validation retained |
| Torch | Reference computation only in tests; production clone uses only existing allowed metadata/allocation/view paths |

| Path | Trigger | Implementation/config | Verification/risk |
| --- | --- | --- | --- |
| Empty/select-all | Existing metadata gates | Unchanged host/TopK | Existing canonical checks |
| Rectangular | Existing rectangular dispatch | Exact combined scorer,only range -> tl.range(loop_unroll_factor=2),same joint-stage34 choices | Both depths/modes/every choice,page/loop tails; register/code size growth or compiler scheduling regression possible |
| Compact | Existing compact dispatch | Unchanged canonical compact scorer | Existing compact/full-entry checks |
| Unsupported | Existing contract | Same explicit errors | No fallback introduced |

Keep/revert is pending same-run H800 correctness and performance. A superset
of configurations does not guarantee improvement. The previous stage-joint
baseline is separately instantiated with original source and independent outer
entry/output buffers; shared LibTuner per-configuration caching behavior is not
changed. Production exports remain canonical. Full candidate correctness and
production-entry validation are required before integration.

The transform uses the documented portable `tl.range` loop-unroll attribute:
https://triton-lang.org/main/python-api/generated/triton.language.range.html .
It does not set loop-level num_stages (which differs from the existing kernel
launch parameter). The old loop body is AST-identical,including its stores,
page lookup and split-query math. Existing every-config tests use visible1537
with BLOCK_N64/128,so the dynamic last loop group includes odd iteration counts
(e.g.25 key tiles at BLOCK_N64); output sentinels verify masked columns stay
untouched. Both launch depths and both eligible FP8 query modes are forced.

Local validation:95 related CPU regression checks passed,including seven new
loop-transform/factory/baseline/acceptance/coverage/report/CLI checks. Both shell
runners pass bash -n. Changed-file pre-commit and diff checks pass. AST checks
confirm the old loop body and entire GPU-test logic are retained; baseline
production,combined kernel,stage-joint factory and tuning YAML match the parent
byte-for-byte. Core geometry/numerics/environment/timer/guard helpers are
unchanged. Reports archive all new implementation and test sources. No local
Torch/Triton/CUDA execution is available: H800 compilation,numerical correctness,
register/shared/spill changes and speedups are pending. Status: private trial;
no production integration or performance gain is claimed.

## Interrupted H800 result:711010b7 (2026-10-22 06:43:52 UTC)

Report branch `codex/qsa-h800-results-20261022-064352`,commit
`711010b7d85edc2fcb69957b08eca1c5a9baa4df`,tested clean source
`8ca1fa9c5e34d09a9c5b2494660537ae07774d18`,exit2. Verified34 archived
files by Git blob hash/size and118 source snapshots against the tested commit.
Pinned native reference,compiler/runtime/loaded compatibility identities match.

All62 iteration correctness nodes passed in450.33s. No compilation or numerical
failure is reported. Eight complete BF16 cases passed their before/after numeric,
AB/BA paired timing,source/config/TopK and sampled GPU guard checks. Audited56
timed stage rows,224 raw samples and168 guard observations for those eight;
sampled temperatures35–53C,SM clocks1965–1980MHz. They remain partial
measurements,not full30-case acceptance. Sampling cannot prove continuous
exclusive GPU ownership between observations.

The ninth case `perf_n2_q8192_p102400_bfloat16` failed during a post-round
GPU guard check: unowned compute PID2551335,664MiB. Its partial score and any
other measurements must not be retained as a completed case. At the following
case's idle gate,GPU utilization84%,memory1046MiB and compute processes were
observed,causing environment_rejected/exit2. Do not infer ownership of the PID
from its number or stop jobs on the shared machine. The report contains nine
case records,eight passed and one failed;the other21 were never started.
Overall measurement_valid=false,target_met=null is correct. No30-case mean
can be reported.

Arithmetic means for the eight completed cases only:

| Comparison | Score | TopK | Sequential pipeline |
| --- | ---: | ---: | ---: |
| Native/candidate | 1.114695 | 0.902298 | 0.994624 |
| Previous joint-stage/candidate | 0.925928 | Unchanged code | 0.971818 |

All eight cases regress versus the same-run joint-stage baseline in both paired
rounds for both score and pipeline. Mean score speedup0.925928 means roughly
8.00% greater candidate score latency under the mean-ratio representation;
pipeline0.971818 similarly corresponds to roughly2.90% greater latency. These
are reciprocals of means,not means of per-case latency increases. TopK remains
unchanged and its native comparison is retained rather than optimized.

| Case | Native/score | Native/TopK | Native/pipeline | Joint-stage/score | Joint-stage/pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 1.200252 | 0.962512 | 1.029069 | 0.926909 | 0.970606 |
| perf_n1_q8192_p51200_bfloat16 | 1.068113 | 0.909738 | 0.996043 | 0.921856 | 0.975717 |
| perf_n1_q8192_p102400_bfloat16 | 1.104633 | 0.803671 | 0.942727 | 0.909031 | 0.950780 |
| perf_n1_q16384_p8192_bfloat16 | 1.079287 | 0.995892 | 1.031155 | 0.933504 | 0.977759 |
| perf_n1_q16384_p51200_bfloat16 | 1.124182 | 0.888106 | 0.991818 | 0.923076 | 0.973573 |
| perf_n1_q16384_p102400_bfloat16 | 1.085191 | 0.782956 | 0.935098 | 0.912133 | 0.957550 |
| perf_n2_q8192_p8192_bfloat16 | 1.116912 | 0.963257 | 1.009271 | 0.951148 | 0.978592 |
| perf_n2_q8192_p51200_bfloat16 | 1.138992 | 0.912255 | 1.021807 | 0.929770 | 0.989969 |

Decision: reject enabling unroll2 globally;retain the previous joint-stage private
candidate as the optimization baseline. No production entry has been changed.
Keep this experimental code and raw report for reproduction,without promoting
it. No need to spend another22-case run establishing a global keep decision
that the eight consistent regressions already rule out. This does not establish
FP8 behavior: all completed cases are BF16. If unrolling is investigated further,
first isolate it to FP8 while BF16 uses the original JIT unchanged,and benchmark
against joint-stage on the same verified runtime. Recheck BF16 as controls and
retain all TopK/pipeline stages. Do not weaken GPU guards or treat incomplete
measurements as accepted to finish a report. No kernel/toolchain change is
needed to explain this interruption.
