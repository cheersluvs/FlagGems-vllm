# Joint score pipeline-depth selection

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Op/reference | Existing qsa_select_paged_prefill; pinned native vLLM7a877ae,canonical SHA256 f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c and audited combined kernel SHA2561ef12d431b65971c5617f0a34f614360008c652a36125d5679da00206b212d7c; no single Torch API |
| Editable | New private stage-joint factory,YAML group,GPU/CPU tests,runner,this document; optional routing/profile/source/report hooks; existing stage1 CPU YAML pin only excludes the newly owned group |
| Read-only | Canonical production,exports,all existing kernels/factories/YAML groups,TopK;30 workloads; original timer,guard,reference,tolerances and validation helpers |
| Build/validation | Local syntax/config/cache/CLI/acceptance/source invariants and pre-commit;62-node H800 iteration profile,all required to pass; full canonical +40 new candidate nodes available before integration |
| Active/metric | Frozen30 scored-prefill; same verified FlagTree/ptxas13.4/CUDA13.4compat,--mode kernel us; score,TopK,sequential pipeline retained |
| Aggregation | Equal-weight arithmetic mean of30 native pipeline ratios>1.1,every case near1 (proposed floor0.98); experimental,not production acceptance |
| Baselines | Existing native and canonical stages; original combined stage2 via unchanged previous-tuning machinery,own caches/output/logits,score+pipeline comparisons; fixed query-mode and short_batch3 swap retained |
| Runtime |2 rounds,warmup200ms/rep100ms,initial80C;case1800s,accuracy7200s,cooldown900s/10s;automatic complete/partial upload |
| Autotune | New qsa_select_paged_prefill_stage_joint_score: exact17 original stage2 entries followed by exact17 stage1 entries; existing key/strategy/query-mode pruner;BF1620/FP834 choices; compact original6 |
| Profiler/evidence | Prior report f72b4b7:upstream/FP8 pipeline versus combined1.041904/1.011425,but observed BF16 three repeatable regressions; shared reduction alone did not ensure gain. Reuse actual per-chunk configs/resources; hardware counters unavailable locally |
| Fallback | No Torch compute fallback,workspace,asm,policy,math,TopK or production change |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn,H/D,integer metadata,query/cache/table strides,bases,visible bounds,invalid pages and chunks |
| Outputs | Existing int32 indices,-1 padding,caller buffers,None return |
| Math/promotion | Exact combined full-D FP32 dot,per-head ReLU before real-head reduction; no scaling,reduction split or tolerance change |
| Host/alias/layout | Exact canonical host bytecode,dispatch,early returns,compact routing and buffer contract |
| Scope | Forward only;unchanged unsupported/special-value/dtype/autograd scope |
| Torch | References only in existing tests/benchmark;production never imports this private factory |

## Paths

| Path | Trigger | Implementation/config | Validation/risk |
| --- | --- | --- | --- |
| Rectangular | Existing canonical dispatch | Same audited combined JIT,independent entry;20 BF16/34 FP8 stage/query/tile choices | Every config at four H/D pairs,both stage depths in forced mode/page/tail/full-entry checks;no fixture-name gate |
| Compact | Existing ragged envelope | Exact canonical compact kernel/configs | Compact/full-entry coverage retained;stage2 unchanged |
| Early/unsupported | Existing metadata gates | Exact canonical host | Original contract nodes;no compute fallback |
| Previous baseline | Explicit combined comparison | Original combined17 stage2 choices,original factory | Same source/TopK,runtime identity,independent buffers and per-chunk trace |
| Report | query-stage-joint | Original stages and timer plus choices/actual depth metadata | Reject missing/changed/out-of-space configs;private label,no formal production claim |

Only the candidate space changes. Original stage2 choices and ordering form the
first half; stage1 choices form the second. All non-stage parameters are paired
exactly. Selection uses the existing rows/query length/logits width/requests,
page/H/D/dtype and stride keys,not workload names or a new backend threshold.
Both candidate depths reuse the same kernel/math/host. Autotune still measures
score execution; whole-pipeline acceptance remains a separate requirement.

The new62-node iteration profile is40 canonical plus22 new candidate nodes.
Forced boundary/mode helpers explicitly launch both depths; every-config nodes
cover20/34 choices. Full files remain available with --accuracy-profile full.
This is a private tuning experiment,not a production integration. Broader tuning
can still choose a noisy winner; record actual per-chunk selections and retain
all raw pairs,especially the prior short-batch context anomaly. No guaranteed
non-regression or performance gain is assumed.

optimization.md and deep_opt.md are absent in this checkout; this board records
the deep-optimization hypothesis based on audited prior measurements. No new
profiler/timer/guard or validation threshold is introduced. No export/registration
change is needed for this private experiment.

## Implementation and measurement scope

The factory reuses the exact combined JIT and host globals with a new
independent rectangular entry/LibTuner,plus the original independently cloned
compact entry. It rejects any change outside the exact34 paired choices,
records the full space and actual compiled selections,and checks metadata
stability after timing. Compiled configs and rectangular traces must belong
to the declared choice set; unknown depth/geometry is rejected. Existing native,
canonical and combined previous-baseline stages retain their validation and
AB/BA timer. New candidate/report names are query-stage-joint and
flagtree134-stage-joint;the TSV latency column explicitly names the private trial.

Cache scope was checked in the unchanged LibTuner source: winner keys include
the complete config-space hash,so the34-choice tuner cannot directly reuse the
17-choice final winner. Its existing per-config benchmark cache is keyed by JIT
and runtime tuning key,and may be shared for identical configs within a worker.
This behavior is retained,not relabeled as fully independent tuning caches.
The original child_environment assigns separate fresh cache directories per
worker/report,preventing cross-report winner/timing reuse. Outer entries/device
caches and captured output/logits buffers remain separate. A superset choice
space does not guarantee better independently measured timings under noise.

Run the child-shell entry under the existing QSA vLLM0.31 activation and
QSA_STOCK_PYTHON:

```bash
bash tools/run_qsa_prefill_stage_joint_h800.sh --flagtree /opt/flagtree
```

The unchanged verified FlagTree/PTXAS13.4/CUDA13.4 compatibility runtime is
selected explicitly,without installing a compiler into the virtualenv. The
existing runner uploads complete or partial JSON/TXT/TSV reports automatically.
Both compiler and source bindings,actual per-chunk stage/query/tile selection,
compiled resources,raw timing pairs and TopK data remain recorded. No additional
self-control cycles or benchmark shapes are added; the broader candidate space
can increase compilation/autotune work outside timed regions.

Decision remains pending H800: compare all30 native pipeline ratios and the
same-run combined baseline,especially the three stage1 regressions tail32k,
unequal2 and chunk_tail64. Retain small/context-sensitive short-batch measurements
without replacement. Keep/revert requires audited correctness and whole-pipeline
data; score-only gains cannot satisfy acceptance. Production exports/registration
remain canonical and full integration validation is deferred.

Local validation:109 related CPU regression checks passed,followed by a final
7-check focused run including the added dtype/depth-pruning gate (110 unique
checks in total). New checks cover exact paired YAML choices,unchanged legacy
groups,shared JIT/host/TopK and separate entries,accepted stage1/stage2 compiled
selections,out-of-space rejection,metadata/acceptance,coverage and CLI/source
routing. Python/shell syntax,diff checks and changed-file pre-commit passed.
Byte comparisons confirm production,combined/stage1 factories and kernels,
workloads,reference,timer,guard and LibTuner source are unchanged; core benchmark
validation/environment ASTs are unchanged. No local Torch/Triton/CUDA is
available. Status: private deep-optimization trial,pending H800 JIT,numerics,
actual selections/resources and30-case whole-pipeline results. No production
integration or performance improvement is claimed at delivery.

## Startup failure: report b8a8d987 (2026-10-22 04:11:15 UTC)

Report commit b8a8d98784aa3fc86202380901967da5186289dc contains four
hash/size-verified files. It failed in with_cuda_compat134.py immediately after
downloading the pinned CUDA compatibility deb: SHA256 mismatch. No preflight,
correctness or benchmark worker ran. This is not a kernel compilation or
numerical failure. The old log did not retain observed hash/size/HTTP metadata,
so truncation,error response and upstream content change cannot be distinguished.
No new tuning or performance conclusion follows from this report.

The repair scope is only compatibility-package preparation,its CPU tests,this
incident record and source archiving of the tests. The fixed package URL/SHA256,
extraction/fresh-process library loading,all GPU kernels/configs/TopK,timers,
guards,correctness profiles and30-case definitions remain frozen. Prefer a
previous temporary package only after exact SHA256 verification,copy into a new
isolated prefix and verify again before extraction. If no verified copy exists,
retry download up to three times and record byte size,actual/expected hashes,
response metadata and errors in compat134-package.json. A mismatched package
must never reach dpkg-deb or execve. No system/virtualenv install or integrity
check bypass is permitted. Actual startup and GPU testing remain pending.

Repair validation:33 compat/runtime/runner/PTXAS CPU regression checks and a
final14-check compat + joint-stage focused run passed. New checks reject bad
cache copies,verify copied bytes again,exercise retry after invalid content,
retain partial-download diagnostics,record response metadata and prove that
unverified bytes never reach extraction/exec. Original package URL/SHA256 are
unchanged. Source comparisons confirm joint choices,kernels,TopK,profiles,
benchmark settings and validation remain frozen. The preparation record now
also binds the bootstrap source/hash when failure occurs before source archival.
No real GPU/driver execution was available locally. The failed report contains
no reusable accuracy/performance result; all62 checks and30 timings still await
successful startup with this repaired bootstrap.

## Startup handoff fix: report eaa9704e (2026-10-22 04:25:17 UTC)

Six archived files were verified by Git blob hash/size. Package reuse worked:
103922442 bytes,exact pinned SHA256 bfacf8a0c2bc5cac76880d443f5bd542b209166b14f90dbe2384f962b379d00a,
zero download attempts,and expected libcuda615.71.09 extracted in a fresh prefix.
The subsequent engine parser rejected the newly generated compat134-package.json
because the fresh-directory allowlist had not been updated. This was a bootstrap
integration omission in the previous repair,not a GPU/compiler/numerical result.
No preflight,correctness or performance data was produced.

Add only compat134-package.json to the FlagTree startup-artifact allowlist.
Do not allow existing stage-summary/case reports or remove the fresh-directory
guard. A new CPU integration test runs actual compatibility preparation with
mocked download/extraction,then actual FlagTree prepare/exec argument routing,
then the actual benchmark parser up to a mocked GPU parent. It must accept all
real startup artifacts and the frozen30 selection,while rejecting existing
summary/case files without overwriting their contents. GPU kernels,configs,
TopK,metric,profiles,timers and runtime identity checks remain unchanged.

Handoff repair validation:32 related CPU checks passed,including the new real
bootstrap-to-parser integration path and preservation/rejection of existing
summary/case files. Changed-file pre-commit and diff checks passed. This verifies
startup orchestration without CUDA;actual driver/JIT/numerics/performance remain
pending. The integration test source is now archived with future reports.

## H800 result: e861c93c (2026-10-22 05:43:31 UTC)

Report branch `codex/qsa-h800-results-20261022-054331`, commit
`e861c93c4a916bf309e37768430106b6bc5d5873`, tested clean source
`9e45b77dd89e61517559d9ce46468268fcf57a26`. Exit code0. Audit verified76
archived files,112 source snapshots,241 stage rows,880 timing samples and660
sampled guard observations. All30 cases passed measurement validation; raw
paired rounds,configuration membership,source bindings,TopK fingerprints and
runtime identity matched. Recomputed tables/acceptance match within floating
point roundoff. Both scoring implementations used the verified FlagTree/runtime
profile; native TopK remains its CUDA extension and candidate persistent_topk
is unchanged. Sampled temperature32–54C,SM clock1965–1980MHz; no foreign PID or
guard rejection was observed.

Iteration correctness:62 passed (40 canonical +22 private candidate),311.03s.
This is not the full correctness suite or formal production-entry acceptance.
Benchmark telemetry spans1794.61s; accuracy plus that span is about35m06s,
excluding setup and gaps outside telemetry. Two rounds and all timer/guard
requirements remain unchanged.

All means below are equal-weight arithmetic means of per-case speedups;
whole pipeline was timed directly,not reconstructed by summing stage timings.

| Group | Count | Native/score | Native/TopK | Native/pipeline | Combined/score | Combined/pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.210177 | 0.899041 | 1.030492 | 1.085889 | 1.040380 |
| Observed BF16 | 8 | 2.723415 | 0.908047 | 1.649331 | 1.009275 | 1.004600 |
| FP8 | 10 | 1.119405 | 0.923753 | 0.993914 | 1.035020 | 1.013313 |
| All | 30 | 1.583450 | 0.909680 | 1.183323 | 1.048502 | 1.021816 |

Native/pipeline mean1.183323 exceeds1.1,but11 cases remain below1 and five
below the proposed0.98 floor; target_met=false. Native/score minimum0.991069
has no case below0.98. Same-run canonical-production comparisons average
1.084705 for score and1.035627 for pipeline. Do not compare means from separate
runs as a precise incremental improvement.

Joint tuning selected stage1 for25 rectangular cases,stage2 for three rectangular
cases,with the two compact cases unchanged at stage2. Across actual chunks:
47 rectangular stage1,4 rectangular stage2,2 compact stage2. The three previously
regressing BF16 cases now select stage2 via the existing tuning key/pruner,
without a fixture-name gate:

| Case | Score vs combined | Pipeline vs combined | Paired pipeline rounds |
| --- | ---: | ---: | --- |
| observed_tail32k_bfloat16 | 0.999354 | 1.003795 | 1.007241, 1.000361 |
| observed_unequal2_bfloat16 | 0.995340 | 0.999900 | 1.000287, 0.999514 |
| observed_chunk_tail64_bfloat16 | 1.000210 | 1.000619 | 1.002227, 0.999013 |

Compared with the same-run combined baseline,25 cases improve in both rounds
for score and25 for pipeline. No pipeline comparison is below0.98. Retain the
small repeatable short96 pipeline loss:0.987682 vs combined (paired0.988385,
0.986961),and0.976874 vs canonical production. Its score vs combined improves
1.007859,so this is not explained by score-only results. TopK code is unchanged;
no cause is established. Short_batch3 now has pipeline/combined1.015479 in both
positive rounds and no previous31us cluster; the prior anomaly's cause remains
unresolved.

Remaining native pipeline floor failures:

| Case | Pipeline speedup |
| --- | ---: |
| perf_n1_q16384_p102400_bfloat16 | 0.979147 |
| fp8_tail64_float8_e4m3fn | 0.974793 |
| fp8_tail65_float8_e4m3fn | 0.975339 |
| fp8_two_skewed_float8_e4m3fn | 0.968851 |
| fp8_long100k_float8_e4m3fn | 0.898815 |

The worst case FP8 long100k illustrates why the pipeline goal remains open:

| Stage | Native median us | Candidate median us | Speedup |
| --- | ---: | ---: | ---: |
| score | 1341.016 | 1265.328 | 1.059817 |
| topk | 1154.376 | 1471.848 | 0.784304 |
| pipeline | 2492.320 | 2772.896 | 0.898815 |

Do not add standalone stage medians to predict pipeline latency. TopK remains
frozen; its measurements remain part of every report.

Per-case speedups (native/candidate for the first three columns; original
combined/candidate for the last two):

| Case | Score | TopK | Pipeline | Combined/score | Combined/pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 1.305195 | 0.961449 | 1.074314 | 1.066695 | 1.025702 |
| perf_n1_q8192_p51200_bfloat16 | 1.182348 | 0.910777 | 1.036357 | 1.105949 | 1.058953 |
| perf_n1_q8192_p102400_bfloat16 | 1.198700 | 0.805704 | 0.994330 | 1.085115 | 1.040360 |
| perf_n1_q16384_p8192_bfloat16 | 1.179804 | 0.996577 | 1.054730 | 1.073192 | 1.021220 |
| perf_n1_q16384_p51200_bfloat16 | 1.220493 | 0.889817 | 1.037759 | 1.074327 | 1.040311 |
| perf_n1_q16384_p102400_bfloat16 | 1.219914 | 0.786572 | 0.979147 | 1.136114 | 1.062498 |
| perf_n2_q8192_p8192_bfloat16 | 1.188723 | 0.963212 | 1.046541 | 1.072309 | 1.024383 |
| perf_n2_q8192_p51200_bfloat16 | 1.219748 | 0.911673 | 1.038278 | 1.083534 | 1.042577 |
| perf_n2_q8192_p102400_bfloat16 | 1.177453 | 0.813104 | 0.991613 | 1.098706 | 1.054384 |
| perf_n4_q4096_p8192_bfloat16 | 1.234709 | 1.011101 | 1.085931 | 1.091976 | 1.034539 |
| perf_n4_q4096_p51200_bfloat16 | 1.228957 | 0.924619 | 1.044224 | 1.081837 | 1.036418 |
| perf_n4_q4096_p102400_bfloat16 | 1.166086 | 0.813884 | 0.982686 | 1.060910 | 1.043218 |
| observed_short96_bfloat16 | 6.273879 | 0.730606 | 1.985365 | 1.007859 | 0.987682 |
| observed_tail16k_bfloat16 | 3.422723 | 0.912220 | 2.263390 | 1.013306 | 1.005504 |
| observed_tail32k_bfloat16 | 2.813066 | 0.830329 | 1.929812 | 0.999354 | 1.003795 |
| observed_short_batch3_bfloat16 | 4.431267 | 1.022274 | 2.825018 | 1.028878 | 1.015479 |
| observed_ragged54_bfloat16 | 1.056494 | 1.123242 | 1.091571 | 1.001298 | 1.005758 |
| observed_unequal2_bfloat16 | 1.234092 | 0.825104 | 0.984873 | 0.995340 | 0.999900 |
| observed_chunk_boundary_bfloat16 | 1.182837 | 0.972781 | 1.042760 | 1.027958 | 1.018064 |
| observed_chunk_tail64_bfloat16 | 1.372961 | 0.847819 | 1.071858 | 1.000210 | 1.000619 |
| fp8_first_chunk_float8_e4m3fn | 1.061870 | 1.018831 | 1.029669 | 1.090949 | 1.020685 |
| fp8_budget_exact_float8_e4m3fn | 1.000416 | 0.976414 | 0.984720 | 1.033665 | 1.013524 |
| fp8_dominant_long_float8_e4m3fn | 1.085851 | 0.961699 | 1.012610 | 1.033042 | 1.012124 |
| fp8_tail64_float8_e4m3fn | 1.253348 | 0.856127 | 0.974793 | 1.018875 | 1.007522 |
| fp8_tail65_float8_e4m3fn | 1.230867 | 0.849701 | 0.975339 | 1.019879 | 1.008986 |
| fp8_two_balanced_float8_e4m3fn | 1.118258 | 0.922271 | 0.995818 | 1.030868 | 1.013830 |
| fp8_two_skewed_float8_e4m3fn | 1.209974 | 0.849653 | 0.968851 | 1.014662 | 1.005116 |
| fp8_short_tail_float8_e4m3fn | 1.182580 | 0.896894 | 1.024323 | 1.075184 | 1.037768 |
| fp8_compact_float8_e4m3fn | 0.991069 | 1.121635 | 1.074205 | 1.004173 | 1.000154 |
| fp8_long100k_float8_e4m3fn | 1.059817 | 0.784304 | 0.898815 | 1.028907 | 1.013418 |

Decision: retain stage-joint as the current private tuning direction. It recovers
the three forced-stage1 regressions and improves the same-run combined pipeline
mean by about2.18%,but does not yet satisfy all-case acceptance. Production
exports remain canonical. Prioritize score-path work on the five remaining floor
failures,especially FP8 long100k,while retaining TopK measurements and the short96
residual. Full candidate correctness and formal production-entry validation are
required before integration. This result-recording change modifies no kernel,
configuration,benchmark shape,timer or guard.
