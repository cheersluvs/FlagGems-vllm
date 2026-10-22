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
