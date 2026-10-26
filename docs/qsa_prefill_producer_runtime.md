<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Producer-first runtime validation and short A/B

## Project board (2026-10-26, before code)

|Field|Frozen choice|
|---|---|
|Evidence/target|93eb254a validated two producer-first binaries: R32 registers63→68/shared72→88KiB;R64 registers207→217/shared104→120KiB;K slots2→3,zero spills;runtime/numerics/timing pending|
|Editable|This board, private runtime driver, CPU tests, runner|
|Read-only|Production0d7e80e2,configs,references,all prior helpers/probes,timers,tolerances,GPU guards and captured artifact report|
|One change|Execute the already validated producer-first final IR through the original production JIT's per-launch ir_override option; no new scheduling change or ABI reconstruction|
|Active set|Only perf_n1_q8192_p102400_bfloat16 and perf_n4_q4096_p102400_bfloat16; original captured chunk geometry/configs/inputs|
|Build|CPU tests,CLI help,bash syntax,changed-file pre-commit;H800 actual JIT warmup validates raw candidate PTX before any candidate launch|
|Validation|Three boundary scenarios on both BF16 configurations: all-zero,single-tile,mixed page/tile/width edges;25 logical rows with offset5/count17,nonzero query base,empty/unequal request where N4;all17 selected rows against existing FP32 reference,canaries and inactive outputs|
|Benchmark|Same --mode kernel timer,2 alternating rounds,20ms warmup/20ms rep;score,unchanged TopK,sequential pipeline;four identical-callable A/A controls|
|Acceptance|Incremental production/candidate ratios;NV score excludes TopK;report all stages,do not claim native-vLLM speedup or30-case acceptance|
|Safety|Same FlagTree/PTXAS13.4/CUDA13.4 compat identity,original exclusive GPU/PID/thermal checks,80C initial ceiling/120s cooldown;case timeout1800s|
|Autotune|Private fixed-config exemption;no production config or search-space change|
|Delivery|Terminal tables,source/artifact identities,actual compiled resources and automatic upload;fail before timing on mismatch or numerical failure|

## Contract board

|Field|Preserved/limited|
|---|---|
|Math/API|Original public and private host functions and JIT signature;candidate IR changes only previously validated cluster ordering;no manual CUBIN parameter binding|
|Dtype/layout|BF16,H4,D128,original cache/page-table/query strides and metadata types for these two captures;this private artifact mapping explicitly rejects other executable signatures|
|Boundary coverage|Reuse captured cache/table shapes and strides,change only dynamic query bounds and device metadata values;25-row no-copy Q view preserves query stride;reference tolerances unchanged|
|Unsupported/private scope|No new FP8 or arbitrary-stride candidate coverage;all production paths remain unchanged;private binary-profile miss is an error,not a fallback|
|Torch use|Test input creation,views,reference checks and canaries only;no production Torch compute fallback|
|Performance limit|Compilation,profile checks and references excluded from timing;normal timed path launches exactly one candidate score kernel per chunk|

## Paths board

|Path|Action|
|---|---|
|Artifacts|Validate captured source hashes,PTX/IR/resources and cluster-only mutation;copy candidate IR to report with .ttgir suffix;verify original/copy hashes again after execution|
|Prepare JIT|Use original JIT warmup without execution to identify the exact executable signature;baseline comparison removes only comments,file/loc directives and balanced DWARF debug sections,keeping all program declarations,instructions,registers and labels|
|Candidate JIT|Warmup same original JIT with per-launch ir_override;require exact raw PTX hash of verified candidate before launch;retain normal signature binding,grid,stream and launch metadata|
|Steady state|Prepared profile per captured chunk;no warmup,artifact reads or extra baseline launches in the timed callback;freeze preparation before timing|
|Boundary|Both candidate profiles must execute in each scenario;check all selected rows,output guards,masked outputs and unchanged inputs;do not prefill output with baseline results|
|Main correctness|Original before/after score and TopK/pipeline selection checks,source/input/final compiled/common-logit identities unchanged|
|Report|Production/candidate score,TopK,pipeline plus A/A;actual68/217registers,88/120KiB,zero spills and archived raw candidate PTX must reproduce|
|Integrate|None until numerical validation and useful measured gain;R64 shared-memory residency cost remains a concern|

## Local validation

Eight new CPU tests plus sixteen related compiler/replay tests pass. Tests cover
strict executable comparison with debug-only differences,unknown profile/raw
candidate mismatch before any launch,no hidden baseline execution,no warmup in
frozen callbacks,scope/child routing,resource freeze and request/chunk boundary
coverage. Actual93eb254a artifacts load as two distinct executable profiles;
source/PTX/CUBIN hashes and copied IR bytes verify. Deleting a prepared IR file
is detected rather than silently repaired during final identity checks.

The pinned FlagTree JIT source confirms warmup calls run(warmup=True),and the
runtime launch is guarded by not warmup. Backend kwargs parsing accepts the
per-launch ir_override option. Candidate IR copies use the required .ttgir
suffix. Warmup PTX/IR are exported before numerical validation,including on a
profile mismatch;the timed branch does not export or re-read artifacts.

CLI help,bash syntax and all applicable pre-commit checks pass. This Mac has
not executed the candidate on GPU;runtime correctness and performance remain
pending the card run.
