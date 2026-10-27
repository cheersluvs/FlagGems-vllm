<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# FP8 three-buffer producer-first runtime probe

Pre-code boards,2026-10-27. Report99bf2f5c proves exact original replay/control
and candidate copy motion for two independently preserved FP8 cases. Candidate
moves4/4 K fragments before first MMA,retains8 MMAs/two waits and zero spills;
registers158->162,shared56->64KiB,K ring2->3. No prior runtime result exists.

|Project truth|Frozen decision|
|---|---|
|Editable|New runtime adapter,CPU tests,upload runner and this board only|
|Read-only|Production0d7e80e2,YAML,TopK,fixtures/native/reference,timer/guards,compile/profiling helpers and returned artifacts|
|One change|Execute exact verified compiler-generated three-slot FP8 IR via original JIT per-launch ir_override;no new schedule/slot/config change|
|Active set|tail113 and long100k;only FP8 H4D128/TR64/BN64/KT16/P2/W4/stages2 main chunks change.113-row TR8/P1 tail stays original|
|Build|Original production JIT warmup for executable identity,then per-launch final-TTGIR override;raw candidate PTX/CUBIN/resources must equal compile probe before timing|
|Validation|Three FP8 boundary scenarios on both saved case layouts,all35 offset rows,left32/right3 split,guards/masked lanes,0/1/2/3/15/16-tile and page edges;real score/selection/input/source checks before/after timed stages|
|Benchmark|Original kernel timer,2 alternating rounds,20ms warmup/20ms rep;score excludingTopK,unchanged TopK,actual pipeline and four identical-callable A/A controls;ratios current production/candidate|
|Lifetime|Read-only FP8 ring/counter/token protocol review plus abstract three-slot one-warp-group model for0..16 iterations;no buffer shrink or BF16 allocation transform|
|Autotune|Fixed-capture diagnostic exemption;original configs/host/chunks/workspaces unchanged|
|Profiler|No new NCU collection;actual IR/PTX/cubin/resources exported outside timing|
|Safety|Existing exclusive idle/PID and thermal guards80C/120s,1800s worker timeout;safe child bash and automatic partial/success upload|

|Contract|Preserved|
|---|---|
|Math/API|FP8 E4M3FN Q/K,FP32 full-D dot/ReLU/head sum/logits,INT32 selected indices,original pointer ABI and masks;production exported entry untouched|
|Lookahead|Compiler-generated ring3,copy next tile to write slot,wait current committed token before two dot consumers,post-WGMMA CTA barriers,epilogue drain;review before execution|
|Compiler|Same FlagTree/PTXAS13.4/CUDA13.4 compat as captured;original JIT signature/options used,not reconstructed ABI or global IR override|
|Profile reuse|Only after proving both captures' control/candidate binaries,IR,options and protocol identical,collapse equivalent matching profiles to one with explicit covered_cases;otherwise fail before launch|
|Inactive|TR8/P1 tail and all other routes use original JIT/config;exact same-run fallback PTX/CUBIN equality required|
|TopK|Same persistent_topk callable;isolated TopK uses common frozen logits and original buffers|
|Scope|Small tested numerical scope and two-case incremental ratios,not full10/native-vLLM acceptance or production integration|

|Path|Gate|
|---|---|
|Load artifacts|Require both positive compile decisions,six successful jobs,all compile-source/input/options/binary identities,reversible cluster-only rewrite,FP8 sync/resource gates and original captured runtime compiler|
|Prepare|Deduplicate only proven identical artifacts;copy verified final IR into new report once;raw candidate PTX checked during untimed warmup before first actual candidate launch|
|Route|FP8 main only;never attach override to original rectangular BF16 or small FP8 tail. Freeze rejects any unexpected profile/config/binary/resource change|
|Boundaries|Keep same saved cache/table capacities/strides;small Q view and device-only test metadata,nonzero base97 and offset5;real dimensions/head/query halves remain observable|
|Time|Reuse original coordinator/worker/numeric/timer/GPU machinery;timed candidate path performs one JIT launch,no baseline execution/warmup/file read|
|Next|Benefit beyond A/A and correct results -> broader FP8 validation or maintainable implementation work. Reject regression;do not automatically combine two-buffer optimization|

Local validation:36 applicable CPU checks pass,including seven new routing,
raw-PTX-before-launch rejection,lifetime/negative controls,actual main resource/
cubin checks,exact tail fallback,profile dedup/IR immutability and real CLI
worker/scoped-callback tests. Both real99bf2f5c final FP8 IR/PTX pairs pass the
read-only ring/token/phase review and17 abstract0..16-iteration lifetime checks.
Their independently verified controls/candidates/final IR/options/reviews match,
so a single executable profile with explicit two-case coverage is prepared;
missing or changed prepared IR fails rather than silently recreating it.
CLI help exposes only the two FP8 cases;bash syntax and changed-file hooks pass.
No local GPU compilation/execution or speedup claim;card numerical and runtime
binary/resource/fallback gates remain mandatory.
