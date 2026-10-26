<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Current FP8 production profiling

Pre-code boards,2026-10-26. FP8-only optimization scope. Collect current
production first_chunk,tail113,long100k;no BF16 kernels or private candidates.
Reuse the verified current-production hardware collector,not the old three-side
stage-joint profiler. Production source0d7e80e2 already includes vector stores.

|Project truth|Frozen decision|
|---|---|
|Editable|New adapter,three-case manifest,CPU checks,runner and this document|
|Read-only|Production/YAML,existing collectors/parsers/guards,timers,native source,workloads and historical reports|
|Active set|Three existing FP8 cases:first_chunk1chunk,tail113main+113-row tail,long100k4chunks;seven score launches|
|Build|Exact canonical production JIT with captured production configurations;no compiler or kernel modification|
|Validation|Original sampled FP32-reference checks before/after,source/input/compiler identities,logits fingerprints,exact independent-preparation versus NCU PTX/CUBIN/resource equality,per-chunk NVTX/CSV matching|
|Benchmark|No retiming. Preserve historical78c3ca4f native/production score,TopK,pipeline timing as explicitly labelled context;binary mismatch disqualifies transferring those timings to the fresh binary|
|Profiler|Original metric-only NCU kernel replay,cache-control none,clock-control none;only warmed production scoring with real output stores inside capture|
|Counters|Traffic/shared wavefronts and bank conflicts,long/short scoreboard,barrier/MIO stalls,register/shared/occupancy,tensor/issue/eligible;retain missing optional values and percentage anomalies|
|Autotune|Fixed-capture diagnostic exemption;no new choices,retuning or dispatch changes|
|Timeout/thermal|1800s per worker,80C initial/120s cooldown,original exclusive GPU/PID guards;no bypass or GPU job termination|
|Tools|Reuse installed/portable NCU discovery;--prepare-ncu allows existing pinned isolated helper;no system/compiler install|

|Contract|Preserved|
|---|---|
|Inputs|Unchanged FP8 E4M3FN H4/D128/page196 fixtures,cache/table layout,per-request lengths/visibility/chunks and512MiB budget|
|Math/output|Same full-D FP32 dot/ReLU/head-sum and FP32 stores;INT32 indices and persistent_topk unchanged|
|Snapshot scope|Three configurations from validated production report78c3ca4f,not new autotune results. first_chunkTR32/P1/stages1;tail113mainTR64/P2/stages2,tailTR8/P1/stages2;long100kTR64/P2/stages2|
|Semantics|No approximation,new dtype/layout/API,quantization scale,host compute fallback or production import from tools|

|Path|Action|
|---|---|
|Manifest|Pin report/source/compiler/inputs/configs and old binaries;validate every geometry against current fp8-prefill definitions|
|Prepare|Reuse original collector's untimed current preparation and numerical guards;save actual artifacts before identity checks|
|Capture|Original collector code unchanged,scoped adapter replaces only fixture/source identity and child entry path;seven actual score launches,TopK outside collection|
|Resume|Reuse complete same-adapter captures only after all original source/compiler/input/artifact proofs;no mixed stale source evidence|
|Report|Original terminal traffic/stall tables and labelled historical three-stage times;JSON/CSV/PTX/TTGIR and failure logs auto-upload|
|Next gate|Inspect current compiled async/layout/wait/resource evidence and counters;choose one FP8 hypothesis,compile-first probe,short A/B beyond A/A noise,then all10 regression|

Local validation:37 applicable CPU checks pass,including four adapter/manifest/
child-dispatch/offline-display checks and the unchanged collector/traffic/profile
regressions. Real downloaded78c3ca4f JSON confirms exact three-case fixture/input/
compiler identities and every selected constant/PTX/CUBIN record. The pinned
manifest independently checks current FP8 geometry;seven chunks are retained,
including113-row tail. CLI help,runner bash syntax and changed-file repository
hooks pass. No local GPU execution,NCU capture,new timing or bottleneck claim.
