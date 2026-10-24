<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Production shared-memory and synchronization attribution

Pre-code boards,2026-10-24.64f15a08 supplies current-production global counters,
matched preparation/capture binaries and source/PTX/TTGIR. Whole-kernel barrier
or bank-conflict counts do not identify which conversion or wait dominates.
8f595cda rejects the query-major+vector-store candidate; return to production.

|Project truth|Frozen decision|
|---|---|
|Editable|New attribution collector/analyzer,CPU checks,runner and this board|
|Read-only|All production/candidate kernels,YAML,TopK,production-profile helper and its guards,fixtures,tolerances,timers|
|Static analysis|Map actual TTGIR local_alloc/convert_layout/reduce/async/MMA waits and PTX shared/barrier instructions to canonical source lines. Static sites are not executed counts or time|
|Dynamic collection|Same three production configurations and untimed independent preparation; NCU SourceCounters plus prior traffic metrics, first actual chunk only per case (three profiled kernels). Remaining chunks only warmed/validated; no TopK in capture|
|Binary proof|Reuse existing exact preparation/capture PTX/cubin/source/compiler/input/numeric/GPU gates. No across-directory historical binary equality assumption|
|Profiler|Kernel replay,cache-control none,clock-control none,profile-from-start off,launch-count1 without kill. Export local .ncu-rep, source/SASS CSV and text,raw counters and per-PC numeric JSON through NVIDIA ncu_report|
|Metric scope|PC warp stall samples,per-instruction shared wavefronts/N-way,barrier/MMA wait/load/store/source correlation when available. Keep unknown/unmapped/missing fields; no zero substitution|
|Benchmark|No fresh benchmark/A/A/full suite. Archived score/TopK/pipeline context still printed; profiler durations never speedups|
|Tool preparation|Existing pinned isolated NCU helper only. CLI+Python report interface preflight before GPU; no compiler/driver/system install|
|Autotune|Fixed captured production configs, no search or new candidate|
|Timeout/thermal|1800s per preparation/NCU child,80C admission ceiling/120s cooldown,exclusive PID guard|
|Recovery|Fresh output; --resume-report reuses only verified complete case capture plus its preparation/artifacts. --analyze-report re-exports existing local .ncu-rep without GPU/benchmark|

|Contract|Preserved|
|---|---|
|Math/API|Same qsa_select_paged_prefill forward/native pinnedvLLM0.31,BF16/FP8,H4/D128,strides/pages/visible bounds,FP32 full-D dot/per-head ReLU/sum and stores|
|Torch|References/fingerprints only in diagnostics, no production compute fallback|
|Attribution limits|Sampled instruction is a waiting consumer; its source line may be inlined/generated. A sample fraction is not runtime fraction or removable speedup. Do not sum derived ratios or translate static counts into time|
|Coverage|First full chunk of each configuration only, not tails/all30. Source-unmapped samples remain a separate bucket|

|Path|Action|
|---|---|
|Existing report static|Validate source/compiler/artifact hashes; generate source/IR/PTX maps locally, no GPU|
|Untimed preparation|Existing original production worker,actual inputs/configs and numeric checks; no profiler|
|PC collection|Existing worker,only first score chunk profiled,all old correctness and identity checks retained|
|Postprocess|Validate raw launch NVTX chunk0,compiled resources and all prepared/captured binaries; export SASS/source plus raw per-PC correlations; print top reason/location counts|
|Partial sampling|Preserve .ncu-rep and text/raw data; mark partial/exit2 if no positive PC samples. No fabricated attribution|
|Delivery|Upload JSON/TXT/CSV via existing safe roundtrip. Binary .ncu-rep stays on card for offline re-export; path/hash recorded|

Official interfaces used:
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html

## Verified static findings before card sampling

Actual first-chunk artifacts from 64f15a08 match the current production source
SHA256 0d7e80e240966565a7c78151c8344f17e0627e73920f5229aadcc0f0fb42f10e.
Counts below are static instruction sites, not dynamic events or cycles.

|Configuration|Output convert_layout location|Output shared stores/loads/barriers|Total CTA barrier sites|
|---|---|---|---|
|BF16 R32,W8|canonical line470,TTGIR271|16/8/1|3|
|BF16 R64,head pairs,W4|canonical line470,TTGIR287|8/8/3|8|
|FP8|canonical lines261/279,TTGIR313/336|16/8/3|8|

All these output conversions are inside the key loop. BF16 head reduction
produces a register distribution different from the final logits store layout;
TTGIR convert_layout lowers through shared stores, CTA barriers and shared
reloads. The grouped BF16 PTX uses scratch at offset98304 and above for this
path. This is distinct from Q staging before the key loop, K async-copy waits,
and WGMMA-result waits. In the grouped BF16 case two WGMMA waits map to head
reduction consumer lines453/457; they do not prove the reduction itself causes
those waits. Visible-max reduction contributes two setup barrier sites in the
R64 and FP8 captures. Do not attribute all whole-kernel barrier samples to the
output conversion.

Validation: 31 CPU tests (8 new plus existing production-profile/profiling
checks), targeted pre-commit, bash syntax and verified old-artifact static
mapping. GPU PC sampling remains pending; no new timing or speedup is claimed.
