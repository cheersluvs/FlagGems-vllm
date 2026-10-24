<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Current production score-only hardware diagnosis

Pre-code boards,2026-10-24. Historical hardware counters predate production
head grouping and FP8 vector stores. Query tile doubling91236117 regressed,
with R128/grouped255 registers and spill_count106. Refresh current production
counters before creating another optimization kernel. No new candidate here.

|Project truth|Frozen decision|
|---|---|
|Editable|New profiling driver/runner/CPU checks and this document only|
|Read-only|Production0d7e80e2, NVIDIA YAML, native source, TopK, captured78c3ca4f inputs/configs and compiler identity|
|Build|Compile exact production JIT at captured configs; require original PTX/cubin hashes. No algorithm/config/source changes|
|Active set|n1/q8192/p102400 BF16 R32/W8, n4/q4096/p102400 BF16 R64/grouped/W4, FP8 long100k R64/QUERY_PARTS2/W4;2+4+4 score launches|
|Validation|Public call observes chunk geometry outside capture; independent sampled score checks before/after, input/logits fingerprints and compiled records unchanged. Previous full-entry correctness is historical, not rerun|
|Benchmark|None. Reprint archived native/production score,TopK,pipeline timing with source label; no new latency or speedup claim|
|Profiler|NCU metric-only kernel replay/cache-control none/clock-control none/profile-from-start off; one warmed score sequence per case, only production side. All chunks retained with NVTX labels|
|Counters|Global LD/ST transactions, DRAM bytes, shared wavefronts/bank conflicts, warp stalls, tensor/issue/eligible/occupancy; query available suffixes first, retain missing optional fields and raw CSV. Traffic required; missing diagnostic counters make diagnosis partial|
|Aggregation|Per-chunk raw counters and derived ratios; no averaging across geometries or physical clamping of anomalous percentages. L1 sector amplification is not HBM traffic or a speedup bound|
|Autotune|Fixed-capture exemption; exact captured production configs, no retuning|
|Timeout/thermal|1800s per NCU worker,80C admission ceiling/120s cooldown; exclusive idle GPU guards before/after capture|
|Tools|Reuse existing ncu incl bounded portable /tmp discovery. --prepare-ncu permits pinned isolated extraction/download via existing helper, no system/compiler/driver install|

|Contract|Preserved|
|---|---|
|API/reference|qsa_select_paged_prefill current production; pinned vLLM0.31 source/runtime identity retained, no public torch equivalent|
|Math/layout|Unchanged BF16/FP8 H4D128, full-D FP32 dots, per-head ReLU/sum, arbitrary captured page/cache strides and score masks|
|Torch use|References/fingerprints only in diagnostic tools; no production compute fallback|
|Output/autograd|Actual FP32 logits stores retained; inference forward, no algorithm or public output change|
|Scope|Sampled actual-case correctness, hardware diagnosis only; not full suite/native acceptance or cold-cache serving simulation|

|Path|Treatment|
|---|---|
|Prepare|Verify pinned baseline report/code/input/compiler/validated timings; preserve source snapshots and historical timing rows|
|Capture|Load canonical production module, clone only function globals for FixedEntry captured configs, verify original binaries, warm and validate outside profiler region|
|Profile region|One score sequence, separate NVTX label per chunk; no TopK, allocation, validation, autotuning or altered stores|
|Postprocess|Validate exact chunk IDs/labels, retain all queried counters, show traffic/resource/stall tables and export JSON; missing/invalid required counters fail|
|Failure/retry|Upload partial report on failure; successful case captures can be reused with --resume-report after exact source/input/compiler/binary checks, no new benchmark|
|Next decision|Use actual bottleneck evidence to choose dot→head reduction→store layout probe or live-state probe; no production integration in this step|

Local validation:32 CPU checks passed (10 new production-profile checks plus
existing memory/profiling checks). Tests execute actual capture sequencing and
failure cleanup, frozen original-config/binary proofs, raw CSV chunk matching,
traffic ratios, resource consistency, required/optional/N/A handling, unmodified
out-of-range percentages, corrupted artifact rejection, resume revalidation,
three-child coordinator launch construction and offline CLI without CUDA setup.
Historical TopK rows remain explicitly labelled; no test fabricates measured
performance evidence. Shell syntax/CLI help and changed-file style checks pass.
No local GPU or NCU collection is available; hardware capture remains pending.

## Recovery board: ec467779

The original collector failed before cudaProfilerStart: the compiled scorer
hash differed from the historical78c3 manifest. Inputs/source/compiler/config
checks passed and sampled scoring validation completed, but actual compiled
records/artifacts were not persisted before rejection. There are no completed
hardware counters. The precise PTX/cubin difference is unknown; do not label it
as harmless debug data or profiler instrumentation without comparing artifacts.

|Recovery truth|Decision|
|---|---|
|Editable|This document, collector and its CPU checks only|
|Read-only|All production code/configs, numeric tolerances and historical records|
|New preparation|Run an unprofiled, untimed preparation worker per case; record actual configs, hashes, artifacts and numerical/GPU checks before collection|
|Identity gate|NCU capture must exactly match that successful current preparation, including PTX/cubin/resources/input/source/compiler. Historical binary matches/differences are explicitly reported separately|
|Historical timings|Context only; never assert their binary is identical when its hash differs. No fresh speedup or benchmark claim|
|Failure evidence|Save actual PTX/TTGIR/resource records before any binary comparison, including a failed NCU/preparation comparison|
|Retries|Resume needs both the current preparation and capture proofs; stale collector-source records are rejected. This failed run has no reusable hardware capture|

Additional evidence: the successful227ce1c1 and91236117 R32 production PTX files
have identical text before trailing `.file 1`/DWARF sections, while those debug
sections encode different temporary source directories and their full PTX/cubin
hashes differ. This proves whole-file hashes vary across these two checkouts;
it does not prove the unexported ec467779 binary differs only in debug data.
The new preparation/capture identity check remains byte-for-byte, with no
normalization of executable code or debug sections for acceptance.

Recovery validation:33 related CPU checks pass. The added checks accept an
explicitly recorded historical mismatch only when current preparation and NCU
capture match exactly, reject a capture/preparation cubin change, and verify
three untimed preparation children precede their three NCU workers. Required
config/source/compiler/numeric/GPU/artifact gates remain. Actual GPU validation
of this recovery is pending; no profiler counters or speedup are claimed.
