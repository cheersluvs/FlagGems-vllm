# INT8 score profiling before optimization probes

## Frozen project truth

| Field | Choice before implementation |
| --- | --- |
| Task | Locate the score bottleneck behind initial mean overall speedup0.0180, benchmark report e8b1eea0 |
| Editable | New profiling script/runner and this document; backward-compatible profiler helper options |
| Read-only | INT8 score kernel, TopK, workload generator, benchmark and production files |
| Build/validation | Existing FlagTree JIT; local syntax/format/parser checks; CPU sampled scores before/after capture |
| Active set | int8_budget_exact and int8_long100k, original full logical Q/cache/input generation |
| Measurement | One full score-workload CPU+NPU timeline; three repetitions of one middle score chunk per counter group |
| Benchmark/aggregation | No speedup measurement; instrumentation overhead is excluded from benchmark conclusions |
| Timeout | 900 seconds per case, stream logs and retain partial artifacts |
| Profiler | Previously validated TorchNPU CPU/NPU profiler; try five separate metric groups |
| Autotune/fallback | No new kernels or tuning; no score/TopK fallback; optional counter absence recorded |

## Contract

| Field | Frozen requirement |
| --- | --- |
| Input/reference | Same synthetic INT8/FP16 inputs and CPU INT32-dot/FP32-score reference as the passed ten-shape suite |
| Scorer | Exact benchmark source hash, debug store disabled, H4 D128 N32/head16 tile, <=8 programs |
| Output | FP32 scores; finite for visible columns and -inf elsewhere; CPU rtol/atol1e-3 |
| Scope | Score only; no native or TopK kernels captured |
| Representative | Middle chunk at row8192/count128 for budget_exact, row8176/count112 for long100k |
| Resources | Bounded score scratch only; no full-logit benchmark staging and no raw-dot buffer |
| Unsupported | Other workloads/dtypes/layouts/gradients outside this capture; missing mandatory PipeUtilization or unusable trace returns failure |
| Torch use | Input generation/transfers and correctness checks outside profiling; measured workload is unchanged Triton scoring |

## Capture paths

| Path | Implementation | Validation | Output |
| --- | --- | --- | --- |
| Warmup | Full score invocation, then representative chunk | Check actual last output and representative rows | Correctness and compiler identity |
| Timeline | Full workload, default one iteration | Preserve raw trace and require recognized device tasks | Chrome trace gzip; device sum/union/envelope/gaps |
| PipeUtilization | Representative chunk, three repetitions | Require matched score rows and numeric pipeline metrics | Mandatory per-task hardware evidence |
| ArithmeticUtilization | Same chunk | Optional availability and numeric group fields | Arithmetic instruction evidence |
| Memory | Same chunk | Optional availability and numeric external-memory fields | External memory traffic/bandwidth evidence |
| MemoryUB | Same chunk | Optional availability and numeric UB fields | UB activity evidence |
| ResourceConflictRatio | Same chunk | Optional availability and numeric conflict/bank fields | Resource conflict evidence |
| Compiler | Representative compiled kernel | INT8->INT32 dot and one output store | TTIR and available textual ttadapter/bcmlir/source |

The helper's historical defaults remain three timeline iterations,
PipeUtilization and the BF16 kernel-name filter. New options are opt-in.
The INT8 script parses the raw CSV archive again and requires useful metric
columns; exported CSV presence alone does not establish counter availability.
Aggregates use only one preferred detailed table to avoid counting both
summary and per-task rows. Raw columns/units and selected table are retained.

Pipeline ratios may overlap and must not be added. Exported-row medians are
not whole-device utilization. Grid8, real heads4/padded16 and static IR token
counts are scheduling facts, not proof of the bottleneck. Gaps between
recognized tasks do not alone prove host starvation. Compiler and counter
evidence must be interpreted together before proposing a change.

PipeUtilization and a usable timeline are required for exit0. Optional groups
can be unavailable and remain explicit in JSON. Raw trace/CSV/IR artifacts
are uploaded even after a parsing or metric failure, so postprocessing can
be repaired without rerunning GPU collection. This does not assert device
isolation or stable benchmark performance.

Metric names follow the official [Ascend profiler documentation](https://github.com/Ascend/msmonitor/blob/master/docs/en/nputrace_instruct.md).
Run `bash tools/run_qsa_int8_score_profile_910b.sh --trace-iterations 1
--counter-iterations 3`. No algorithm or production code changes in this round.

## 2026-11-03 capture and offline reanalysis

Report branch: `codex/qsa-ascend910b-results-20261103-042215-769301`,
commit `8dda14206542006974f4c72394e316d5d4a70782`.
Both workers finished all correctness checks and all five metric groups.
The original exit code 1 was a parser false negative: the Chrome trace uses
`MIX_AIC` rather than the Triton function name for device tasks.

The corrected parser accepts generic tasks only in this isolated score-only
capture, with exact expected device-task and named host-launch counts. It does
not claim a per-task flow mapping. Raw reports and original exit codes are
preserved; offline replay validates 128/128 and 147/147 launches. Count mismatches
and mixed device task names remain rejected. No device rerun is required.

| Metric | budget_exact | long100k |
| --- | ---: | ---: |
| Full score device-task sum, ms (instrumented) | 929.785 | 3831.887 |
| Uncovered device envelope | 0.0389% | 0.0108% |
| AIC MAC ratio, median | 0.021 | 0.021 |
| AIC scalar ratio, median | 0.379 | 0.341 |
| AIV scalar ratio, median | 0.649 | 0.950 |
| AIV vector ratio, median | 0.244 | 0.292 |
| AIV MTE2 ratio, median | 0.431 | 0.469 |
| AIV vector bank conflict ratio, median | 0.022 | 0.020 |

Ratios are the exported counter fields, with overlapping pipelines and separate
metric passes. They are not additive stage-time shares. The budget case's AIV
scalar ratio ranges 0.608–0.662; long100k ranges 0.946–0.951 across three samples.
The exported `cube_utilization(%)` is about 39–40 while `Block Num` is 8 and the
reported AICore count is 20; it must not be substituted for `aic_mac_ratio`.
Neither low/zero exported memory bandwidth nor kernel-level activity alone proves
chip-wide bandwidth saturation or isolation.

Inference: device execution dominates the score-only envelope; tiny per-row
32-key tiles, repeated scalar indexing/metadata and per-tile processing are the
first hypotheses to test. The low MAC ratio does not identify a specific wait
instruction. First compare grid 8 versus 20 with N32 fixed, then N32 versus N128
with grid fixed. Keep Q/K data, scales, masks, TopK and timer fixed. Only after a
correctness-checked measured gain should query-row packing or production changes
be considered. This round changes no scoring or production implementation.
