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
