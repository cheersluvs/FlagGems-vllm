# QSA prefill: evidence before optimization

Pre-coding board, 2026-10-22. Read Claude sessions 1b9d3d41 and14845937:
production-path attribution -> one-hypothesis probe -> representative A/A and
A/B -> full regression -> integration. Device-specific Ascend conclusions are
not H800 evidence. optimization.md/deep_opt.md are absent. This round changes
only diagnostic tools; no new optimization or production integration is claimed.

| Project truth | Frozen choice |
| --- | --- |
| Operator/reference | Existing QSA prefill API and SHA-pinned native vLLM7a877ae |
| Production | Canonical SHA256 f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c |
| Candidate | Retained query-stage-joint, unchanged combined scorer and34 choices; no unroll2 |
| Editable | Profiling coordinator/capture helpers, opt-in engine hooks, focused CPU checks, wrapper and this board |
| Read-only | All kernels, tuning policies/YAML, references,30-case definitions, timers, GPU guards and tolerances |
| First active set | BF16 n1/q16384/p102400, FP8 long100k, FP8 tail112, BF16 short96 and short_batch3 |
| Timings | Independent unprofiled kernel-mode score/TopK/sequential pipeline, two rounds,200ms/100ms; canonical comparisons retained |
| Profiling | Separate worker per case, actual warmed score launchers only, native then canonical then candidate, all real chunks once; Nsight Compute optional, bounded sections and timeout |
| Autotune | Existing original keys and choices; JIT/autotune outside profiler capture; compare actual per-chunk configs and binaries against unprofiled run |
| Validation | Existing benchmark preflight, pre/post score/TopK/pipeline numerical guards, runtime/source identities; standalone pytest explicitly skipped for this unchanged-code diagnostic |
| Acceptance | Diagnostic only; five cases cannot pass full30 performance/correctness acceptance |
| Torch use | Existing benchmark inputs/references only; production untouched, no compute fallback |

| Contract | Preserved |
| --- | --- |
| Semantics | Same BF16/FP8, H/D/layout, chunk/visible/page bounds, FP32 dot and per-head ReLU/sum, output/-1/ties and unsupported behavior |
| Host/TopK | Same public entry, validation, routing, workspace and persistent_topk |
| Empty/select-all | Unchanged, excluded from this scored diagnostic subset |
| Representative routes | Existing rectangular scorer; compact remains in full30 regression, not claimed profiled here |
| Training | Existing inference-only scope |

| Path | Diagnostic | Gate/risk |
| --- | --- | --- |
| Independent timing | Existing engine plus identical-function A/A controls on score and pipeline for each of native/canonical/candidate | Existing timer, source, numeric and foreign-PID guards retained |
| Hardware capture | Original score functions with output stores intact; NVTX labels and cudaProfilerStart/Stop delimit only warmed score invocations | Capture is not a speedup measurement; kernel replay/cache policy differs from graph benchmark |
| Missing profiler | Record executable/version/section availability and static resources | No dynamic utilization/cache/bandwidth inference; hardware stage remains incomplete |
| Probe selection | After returned counters, identify1-3 falsifiable hypotheses | No speculative modified kernel before evidence; keep stores/dataflow observable and check boundary shapes |
| Production integration | Only after representative benefit exceeds A/A variation and full30/full correctness pass | No cumulative gain inferred from different reports |

## First H800 collection

`tools/run_qsa_prefill_profiling_h800.sh` bootstraps the existing verified
FlagTree/ptxas13.4/CUDA13.4 compatibility profile in an isolated process. It
runs the five cases above with the original engine and adds six A/A controls
(native/canonical/candidate times score/pipeline). TopK is still reported
against the common frozen native logits. Standalone pytest is explicitly
skipped because neither implementation changes; pre/post numerical checks
and exclusive-GPU telemetry remain. This is not a production release gate.

A second process per case runs under Nsight Compute. Only the three real
score launchers execute between cudaProfilerStart/Stop; all JIT/autotuning,
validation, TopK and timing setup precede collection. Each side runs all real
chunks once, with NVTX labels. Actual chunk configs, cubin hashes, input
fingerprints and runtime identities must match the independent timing run.
The CSV must contain real metric rows and exactly three times the chunk count
in distinct kernel IDs. A mismatch/failure stops subsequent counter runs and
preserves logs. Missing ncu returns2 with `profiler_unavailable`; permission
errors or timeouts are failures, never successful hardware collection.

Sections: LaunchStats, Occupancy, SpeedOfLight, MemoryWorkloadAnalysis,
ComputeWorkloadAnalysis and SchedulerStats. Kernel replay is bounded to the
captured launches; no full section set, source sampling or TopK profiling.
`--cache-control none --clock-control none` avoids requesting cache flushes
or clock changes. These replayed kernels still do not reproduce the graph
benchmark's complete cache/scheduling context; counter duration is not a
speedup measurement. NVTX ranges identify sides; CSV kernel IDs follow their
native/canonical/candidate launch order and each side's original chunk order.
See NVIDIA's [CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
and [section descriptions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

Reports include the ordinary stage-summary/source/runtime/trace files,
`diagnostic-timings.tsv`, per-case A/A raw samples under diagnostics,
`profile-CASE.json/.txt/.csv`, and `profiling-summary.json`. The wrapper uploads
all top-level JSON/TXT/TSV/CSV files on success or failure. It does not install
profilers, modify device permissions/clocks, or kill other GPU jobs. Timeout
cleanup only kills the coordinator's own subprocess group. Initial GPU limit
remains80C. An explicit ncu executable can be provided with `--ncu /path/to/ncu`.
Use `--cases NAME ...` to narrow within these five cases without changing the
full30 benchmark table.

Next decision uses returned evidence: compare A/A and paired A/B, inspect
actual selected configs and counters, then formulate at most1-3 hypotheses.
A probe must retain observable stores, distinguish diagnostic substitutions
from equivalent implementations, and include relevant tails/chunk boundaries.
No production edit until representative improvements exceed measured noise;
then full30 and full numerical regression before integration.

## Local validation and scope

All286 `tools/test_qsa*.py` CPU checks passed using the bundled Python3.12
runtime, including nine focused profiling checks. Changed-file pre-commit,
Python compilation, shell syntax and CLI help passed. This does not establish
CUDA/Nsight compilation, permissions, counters or on-card numerical results.
The full CPU sweep exposed an existing fake-address fixture issue: eight
updated FP8 caches exceeded the old10MB spacing between synthetic independent
allocations. The production-host differential test now uses1TiB spacing; no
production alias check, workload shape or tolerance was changed.

No device-kernel source, production entry, config YAML, workload definition or
reference changed. This collection diagnoses the retained candidate; it is
not a new candidate or an optimization acceptance report.
