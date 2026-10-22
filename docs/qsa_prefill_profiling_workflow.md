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

## Returned collection ae841236: counters unavailable, timings preserved

Report `ae841236c23e7b29c76639331da7a341d83a6848`, source c362d4e,
completed all five independent timing workers. Exit2 is the explicit
`profiler_unavailable` result: `ncu=null`, `profiles=[]`. It is not a numerical
or benchmark-worker failure. Hardware counters have not been collected.

Audit verified26 report blobs,124 source snapshots against c362d4e,152 ordinary
raw timing samples,120 A/A samples and204 clean GPU observations. Temperature
33–51C and sampled SM clock1980MHz. No foreign PID rejection was observed.
The telemetry span is336.47s, not total setup/JIT duration. Full pytest was
explicitly skipped; pre/post timing numeric checks passed.

All ratios below are same-run baseline/candidate, kernel mode. The candidate
is private stage-joint, not a newly integrated production operator.

| Case | Score/native | TopK/native | Pipeline/native | Score/canonical | Pipeline/canonical |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 n1/q16384/p102400 | 1.183783 | 0.786475 | 0.981336 | 1.124731 | 1.067775 |
| BF16 short96 | 6.339250 | 0.741110 | 2.201428 | 1.007843 | 1.006870 |
| BF16 short_batch3 | 4.366801 | 1.017585 | 2.433667 | 1.046854 | 0.939458 |
| FP8 tail112 | 1.134893 | 0.963206 | 1.026438 | 1.105439 | 1.037542 |
| FP8 long100k | 1.061236 | 0.785745 | 0.909650 | 1.098643 | 1.040603 |

A/A does not remove the timing-context issue. Short96 candidate pipeline is
25.760/25.792us in the native comparison but16.288/16.320us in the canonical
comparison; later identical-function A/A samples are26.112–26.624us. For
short_batch3, canonical/candidate pipeline paired ratios are1.003623/0.874310;
its later candidate A/A is1.001453 in both rounds at22.016/22.048us. No stable
6% regression or cause is established from the0.939458 median ratio. BF16
long native score A/A is1.034476/1.025064, so even long-case controls are not
all perfectly centered. Keep all samples; do not silently discard these rows,
attribute differences to a compiler/cache mechanism, or treat two rounds as a
confidence interval. Independent phase medians cannot be added to bound the
pipeline speedup.

### Resume scope, frozen before implementation

| Field | Choice |
| --- | --- |
| Editable | Counter-loop extraction, bounded ncu discovery, resume coordinator/wrapper and focused CPU tests/docs |
| Read-only | All original measurements, canonical/private/native kernel sources and configs, original worker, reference, timer and guards |
| Dispatch | Verify all archived original sources/tools, copy reports to a new output directory, launch original worker in profile-capture mode only |
| Runtime | Reuse original libcuda/ptxas/FlagTree/Python/GPU UUID and compiler caches; original worker still requires matching inputs/configs/binaries/runtime |
| Acceptance | Hardware diagnostic only; no benchmark rerun, new timings, production change or inferred performance acceptance |

`--resume-report /tmp/qsa-profiling-h800.Ji2cwS` preserves the returned five
measurements and A/A. It verifies the original checkout and124 source hashes,
reference and compatibility/assembler identities before copying original report
files to a fresh output folder. The new resume driver is archived separately
in resume-sources.json. Only the original c362d4e worker executes under ncu;
no old code files are overwritten. Original caches, library paths and GPU UUID
are reused so matching remains strict. If these temporary files are gone or
the original GPU is occupied, capture stops rather than rerunning benchmarks
or switching the comparison runtime.

Profiler discovery now checks PATH and bounded common /usr/local and /opt
CUDA/Nsight installation directories. `--ncu /absolute/path/to/ncu` overrides
that discovery. No package installation, sudo or device-permission change is
performed. If still absent, return2 and upload discovery status without
repeating timing. Installing/exposing Nsight Compute is then the missing
prerequisite; CUDA compatibility libraries and ptxas do not provide ncu.

Resume implementation validation:49 affected CPU checks passed (including
source/tool tamper rejection, original report/log preservation, pinned worker
paths/reference copying, missing ncu/counter failures and normal benchmark
regression checks). Changed-file pre-commit and shell syntax passed. CUDA/ncu
execution remains unverified locally; no production files/configs changed.
