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

## Returned resume48e8a343: provision the missing profiler

Report48e8a343932f6f4db822c6b4f16647d2b6bbc849 verified30 archived blobs.
`profiling-summary.json` confirms profiler_unavailable, ncu=null, profiles=[],
independent_benchmark_rerun=false and124 verified original sources. Original
measured case files and source snapshots remain unchanged. No new latency
measurements or hardware-counter conclusions are available.

Before implementation, freeze the next change: only a standalone SHA-pinned
profiler provisioner, its CPU tests and the resume GPU-UUID normalization;
all original worker/kernel/timing/configuration files stay read-only. The
returned runtime UUID is bare2d17872f-4327-2beb-8049-604ecf6a3d66, whereas the
NVML/CUDA-visible guard requires GPU- prefix. Normalize either representation
to the same full UUID and reject malformed/mixed GPU records; this correction
must not choose another GPU or weaken idle checks.

Resume option `--prepare-ncu` searches for an existing ncu first, then downloads
the standalone Linux-x86_64 Nsight Compute2026.3.1.2 archive only when absent.
The [official CUDA13.4.3 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json)
pins397165560 bytes and SHA256
`a0039be47cfe551f89d82087256080a9f33c6257d99a72c1164a87466ab072aa`.
The archive is cached in /tmp/qsa-ncu-2026.3.1.2.tar.xz and extracted into a
fresh /tmp/qsa-ncu-portable.* prefix using Python3.12's data extraction filter.
It does not execute an installer or touch apt, drivers, system CUDA, FlagTree,
permissions or clocks. A matching local archive can be supplied with
`--ncu-archive /absolute/archive.tar.xz`. Version, extraction path, executable
hash, archive hash/size and errors are saved in ncu-preparation.json, and the
new provisioner source is archived in resume-sources.json.

Only CLI --version/section listing is checked before the original hardware
capture. Successful extraction does not prove CUDA13.4 instrumentation or
counter permissions; retain the actual first capture error if either fails.
No further benchmark/A/A/standalone pytest is run in this resume mode.

Local provisioner/UUID validation:55 affected CPU checks passed, including
size+digest validation, safe archive extraction/link rejection, local-archive
reuse without network, version metadata and bare/full GPU UUID normalization.
Changed-file pre-commit, CLI help and shell syntax passed. The real397MB Linux
archive was not downloaded/executed on the Mac; on-card loading, instrumentation
and hardware-counter permissions remain to be checked by the resumed run.

## Returned capture d9d8b953: successful first hardware capture, CSV parser bug

Reportd9d8b9535448bea349432b3392c50c0902ad61fe has34 verified blobs.
The standalone archive SHA256 matched and ncu --version/section listing passed.
The first worker (BF16 n1/q16384/p102400) and ncu both exited0. The original
worker validated input/config/cubin/runtime matching, pre/post numeric checks
and clean GPU guards; all three sides and four chunks completed. Hardware
counter permissions worked for these kernels. Other four cases were not run.

Coordinator failure was a parser compatibility bug: it expected long CSV with
Metric Name/Metric Value columns, but ncu --page raw emitted a wide table with
804 columns, a separate units row and12 kernel rows. These contain9396
metric values (including device/resource attributes), not zero metrics. NVTX
labels confirm IDs0-3 native,4-7 canonical production,8-11 private candidate.
The false empty-export diagnosis is withdrawn; the original failed status is
preserved as provenance, and the verified first capture can be recovered.

Before coding, restrict this fix to long/wide CSV normalization, exact-case
counter reuse and focused CPU tests. No capture timer, hardware section,
replay/cache/clock policy, original worker, kernel/config/reference or numeric
guard changes. Reuse requires passed worker/pre-post validation, matching
baseline inputs/case/configs/native+canonical+candidate cubins/runtime, clean
GPU guard, all three side labels in the capture manifest and exactly the
expected kernel IDs. Preserve original JSON/CSV/TXT bytes and their hashes.
An invalid or partial prior capture must fail reuse, never silently pass.

Selected four-chunk configurations:
canonical TR32/BN64/KT16/W8/stages2;
candidate TR64/BN64/KT16/W4/stages1/QUERY_PARTS1.
Ranges below are per-kernel counters across the four chunks, not averages of
benchmark speedups. Shared memory is the dynamic per-CTA portion.

| Observation | Native | Canonical production | Private candidate |
| --- | ---: | ---: | ---: |
| Registers/thread | 192 | 63 | 234 |
| Dynamic shared bytes/CTA | 106496 | 73728 | 81920 |
| Active warps, % peak sustained active | 12.29–12.66 | 36.31–36.82 | 12.06–12.30 |
| Tensor cycles active, % peak sustained active | 23.97–24.50 | 25.55–25.67 | 29.03–29.53 |
| Issue active, % peak sustained active | 25.88–26.33 | 46.67–46.97 | 34.07–34.61 |
| DRAM throughput, % peak sustained elapsed | 27.11–28.90 | 28.44–29.75 | 33.24–34.66 |
| L2 sector hit rate, % | 89.87–93.80 | 96.92–97.60 | 94.35–95.49 |
| Eligible warps/scheduler/active cycle | 0.32–0.33 | 1.20–1.21 | 0.44–0.45 |

The independent original benchmark measured candidate/canonical score
speedup1.124731. Lower active-warp percentage does not imply lower performance:
the larger query tile, different warp/stage counts and BF16 page fetch change
coexist here. This is evidence of a reuse/live-state tradeoff, not a causal
proof that registers, occupancy or DRAM bandwidth is the sole bottleneck.
Capture uses kernel replay, uncontrolled caches and unmodified clocks, so
profiled duration is not substituted for graph-mode benchmark speedup. Do not
use the high warm-cache L2 hit rate to promise the same serving-cache behavior.

Next: use --reuse-profile-report /tmp/qsa-profiling-h800.70fDb1 with the original
--resume-report /tmp/qsa-profiling-h800.Ji2cwS. Reuse all12 first-case launches
without remeasurement, collect only remaining four cases, and reuse the already
extracted /tmp/qsa-ncu-portable.5qd8fj5x/.../ncu via --ncu. No download,
benchmark/A/A/full pytest or production modification is required. After the
remaining FP8/tiny counters, choose a fixed-configuration, one-hypothesis probe
for query reuse versus accumulator lifetime, preserving full-D math and stores.

Validation:58 affected CPU checks passed, including long/wide export parsing,
unit-row/zero/N/A handling, exact binary matching for reuse and a check that
reused cases launch no worker. The actual d9d8b953 CSV/JSON was recovered locally
against the original baseline and yielded12 matching IDs and9396 values.
Changed-file pre-commit and shell syntax passed. Remaining four hardware
captures and all proposed optimization probes are still pending H800 execution.

## Console reporting scope (2026-10-23, before implementation)

| Field | Frozen choice |
| --- | --- |
| Editable | Read-only report formatter/CLI, post-worker/post-capture print hooks, wrapper text artifact, focused CPU display tests and this document |
| Read-only | Production/private/native kernels, configs/keys, builders, input data, timers, numerical/idle guards and raw metrics |
| Console timing | Separate native/candidate and canonical/candidate tables with actual segment latencies; score/TopK/pipeline, A/A descriptive ratios and saved source identity |
| Console counters | Per-side/per-chunk register/shared/warp/tensor/DRAM/L2/issue/config rows; profiler durations are never speedup samples |
| Missing/anomalous | N/A for absent values; retain and flag out-of-range percentages, especially two FP8 tiny-tail L2 ratios>100; no clamping |
| Offline display | Existing report folder only, no GPU, imports of Torch/Triton, downloads or rerun; display old reports immediately |
| Acceptance | Display status and actual validation scope, not a new optimization result or full30 acceptance |

## Complete five-case capture3621d9ca

Report3621d9cac585ca730196ffd9adc427c3b6e6d6f0 verified45 report blobs and
four resume-driver source snapshots against32bdb357. All five case captures
pass exact original case/input/config/native+canonical+candidate binary/runtime
matching and pre/post numeric checks. NVTX confirms36 total scorer launches:
12 BF16 long (reused),3 short96,3 short_batch3,6 FP8 tail112 and12 FP8 long.
There are28188 exported metric values, including device/resource attributes.
Ten boundary GPU guards were clean at30–32C; these are snapshots, not a
continuous guarantee. The independent benchmark files are byte-identical to
ae841236; no benchmark/A/A/full pytest was rerun and no new speedup is claimed.

For FP8 long100k the canonical TR64/BN64/KT16/W4/stages2 contrasts with the
candidate's same tile/warps and stages1/QUERY_PARTS2. Registers/thread226->154,
dynamic shared bytes57344->49152, active warps12.10–12.30%->17.39–18.07%,
tensor-active18.08–18.37%->19.87–20.17%, eligible warps/scheduler0.46–0.48->
0.69–0.71. These multiple changes coexist; this is not isolated causality for
query splitting. The original same-run score/native1.061236, TopK/native
0.785745 and pipeline/native0.909650 remain the performance observations.

The FP8 tail112 main chunk uses the same154-register split configuration;
its112-row tail uses the unchanged54-register small scorer. Treat those
chunks separately. Two raw L2 derived ratios on the tiny tail are101.72%
(canonical ID3) and102.94% (candidate ID5). Preserve and flag them; do not clamp
them or use them as reliable physical hit-rate evidence. Short96/short_batch3
have very low warm-cache DRAM activity, so pursue their launch/work distribution
and graph-context behavior separately rather than applying long-tile reasoning
indiscriminately. The cross-timing-segment pipeline discrepancy remains open.

Next probe priority: fix the BF16 long candidate's actual TR64/BN64/KT16/W4/
stages1 configuration and isolate query subtiles1 versus2, preserving full-D
FP32 dot/per-head ReLU/sum and stores. This tests accumulator lifetime against
query reuse; it does not assume higher occupancy automatically wins. Only after
that controlled result consider further FP8 subdivisions. Neither probe is
implemented or measured by this console-only change. No production integration
is justified by a hardware-counter improvement alone.

## Console and offline viewing

After each ordinary benchmark worker, the engine prints native/candidate
score,TopK,pipeline latencies and ratios, then canonical/candidate comparisons
in a separate table, followed by available A/A ranges. Captured/reused profiles
print per-ID/side/chunk offset/rows, actual configurations, registers,dynamic
shared KiB,active-warp/tensor/DRAM/L2/issue metrics and percentage anomalies.
The tables deliberately keep candidate latencies from independent measurement
segments separate. Missing fields print N/A; zero remains zero. Counter
percentages retain their active/elapsed denominator labels, and replay duration
is never presented as a benchmark speedup.

The profiling wrapper saves terminal-summary.txt alongside raw reports and
uploads it. Its formatter status does not replace the measured run's exit code.
The same CLI can immediately read the completed report without any GPU,
Torch/Triton import, download or retiming:

```bash
python -I tools/qsa_prefill_report.py \
  --report-dir /tmp/qsa-profiling-h800.RFNO3i
```

`--timing-only` omits the hardware tables. The full display was exercised on the
actual3621d9ca report locally and retained under the task's analysis artifacts.

Validation: all62 affected CPU checks passed, including isolated offline display,
per-segment timing preservation, TopK visibility, NVTX chunk/config mapping,
missing/zero/unit handling and anomalous derived percentages. Changed-file
pre-commit, shell syntax and diff checks passed. The CLI also rendered the full
actual3621d9ca report successfully. No GPU run was performed for this change.
