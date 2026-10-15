# QSA prefill H800: contiguous stride validation result

Decision: **keep** the nonempty-contiguous stride fast path from
`f8ad07e69629dcf17e13fa65131cfc5fb1962171`. Sixteen short scored workloads
show incremental median wall reductions of 1.50–3.71%, corroborated by lower
unprofiled validation cost and removal of sorting in the CPU profile.
The mixed padded workload has a small observed regression, documented below;
keep accepts this tradeoff, rather than claiming all layouts improved.
This closes one metadata optimization trial, not the whole optimization task.

Result branch: `codex/qsa-h800-results-20261015-064543`;
result commit: `9962c89116f319afea559c58355d742254021b8f`.
Raw [summary](https://github.com/cheersluvs/FlagGems-vllm/blob/9962c89116f319afea559c58355d742254021b8f/results/qsa_select_paged_prefill/h800/stages-20261015-064543/stage-summary.json),
[A/B data](https://github.com/cheersluvs/FlagGems-vllm/blob/9962c89116f319afea559c58355d742254021b8f/results/qsa_select_paged_prefill/h800/stages-20261015-064543/validation-ab.tsv),
[host probes](https://github.com/cheersluvs/FlagGems-vllm/blob/9962c89116f319afea559c58355d742254021b8f/results/qsa_select_paged_prefill/h800/stages-20261015-064543/host-summary.tsv),
[correctness log](https://github.com/cheersluvs/FlagGems-vllm/blob/9962c89116f319afea559c58355d742254021b8f/results/qsa_select_paged_prefill/h800/stages-20261015-064543/accuracy-full.txt).

## Verification and method

All 74 report files passed Git blob SHA-1/size checks. All 11 source snapshots
passed SHA-256 checks and exactly matched tested commit `f8ad07e`; the producer
worktree was clean. Individual case JSON matched the aggregate, and all 2,247
formal stage/metric samples and derived statistics were checked.

The independent correctness gate passed **133 tests**, including all eight new
real Torch stride fixtures and the prior 21 alias/storage-boundary tests. The
only skip requires a second visible SM90 device; 33 native-vLLM cases were
excluded deliberately. All 22 performance cases completed successfully.

Every case selected `overlap-retained`, pinned to `f9d7548`, operator SHA-256
`f11a0d61ce4287e6321ee86a5c1870234cd88e906cb7325a8ccfd9e3106ea14e`.
Both entries shared public and validator bytecode, the retained overlap helper,
and actual score/Top-K callables. Only the stride helper differed. Benchmark
baseline construction copies namespaces and does not mutate production globals.
This measures incremental gain after trial 1, not the combined gain of both
optimizations.

All workers reported the same H800 UUID
`198a9496-3d7a-f01d-ee60-190fe0bc5b51`, Torch `2.13.0+cu129`, imported Triton
`3.6.0` under `/opt/flagtree` (FlagTree distribution `0.6.1`), identical compiler
library fingerprints and ptxas `12.8.93`, with TF32 disabled. The installed
Triton distribution metadata `3.7.1` was not the imported compiler. Compiler
and input fingerprints also match the preceding report, but all gains below
use this run's same-process A/B, not cross-run latency subtraction.

Schema 2 timing is unchanged: seven rounds, 104 calls per stage/mode/round,
Graph batches of eight calls replayed 13 times, rotating stage/mode order.
Warmup/JIT and host profiles are outside formal timing. Wall includes host
execution and amortized batch synchronization. Event time in eager mode can
include host supply gaps; Graph replay excludes repeated CPU validation.
Inputs are reused warm, without cache flush. This is not serving throughput.

## Full-entry measurements

All latencies are medians in microseconds per call. Reduction is `1-new/old`;
speedup is `old/new`. Paired ratio is the median of seven within-round ratios
from separate batches. Negative reduction indicates measured regression.
No aggregate is used to hide workload differences.

| Case | Old wall us | New wall us | Reduction | Speedup | Paired ratio | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| core_0 | 151.517 | 149.239 | 1.50% | 1.0153 | 1.0153 | Host benefit |
| core_1 | 110.089 | 106.659 | 3.12% | 1.0322 | 1.0346 | Host benefit |
| core_2 | 128.027 | 128.496 | -0.37% | 0.9964 | 0.9898 | Small slowdown; device variation unresolved |
| core_3 | 1610.010 | 1546.117 | 3.97% | 1.0413 | 0.9909 | Drift; gain/regression unresolved |
| core_4 | 116.727 | 113.672 | 2.62% | 1.0269 | 1.0274 | Host benefit |
| core_5 | 112.736 | 108.972 | 3.34% | 1.0345 | 1.0315 | Host benefit |
| upstream_bfloat16_s0 | 110.087 | 107.008 | 2.80% | 1.0288 | 1.0316 | Host benefit |
| upstream_bfloat16_s1792 | 132.680 | 128.607 | 3.07% | 1.0317 | 1.0320 | Host benefit |
| upstream_float8_e4m3fn_s0 | 138.727 | 135.802 | 2.11% | 1.0215 | 1.0182 | Host benefit |
| upstream_float8_e4m3fn_s1792 | 110.219 | 107.693 | 2.29% | 1.0235 | 1.0235 | Host benefit |
| zero_visible | 30.668 | 28.305 | 7.70% | 1.0835 | 1.0835 | Host benefit; wall outliers |
| hd_1_32_bfloat16 | 111.555 | 108.970 | 2.32% | 1.0237 | 1.0326 | Host benefit |
| hd_3_64_bfloat16 | 115.657 | 112.888 | 2.39% | 1.0245 | 1.0263 | Host benefit |
| hd_5_96_bfloat16 | 111.300 | 108.059 | 2.91% | 1.0300 | 1.0253 | Host benefit |
| hd_8_256_bfloat16 | 116.660 | 113.833 | 2.42% | 1.0248 | 1.0271 | Host benefit |
| one_row_bfloat16 | 3774.015 | 3764.580 | 0.25% | 1.0025 | 1.0024 | Difference small relative to variation |
| hd_1_32_float8_e4m3fn | 110.154 | 107.285 | 2.60% | 1.0267 | 1.0293 | Host benefit |
| hd_3_64_float8_e4m3fn | 110.783 | 107.481 | 2.98% | 1.0307 | 1.0345 | Host benefit |
| hd_5_96_float8_e4m3fn | 112.243 | 108.804 | 3.06% | 1.0316 | 1.0277 | Host benefit |
| hd_8_256_float8_e4m3fn | 113.997 | 109.772 | 3.71% | 1.0385 | 1.0378 | Host benefit |
| one_row_float8_e4m3fn | 3789.282 | 3783.545 | 0.15% | 1.0015 | 0.9994 | Difference small relative to variation |
| mixed_padded | 157.116 | 158.159 | -0.66% | 0.9934 | 0.9914 | Small observed regression |

All sixteen short scored cases favor the candidate in both ratio of medians
and paired median ratio. This is evidence from one seven-round run across
multiple workloads, not an independent repeat campaign or tail-latency
confidence guarantee. Wall outliers remain: core_1 candidate spans
106.027–139.680 us, while baseline spans 109.836–131.307 us.

## Mechanism and tradeoff

Core_1 unprofiled host validation decreases **13.089 -> 11.196 us** (14.47%),
with all seven candidate probe samples below all baseline samples. The full
wall median decreases **110.089 -> 106.659 us** (3.12%, 1.0322x). Probe and
full-entry durations come from separate batches and are not additive cost
accounting. cProfile shows 600 stride checks and 600 sorts over 200 baseline
validations, versus 600 checks and no sorts for the candidate. Instrumented
durations themselves are not production latency estimates.

For `mixed_padded`, all three inputs still require the original conservative
stride algorithm, so the candidate pays three extra `is_contiguous()` calls
without removing sorting. Its validation probe increases **13.227 -> 13.570 us**
(+0.342 us); all candidate wall-probe samples exceed all baseline samples.
Full wall increases **157.116 -> 158.159 us** (+1.043 us, +0.66%), paired ratio
0.9914. Eager event medians are 156.503 -> 156.616 us. The probe supports a
real fallback-path metadata cost; it does not explain the entire wall
regression quantitatively. We retain this bounded small cost in exchange for
consistent benefits across the sixteen contiguous short workloads and the
select-all case. It must remain visible in future layout regression checks.

Core_2 wall increases 0.37% (128.027 -> 128.496 us), paired ratio 0.9898.
Its several-percent GPU/batch variation does not resolve the change as a
repeatable end-to-end regression or benefit. The two one-row-chunk cases show
only 0.25%/0.15% median reductions; FP8's paired ratio is below one. Neither is
counted as established end-to-end improvement.

Core_3 still drifts substantially: baseline wall samples span
1310.002–1806.198 us and candidate samples 1364.650–2014.648 us. Ratio of
medians is 1.0413, whereas paired median is 0.9909; Graph median is 3.50%
slower even though device work is shared. Boundary temperatures rise from
69 to 85 Celsius, with 1590 MHz SM clocks and inactive slowdown reasons in
boundary snapshots. These do not establish the in-kernel cause. Neither an
improvement nor a regression magnitude is accepted for this long case.

For the sixteen short scored cases, full Graph median differences are within
0.27%; core_1 is **19.138 -> 19.146 us**. The zero-visible case is
1.454 -> 1.480 us (about 25 ns slower, 1.73%) with much larger scatter in some
samples. No GPU kernel acceleration or diagnosed device regression follows
from these numbers. The accepted benefit is host validation in eager calls;
Graph replay does not repeat that validation.

## Retained production scope

The only production edit remains the four-line contiguous/nonempty guard in
`_check_nonoverlapping_strides`. Empty tensors deliberately retain the old
checker even when Torch reports them contiguous. All rank/dtype/device/stride
and alias rules, including strided padding holes, remain intact. The public
entry, workspace, score/fill kernels, score launcher and Top-K are unchanged.
No Torch compute/copy/cast, global cache, stream or lifetime change was added.
The existing six score tuning candidates and keys remain unchanged; there is
no new kernel requiring an additional autotune configuration.

This is portable host metadata logic. Only H800 was measured; other vendors
remain untested, and the existing operator device gate is still NVIDIA SM90,
BF16/FP8, inference-only. The native-vLLM FlagTree issue is not addressed by
this trial, and no native-vLLM performance comparison is claimed. Shared
runtime launch overhead and stable long-case measurement remain separate
future work, outside this keep decision.
