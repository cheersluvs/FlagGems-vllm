# QSA prefill H800: prepared chunk setup result

Decision: **keep** metadata preparation and two-dimensional workspace reuse
from `85a78e6aacfb04e62ad919646f8f82c5178b97ce`. The main targeted workload,
36 one-row chunks, reduces full wall medians by **5.31% BF16 / 5.28% FP8**.
For each dtype, every candidate wall sample is below every baseline sample.
The 32/32/9 tail fixtures also favor the candidate in within-round medians.
No correctness failure or clear material full-entry regression was observed.
This closes the host chunk-setup trial; it does not complete all QSA
optimization, establish GPU kernel acceleration or compare with native vLLM.

Result branch: `codex/qsa-h800-results-20261015-074809`;
result commit: `9e945b3d15782114838abfd16cd311d4de3e6b38`.
Raw [summary](https://github.com/cheersluvs/FlagGems-vllm/blob/9e945b3d15782114838abfd16cd311d4de3e6b38/results/qsa_select_paged_prefill/h800/stages-20261015-074809/stage-summary.json),
[full-entry A/B](https://github.com/cheersluvs/FlagGems-vllm/blob/9e945b3d15782114838abfd16cd311d4de3e6b38/results/qsa_select_paged_prefill/h800/stages-20261015-074809/host-ab.tsv),
[accuracy log](https://github.com/cheersluvs/FlagGems-vllm/blob/9e945b3d15782114838abfd16cd311d4de3e6b38/results/qsa_select_paged_prefill/h800/stages-20261015-074809/accuracy-full.txt),
[one-row BF16 profile](https://github.com/cheersluvs/FlagGems-vllm/blob/9e945b3d15782114838abfd16cd311d4de3e6b38/results/qsa_select_paged_prefill/h800/stages-20261015-074809/one_row_bfloat16.host-profile.txt),
[long-case telemetry](https://github.com/cheersluvs/FlagGems-vllm/blob/9e945b3d15782114838abfd16cd311d4de3e6b38/results/qsa_select_paged_prefill/h800/stages-20261015-074809/core_3.json).

## Verification and actual environment

All 80 report files passed Git blob SHA-1 and size verification. All 12 source
snapshots passed SHA-256 checks and exactly matched tested commit `85a78e6`;
the producer worktree was clean. Aggregate and individual case JSON agree.
All 2,457 formal stage/metric samples and derived statistics were recomputed.

The independent correctness gate passed **139 tests**, including all six new
BF16/FP8 single-chunk, one-row and cross-request tail tests. The 21 prior alias
and eight stride fixtures also passed. One test requiring a second visible
SM90 GPU skipped; 33 native-vLLM tests were deliberately excluded. All 24
benchmark cases completed successfully. Baseline and candidate benchmark
outputs were checked before timing and after eager/Graph timing.

All workers actually used:

| Component | Imported/executed path or version |
| --- | --- |
| Python | /tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python |
| Torch | 2.13.0+cu129, from the virtual environment |
| Triton module | /opt/flagtree/triton/__init__.py, __version__ 3.6.0 |
| Compiler library | /opt/flagtree/triton/_C/libtriton.so |
| Compiler library SHA-256 | 0435ea3da94763e73816bd48d9bf3565e258ba709eb0167a382b067ec862357e |
| ptxas | /opt/flagtree/triton/backends/nvidia/bin/ptxas, 12.8.93 |
| H800 UUID | 198a9496-3d7a-f01d-ee60-190fe0bc5b51 |

The compiler is loaded through external `/opt/flagtree` path injection, not
inferred from pip packages installed in the venv. Distribution metadata lists
Triton 3.7.1 and FlagTree 0.6.1 on the process search path; those labels alone
are not evidence of the imported implementation or venv installation. Actual
compiler/library/assembler fingerprints are identical across all 24 workers
and match the preceding report. TF32 was disabled. The original 22 input
fingerprints also match the previous report; gains below still use only this
run's same-process comparison.

## Comparison scope

`host_ab.kind` is `chunk-setup`, baseline commit `ab50d7a`, operator SHA-256
`c424d97849f6276a0716dd573e8387acea3eaefb222fd62d334d49931e3b4a58`.
All cases report shared score kernel, fill kernel, Top-K, validator bytecode
and overlap helper. Public-entry bytecode and score host-launcher identity
differ intentionally. Exact old host sources and exact candidate sources
are pinned; the baseline uses a separate namespace and does not patch
production globals. Both retained validation optimizations are on both sides.

Schema 2 remains seven rounds of 104 calls per stage/mode, with stage/mode
rotation. Graph uses 13 replays of eight captured calls. Warmup/JIT and CPU
profiles are outside formal timing. Wall includes host work and amortized
batch synchronization; eager events may include host supply gaps. Inputs
are reused warm. No serving throughput or cold-start behavior is measured.
The isolated score/pipeline stages now reuse observed prepared metadata;
cross-report stage subtraction is not used to attribute gains.

## Full-entry wall measurements

Microseconds per call, medians over seven batches. Reduction is `1-new/old`;
speedup is `old/new`. Paired ratio is the median of the seven within-round
old/new ratios. The batches are separate measurements, not simultaneous.
There is no aggregate hiding workload-specific uncertainty.

| Case | Old wall us | New wall us | Reduction | Speedup | Paired ratio | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| core_0 | 144.687 | 143.587 | 0.76% | 1.0077 | 1.0077 | Small median benefit; wall outliers |
| core_1 | 106.269 | 103.428 | 2.67% | 1.0275 | 1.0279 | Host median benefit |
| core_2 | 128.953 | 127.877 | 0.83% | 1.0084 | 1.0036 | Small difference; benefit unresolved |
| core_3 | 1710.256 | 1657.746 | 3.07% | 1.0317 | 0.9905 | Thermal throttling observed; do not adopt gain |
| core_4 | 106.281 | 103.814 | 2.32% | 1.0238 | 1.0293 | Host median benefit |
| core_5 | 129.845 | 125.983 | 2.97% | 1.0306 | 1.0313 | Host median benefit |
| upstream_bfloat16_s0 | 108.260 | 104.811 | 3.19% | 1.0329 | 1.0341 | Host median benefit |
| upstream_bfloat16_s1792 | 133.664 | 129.572 | 3.06% | 1.0316 | 1.0272 | Host median benefit |
| upstream_float8_e4m3fn_s0 | 111.602 | 107.700 | 3.50% | 1.0362 | 1.0343 | Host median benefit |
| upstream_float8_e4m3fn_s1792 | 113.671 | 109.578 | 3.60% | 1.0374 | 1.0387 | Host median benefit |
| zero_visible | 28.880 | 28.969 | -0.31% | 0.9969 | 1.0006 | Unchanged path; no resolved wall difference |
| hd_1_32_bfloat16 | 127.846 | 124.747 | 2.42% | 1.0248 | 1.0271 | Host median benefit |
| hd_3_64_bfloat16 | 108.314 | 105.611 | 2.50% | 1.0256 | 1.0268 | Host median benefit |
| hd_5_96_bfloat16 | 110.918 | 107.260 | 3.30% | 1.0341 | 1.0341 | Host median benefit |
| hd_8_256_bfloat16 | 107.300 | 103.812 | 3.25% | 1.0336 | 1.0384 | Host median benefit |
| one_row_bfloat16 | 3714.773 | 3517.536 | 5.31% | 1.0561 | 1.0561 | Clear wall benefit across all seven rounds |
| hd_1_32_float8_e4m3fn | 111.546 | 107.135 | 3.95% | 1.0412 | 1.0412 | Host median benefit |
| hd_3_64_float8_e4m3fn | 114.507 | 110.008 | 3.93% | 1.0409 | 1.0391 | Host median benefit |
| hd_5_96_float8_e4m3fn | 113.531 | 109.866 | 3.23% | 1.0334 | 1.0375 | Host median benefit |
| hd_8_256_float8_e4m3fn | 108.622 | 104.985 | 3.35% | 1.0346 | 1.0320 | Host median benefit |
| one_row_float8_e4m3fn | 3976.465 | 3766.562 | 5.28% | 1.0557 | 1.0532 | Clear wall benefit across all seven rounds |
| mixed_padded | 158.199 | 156.450 | 1.11% | 1.0112 | 1.0084 | Small median benefit; wall outliers |
| tail_chunk_bfloat16 | 363.207 | 352.746 | 2.88% | 1.0297 | 1.0293 | Tail benefit; wall outlier |
| tail_chunk_float8_e4m3fn | 450.725 | 437.212 | 3.00% | 1.0309 | 1.0307 | Paired benefit; host latency level shifts |

The one-row BF16 baseline spans 3705.643–3737.960 us and candidate
3515.157–3554.052 us; FP8 spans 3956.982–4002.120 vs
3742.467–3798.884 us. There is no overlap between old/new wall sample ranges
for either dtype. Same-round median ratios are 1.0561 BF16 and 1.0532 FP8.
Graph medians are 591.025 -> 591.231 us and 556.054 -> 556.283 us, both within
0.05%. This supports an eager host saving, with unchanged device computation.

Tail records confirm offsets/counts **(0,32), (32,32), (64,9)**, workspace
shape **(32,8192)** for every chunk, prepared metadata, direct workspace
identity for the first two chunks and a distinct prefix view for the tail.
Wall medians decrease 2.88% BF16 and 3.00% FP8, with paired ratios 1.0293 and
1.0307. The FP8 tail host latencies change levels during the run: baseline
rounds 1–3 are 373–378 us, rounds 4–7 are 451–478 us; candidate rounds 4–7
are 437–440 us, with an earlier first-round outlier. The paired direction
supports benefit, but the absolute 450.725/437.212 us medians are not stable
latency guarantees. Its eager-event ratio of medians is 1.1497 versus paired
1.0324; the apparent 15% event speedup is not adopted.

Seventeen short scored cases, including the mixed padded fixture, show wall
median reductions of 0.76–3.95%, with favorable paired medians. Small cases
remain noisy: core_0 paired mean is below one despite favorable medians;
core_1 candidate has a first-round 133.861 us outlier vs a 103.428 us median.
The table establishes median behavior within this run, not tail-latency or
independent-repeat confidence guarantees. Core_2's 0.84% median improvement
and 0.36% paired benefit remain unresolved against its variation.

## Mechanism

For 200 one-row BF16 public invocations, both profiles record 7,200 score
launcher calls (36 per invocation). The baseline records 36,000 Tensor view
calls and 50,800 stride calls; the candidate records **28,800 views** and
**8,800 stride calls**, plus exactly 200 `_prefill_score_metadata` calls.
That removes 36 score-workspace views and 210 repeated stride reads per
public invocation. Views still present belong to unchanged Top-K internals.
The counts corroborate the hypothesis; instrumented durations are not
production latency and are not added to the formal timings.

Unprofiled validation is effectively unchanged, as expected: core_1 baseline
11.205 vs candidate 11.266 us. The optimization acts after validation,
without a global cache, new device-value reads or reduced validation scope.
The real functional fixtures also check exact score values, workspace
pointer/stride/shape, same metadata identity across chunks and final selection.

## Long-case thermal evidence and controls

For core_3, baseline wall spans **1311.243–1989.004 us**, candidate
**1375.368–1982.458 us**. Ratio of medians is 1.0317 but paired median is
0.9905; the apparent 3.07% improvement is excluded from accepted benefits.

This report provides new direct evidence: the `after_round_4` clocks snapshot
at **2026-10-15 15:39:21.619** reports **86 Celsius and SM 1230 MHz**, compared
with 1590 MHz in the other boundary snapshots. The corresponding performance
query reports **SW Thermal Slowdown: Active**. Thus thermal throttling is
confirmed at that measurement boundary. It does not prove every slow batch
or every earlier report had the same cause, nor provide continuous in-kernel
frequency traces. A stable long-case baseline is still required before
claiming its speedup or regression; these samples are not normalized by clock.

The zero-visible path does not execute the changed chunk setup. Its wall
28.880 -> 28.969 us and paired ratio 1.0006 do not resolve a change. Its Graph
median is 1.446 -> 1.494 us (+48.6 ns, +3.36%), with larger outliers in both
series. This small observed Graph difference is recorded; its cause is not
resolved, and this trial does not claim Graph replay acceleration or establish
no change at the tens-of-nanoseconds scale.

## Retained scope and remaining work

Keep only the implemented per-call scalar preparation and contiguous 2-D
workspace reuse. Complete chunks reuse the same tensor; a partial tail uses
a prefix-row slice. Legacy private flat-buffer calls still work. Allocation
bytes, chunk budget/counts, Top-K scratch, current-device/stream context and
lifetime remain unchanged. No Torch compute/copy/cast was added. All validation,
GPU kernels, Top-K implementation, shared libentry, numerical references and
the six score tuning candidates/keys remain unchanged; no new kernel or
additional autotune configuration was introduced. Exports/registration are
unchanged.

The host technique is portable, but only NVIDIA H800 has been tested. Existing
SM90 BF16/FP8 inference-only support remains the boundary; other vendors are
unmeasured. GPU scoring optimization and stable long-case measurement remain
future work. This trial does not change the native-vLLM compiler issue or
establish native-vLLM alignment/performance on FlagTree.
