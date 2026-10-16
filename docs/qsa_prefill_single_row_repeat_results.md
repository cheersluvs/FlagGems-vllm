# QSA single-query vector trial: controlled H800 repeat

Decision: do not enable the current vector prototype in production. The
independent correctness gate and GPU environment checks pass, but all 15
vector-using cases have a higher full-entry wall median than the unchanged
production entry. Retain the experimental code as a reproducible artifact.
No production kernel, launcher, validation, Top-K or tuning config is changed
in recording this decision. The existing production implementation remains
the best retained implementation under the frozen wall-latency criterion.

## Provenance and audit

- Code: `13ce7d5cfcbcbba1f9008ced12942c346c8e0bae`.
- Result branch: `codex/qsa-h800-results-20261016-002939`.
- Result commit: `0025302fefd2edbdb18edf8d07f6f29cd6dc3dbe`.
- Report directory: `results/qsa_select_paged_prefill/h800/stages-20261016-002939/`.
- Baseline operator: `454074da0c079cc649d139489ce11fd3141585b7`, SHA256 `d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73`.
- Run exit: 0; correctness: 161 passed, 1 second-device skip, 33 native-vLLM cases deselected (326.06 seconds). All 22 vector-trial tests passed.
- All 38 benchmark cases passed; 15 used at least one vector-scored chunk.
- Verified 122 downloaded files against 85 unique Git blob hashes, all 15 source snapshots against the code commit, and 3,927 formal timing samples with their recomputed statistics/paired ratios.

Both entries run in each case's same process on physical GPU 1, UUID
`GPU-497db188-9a65-ed86-3516-2de784681f95`, SM90 H800. The actual imports
are venv Python/Torch 2.13.0+cu129 and
`/opt/flagtree/triton/__init__.py` (Triton 3.6.0), with ptxas 12.8.93.
Compiler sources, libtriton and ptxas hashes match across all 38 cases.
Top-K, validators, fill and the fallback dot kernel are shared; eligible
chunks deliberately use the distinct experimental vector scorer. This is
production self A/B, not a native-vLLM speedup or a cross-compiler comparison.

## Environment checks

- All 39 idle checks passed on their first attempt: one before correctness and one before each worker. GPU processes were absent at these checks; temperature was 27–35 C.
- All 304 timed-run snapshots passed the guard. Each process query contains exactly the identified benchmark worker PID, with no observed foreign compute PID.
- No sampled SW or HW thermal slowdown; temperatures 29–45 C.
- All 304 SM-clock readings are 1980 MHz.
- Neither ncu nor nsys was available.

These checks provide a controlled run suitable for this trial's comparison;
they are sampled evidence rather than a hardware reservation. Do not compare
these absolute latencies against the rejected GPU 0 report from the previous
run. Both sides were freshly tuned for this selected GPU.

## Full-entry results

Wall time includes host dispatch and synchronization around an eager batch.
It remains the frozen primary metric. Graph time is diagnostic and measures
full graph replay per captured operator invocation. No stage-time subtraction
or summation is used to infer an unmeasured stage or CPU cost.

All 15 vector-using cases have a wall-median increase of 0.249–2.054%; all 15
also have a paired-round ratio median below 1. For the two 36-chunk one-row
cases the candidate is slower in all seven paired wall rounds, even though
Graph replay improves. This does not satisfy the acceptance criterion.

| Case | Production wall us | Candidate wall us | Wall increase | Production Graph us | Candidate Graph us | Graph latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one_row_bfloat16 | 3767.418 | 3796.493 | +0.772% | 489.829 | 466.780 | +4.706% |
| one_row_float8_e4m3fn | 3595.466 | 3607.162 | +0.325% | 463.320 | 459.558 | +0.812% |
| single_4_128_bfloat16 | 144.987 | 147.964 | +2.054% | 17.918 | 17.749 | +0.941% |
| single_8_256_bfloat16 | 145.928 | 146.903 | +0.669% | 19.465 | 19.674 | -1.075% |
| single_8_256_float8_e4m3fn | 153.784 | 154.168 | +0.249% | 18.674 | 19.126 | -2.417% |
| tail_one_bfloat16 | 348.844 | 350.922 | +0.596% | 64.189 | 64.400 | -0.328% |
| tail_one_float8_e4m3fn | 350.381 | 352.051 | +0.477% | 64.941 | 65.213 | -0.418% |

The BF16 one-row Graph benefit is 23.049 us (4.706% lower latency, 1.04938x
speedup). FP8 saves 3.762 us (0.812%, 1.00819x). These are valid observations
for those full Graph fixtures; they do not establish an eager/wall benefit.
H=8,D=256 also regresses under Graph (BF16 +1.075%, FP8 +2.417% latency),
so the current H<=8,D<=256 region is not uniformly faster on the GPU.

The warmed vector configurations observed across 15 cases were BLOCK_N=16,
warps=4 (6 cases); 32,4 (2); 64,4 (1); 64,8 (6). TILE_R=K_TILES=1 and
num_stages=1 throughout. Actual PTX/CUBIN hashes are preserved in each report.
No candidate configuration is changed or selected globally from this audit.

## Limits of causal attribution and next direction

The experiment changes both scoring implementation and its private launcher
route. It uses a cloned module for candidate isolation. Fallback-only cases
also exhibit small wall shifts while their GPU work is shared. Therefore the
wall increase cannot be attributed entirely to vector arithmetic or to the
new host condition. The current prototype fails the criterion as measured;
this is not proof that vector scoring can never help an eager production path.

CPU profiles are instrumented diagnostics, not acceptance timings. For 200
one-row entries, both sides perform 7,200 score launcher calls, 7,200 Top-K
calls and 21,600 GPU driver launcher calls. Launch count is unchanged. The
large gap between full wall and Graph replay and the retained call counts
motivate investigating host/launch overhead; subtracting these methods would
not provide its exact cost.

A useful next experiment would add a dot-only private-module control with
the same host route as the vector candidate, rebuilding all comparisons in
one guarded run. This can separate framework/route cost from the score-kernel
change before any production integration or further vector tuning. Continue
from the retained production implementation for other generic host optimizations.
Top-K stays frozen. A Graph-specific production path would require an explicit
scope decision and its own acceptance criteria, rather than silently replacing
the wall metric. No new experiment or GPU run is performed by this audit.

## Complete wall medians

Change is `100 * (candidate / production - 1)`; positive means slower.
Results are per case, without averaging unlike workloads.

| Case | Vector used | Production us | Candidate us | Wall change |
| --- | --- | ---: | ---: | ---: |
| core_0 | yes | 144.269 | 145.143 | +0.606% |
| core_1 | no | 107.279 | 108.927 | +1.536% |
| core_2 | no | 123.856 | 126.019 | +1.747% |
| core_3 | no | 1163.441 | 1159.445 | -0.343% |
| core_4 | no | 130.131 | 132.987 | +2.195% |
| core_5 | no | 104.322 | 105.343 | +0.978% |
| upstream_bfloat16_s0 | no | 103.947 | 104.814 | +0.834% |
| upstream_bfloat16_s1792 | no | 129.073 | 129.099 | +0.020% |
| upstream_float8_e4m3fn_s0 | no | 108.317 | 109.251 | +0.863% |
| upstream_float8_e4m3fn_s1792 | no | 107.906 | 108.734 | +0.767% |
| zero_visible | no | 29.256 | 29.454 | +0.676% |
| hd_1_32_bfloat16 | no | 109.391 | 110.393 | +0.916% |
| hd_3_64_bfloat16 | no | 103.371 | 105.077 | +1.651% |
| hd_5_96_bfloat16 | no | 103.134 | 104.317 | +1.147% |
| hd_8_256_bfloat16 | no | 105.834 | 106.600 | +0.724% |
| one_row_bfloat16 | yes | 3767.418 | 3796.493 | +0.772% |
| hd_1_32_float8_e4m3fn | no | 107.882 | 109.505 | +1.505% |
| hd_3_64_float8_e4m3fn | no | 130.235 | 131.338 | +0.847% |
| hd_5_96_float8_e4m3fn | no | 108.965 | 110.209 | +1.142% |
| hd_8_256_float8_e4m3fn | no | 131.701 | 132.720 | +0.774% |
| one_row_float8_e4m3fn | yes | 3595.466 | 3607.162 | +0.325% |
| mixed_padded | no | 157.461 | 157.818 | +0.226% |
| tail_chunk_bfloat16 | no | 348.607 | 350.887 | +0.654% |
| tail_chunk_float8_e4m3fn | no | 436.287 | 438.083 | +0.412% |
| single_1_32_bfloat16 | yes | 149.806 | 151.445 | +1.094% |
| single_3_64_bfloat16 | yes | 143.978 | 145.664 | +1.171% |
| single_4_128_bfloat16 | yes | 144.987 | 147.964 | +2.054% |
| single_5_96_bfloat16 | yes | 144.161 | 144.684 | +0.362% |
| single_8_256_bfloat16 | yes | 145.928 | 146.903 | +0.669% |
| single_17_33_bfloat16 | no | 149.988 | 150.550 | +0.375% |
| tail_one_bfloat16 | yes | 348.844 | 350.922 | +0.596% |
| single_1_32_float8_e4m3fn | yes | 144.724 | 146.539 | +1.254% |
| single_3_64_float8_e4m3fn | yes | 181.730 | 182.828 | +0.604% |
| single_4_128_float8_e4m3fn | yes | 179.822 | 181.543 | +0.957% |
| single_5_96_float8_e4m3fn | yes | 151.467 | 153.470 | +1.322% |
| single_8_256_float8_e4m3fn | yes | 153.784 | 154.168 | +0.249% |
| single_17_33_float8_e4m3fn | no | 148.483 | 148.876 | +0.264% |
| tail_one_float8_e4m3fn | yes | 350.381 | 352.051 | +0.477% |
