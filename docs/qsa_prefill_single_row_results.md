# QSA prefill single-query trial: H800 report 20261015-093828

Decision: correctness passed; performance invalid for acceptance. Keep the
experimental candidate pending an uncontended repeat. The production default
stays on commit `454074da0c079cc649d139489ce11fd3141585b7` operator code;
do not promote, tune or reject the vector path using these timing numbers.

## Provenance and validation

- Code commit: `b5b6e6d8476e62f94278ba75bc0b7471f031c165`.
- Report branch: `codex/qsa-h800-results-20261015-093828`.
- Report commit: `96d3f8c32956d789e83998f39f0f6ad2e8475786`.
- Report path: `results/qsa_select_paged_prefill/h800/stages-20261015-093828/`.
- Run exit code: 0; 161 passed, 1 second-SM90-device skip, 33 native-vLLM cases deselected.
- All 22 new trial cases passed, including all four vector configurations and BF16/FP8 score/full-selection checks.
- All 38 benchmark cases completed; 15 executed at least one vector chunk.
- Verified 122 Git blobs/files (85 unique blob hashes), 14 captured sources against the code commit, and all 3,927 formal stage timing samples/statistics.
- Candidate and baseline share the public entry bytecode, validators, Top-K, fill and fallback dot kernel. Eligible chunks intentionally use a different score kernel.

Both sides run in the same worker with actual imported
`/opt/flagtree/triton/__init__.py` (Triton 3.6.0), Torch 2.13.0+cu129,
`ptxas` 12.8.93 and H800 UUID `198a9496-3d7a-f01d-ee60-190fe0bc5b51`.
Actual compiler, libtriton and ptxas hashes match across all 38 cases.
This is not a cross-compiler comparison or a native-vLLM performance claim.
Neither ncu nor nsys was available.

## Why timing cannot establish a result

Already at `core_0` runtime capture, before this worker constructs its inputs,
GPU 0 shows another compute process, PID 896937,
`...uant/.venv-vlmeval/bin/python`, using 54,182 MiB. The card reports 97%
utilization and 87 C. This same PID is present in every case's initial
nvidia-smi snapshot. No other process was stopped or modified.

All 304 telemetry snapshots (before timing and after seven rounds per case)
report `SW Thermal Slowdown: Active`. Observed temperature is 86–87 C and
SM clock varies from 630 to 1125 MHz. The interference therefore affects
short and long cases, rather than only the long diagnostic case.

Same-process/compiler A/B does not control contention or a changing clock.
Round pairing does not make these data suitable for production routing or
autotune decisions. For example, BF16 single H=4,D=128 has a ratio of wall
medians of 1.03494, while the paired ratio median is 1.00009 and individual
paired ratios span 0.36594–1.19252. Neither an apparent gain nor an apparent
regression is accepted from this run. The compiled configurations are useful
execution evidence but are not established optimal choices under clean load.

## Raw wall observations (not accepted performance results)

Values below are the uploaded medians in microseconds. Change is
`100 * (candidate / baseline - 1)`; positive means a larger observed latency.
No aggregate speedup is accepted.

| Case | Vector used | Production us | Candidate us | Observed change |
| --- | --- | ---: | ---: | ---: |
| core_0 | yes | 161.219 | 177.832 | +10.31% |
| core_1 | no | 110.326 | 115.933 | +5.08% |
| core_2 | no | 442.394 | 447.524 | +1.16% |
| core_3 | no | 4354.211 | 4480.582 | +2.90% |
| core_4 | no | 110.673 | 113.785 | +2.81% |
| core_5 | no | 117.215 | 121.361 | +3.54% |
| upstream_bfloat16_s0 | no | 122.998 | 123.365 | +0.30% |
| upstream_bfloat16_s1792 | no | 121.795 | 128.554 | +5.55% |
| upstream_float8_e4m3fn_s0 | no | 133.603 | 144.464 | +8.13% |
| upstream_float8_e4m3fn_s1792 | no | 125.041 | 124.943 | -0.08% |
| zero_visible | no | 33.221 | 30.047 | -9.55% |
| hd_1_32_bfloat16 | no | 140.609 | 148.487 | +5.60% |
| hd_3_64_bfloat16 | no | 127.784 | 126.931 | -0.67% |
| hd_5_96_bfloat16 | no | 128.261 | 129.636 | +1.07% |
| hd_8_256_bfloat16 | no | 116.697 | 113.332 | -2.88% |
| one_row_bfloat16 | yes | 3794.648 | 3815.498 | +0.55% |
| hd_1_32_float8_e4m3fn | no | 125.965 | 126.278 | +0.25% |
| hd_3_64_float8_e4m3fn | no | 124.836 | 123.014 | -1.46% |
| hd_5_96_float8_e4m3fn | no | 111.532 | 110.336 | -1.07% |
| hd_8_256_float8_e4m3fn | no | 110.093 | 109.006 | -0.99% |
| one_row_float8_e4m3fn | yes | 3716.416 | 3743.357 | +0.72% |
| mixed_padded | no | 159.924 | 160.354 | +0.27% |
| tail_chunk_bfloat16 | no | 378.345 | 373.599 | -1.25% |
| tail_chunk_float8_e4m3fn | no | 357.575 | 354.819 | -0.77% |
| single_1_32_bfloat16 | yes | 159.967 | 169.733 | +6.11% |
| single_3_64_bfloat16 | yes | 162.873 | 161.177 | -1.04% |
| single_4_128_bfloat16 | yes | 165.143 | 159.568 | -3.38% |
| single_5_96_bfloat16 | yes | 160.096 | 160.045 | -0.03% |
| single_8_256_bfloat16 | yes | 162.835 | 163.803 | +0.59% |
| single_17_33_bfloat16 | no | 162.642 | 163.210 | +0.35% |
| tail_one_bfloat16 | yes | 364.139 | 361.537 | -0.71% |
| single_1_32_float8_e4m3fn | yes | 183.552 | 182.074 | -0.81% |
| single_3_64_float8_e4m3fn | yes | 160.117 | 183.148 | +14.38% |
| single_4_128_float8_e4m3fn | yes | 160.341 | 159.819 | -0.33% |
| single_5_96_float8_e4m3fn | yes | 187.929 | 204.569 | +8.85% |
| single_8_256_float8_e4m3fn | yes | 181.973 | 186.375 | +2.42% |
| single_17_33_float8_e4m3fn | no | 159.198 | 161.546 | +1.47% |
| tail_one_float8_e4m3fn | yes | 368.942 | 366.486 | -0.67% |

## Next controlled run

Repeat the same 38-case active set and full correctness gate without changing
kernel code/configs, tolerances or timing methodology. Use an idle GPU with
no thermal slowdown and record the same telemetry throughout. GPU 1 was idle
and 26 C in the returned initial snapshot; this is historical evidence, not
a guarantee it is free now. Check current ownership/processes and temperature
immediately before choosing the card. Do not stop other jobs to obtain a card.

Use the existing single-row runner with `QSA_PREFILL_AB_KIND=single-row`,
`QSA_TRITON_RUNTIME_PATH=/opt/flagtree` and an explicitly selected
`CUDA_VISIBLE_DEVICES`. Both sides must retune for the selected device.
The runner still performs correctness first and automatically uploads reports.
If competing work or thermal slowdown appears during that run, retain the
correctness evidence and exclude the affected timing from acceptance.

The candidate remains benchmark-only. No production integration, further
algorithm change or Top-K optimization is warranted by this report.
