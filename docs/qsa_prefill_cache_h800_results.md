# QSA score-cache alignment: H800 result, 2026-10-21

Decision: **keep the operator-local cache alignment**, with limited, targeted
performance gains. This repairs the outer-cache/tuner-key mismatch without
changing score arithmetic, JIT specialization, candidate configs or TopK.
It does **not** meet the complete30 per-case performance goal.

## Report and validation

- Result branch: `codex/qsa-h800-results-20261021-021435`.
- Result commit: `1d9ccf756e1dbc7e36113deaa609983cb7e6af51`.
- Tested clean code commit: `1aeb7aa0e8b4a7792192637afc68fb4d3507d340`.
- Production SHA256: `f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c`.
- Full accuracy: **221 passed, 1 skipped**, including BF16/FP8 cache miss/hit
  checks, actual native comparisons and all retained integration/config gates.
- Downloaded and Git-blob verified75 report files. Audited57 source snapshots
  against the tested commit,150 stage rows,600 raw latency samples and450
  telemetry records. Recomputed the exact frozen30 fixtures, geometry,
  per-stage statistics, runtime identities and complete30 acceptance result.
- Native and production both used `/opt/flagtree/triton` (module version3.6.0),
  ptxas13.4.59 and the isolated CUDA13.4 compatibility driver. The same-worker
  previous-production baseline used identical10 rectangular choices,6 compact
  choices, inner JIT functions, host logic and persistent_topk, with independent
  standard LibEntry caches.
- H800 GPU2, UUID `GPU-2d17872f-4327-2beb-8049-604ecf6a3d66`; sampled32–52C,
  SM1965–1980MHz. All450 timing telemetry guards passed and recorded only the
  allowed worker PID. Two initial utilization checks waited10s each; neither
  recorded an external GPU PID. This is sampled guard evidence, not continuous
  proof of exclusive machine ownership.

## Acceptance and grouped results

All numbers below are **equal-weight arithmetic means of per-case speedups**,
not geometric means, aggregate throughput, or ratios of summed latencies.
Native speedup is native median latency / production median latency. Previous
speedup is previous10 standard-entry median latency / new production median
latency, measured within the same worker/toolchain/GPU. Pipeline is measured
as a sequential score+TopK device pipeline; it is not derived by adding stage
medians. TopK uses identical native frozen logits for its separate comparison.
The kernel mode excludes Python key construction and cold autotuning.

| Group | Cases | Score / native | TopK / native | Pipeline / native | Score / previous10 | Pipeline / previous10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.0933 | 0.8990 | 0.9813 | 1.0011 | 1.0006 |
| Observed BF16 | 8 | 2.6104 | 0.9071 | 1.8048 | 1.0056 | 1.0007 |
| FP8 | 10 | 1.0167 | 0.9236 | 0.9614 | 1.0080 | 1.0042 |
| All | 30 | 1.4723 | 0.9094 | 1.1943 | 1.0046 | 1.0018 |

The pipeline arithmetic mean1.1943 exceeds1.1, but16 cases are below1.0 and
12 are below the explicitly proposed near-one floor0.98. The minimum is0.8804
(`fp8_long100k_float8_e4m3fn`). Therefore `measurement_valid=true` and
`target_met=false`. Exit code0 means the correctness/measurement/report run
completed, not that all performance targets passed. TopK remains unchanged;
its mean0.9094 and FP8 long-context minimum0.7864 continue to limit the full
pipeline. No TopK optimization is included in this trial.

The same-worker previous10 comparison is the evidence for this change:
score mean1.0046, pipeline mean1.0018, minimum pipeline0.9942. This does not
establish a broad performance improvement; the useful gains are concentrated
in two tail cases. Comparing to the previous report's native mean1.1946 is
not an isolated A/B for this change, and the previous report's old-production
comparison used eight choices rather than this trial's ten.

## Actual tail selection and keep/revert evidence

All14 multi-chunk cases established two production outer-cache entries for
different row counts, versus one entry in the previous standard LibEntry.
Repeated full chunks kept the same entry. Eight cases changed the compiled
tail configuration. Every first-chunk config and CUBIN matched its previous
baseline; all trace configs/PTX/CUBIN hashes matched the actual warmed caches.
The table reports `(TILE_R, BLOCK_N, K_TILES, num_warps)`.

| Case | Tail rows | Previous tail | New tail | Score / previous10 | Pipeline / previous10 |
| --- | ---: | --- | --- | ---: | ---: |
| perf_n1_q16384_p51200_bfloat16 | 498 | (32, 64, 16, 8) | (32, 64, 4, 8) | 0.9965 | 1.0020 |
| perf_n4_q4096_p102400_bfloat16 | 1261 | (32, 64, 16, 8) | (64, 64, 16, 4) | 1.0000 | 1.0023 |
| observed_chunk_tail64_bfloat16 | 64 | (32, 64, 16, 8) | (32, 64, 4, 8) | 1.0419 | 1.0130 |
| fp8_dominant_long_float8_e4m3fn | 563 | (32, 64, 4, 4) | (32, 64, 4, 8) | 1.0007 | 1.0005 |
| fp8_tail64_float8_e4m3fn | 64 | (32, 64, 4, 8) | (8, 64, 4, 4) | 0.9970 | 1.0003 |
| fp8_tail65_float8_e4m3fn | 65 | (32, 64, 4, 8) | (8, 64, 4, 4) | 0.9880 | 0.9997 |
| fp8_two_balanced_float8_e4m3fn | 73 | (64, 64, 16, 4) | (8, 64, 4, 4) | 1.0856 | 1.0413 |
| fp8_two_skewed_float8_e4m3fn | 80 | (32, 64, 4, 8) | (8, 64, 4, 4) | 1.0083 | 0.9997 |

The strongest result is `fp8_two_balanced`: the73-row tail now uses TILE_R8,
K_TILES4 instead of TILE_R64,K_TILES16 (both4 warps). Its actual static resources
change from226 to54 registers/thread and57344 to22528 shared bytes; both report
zero spills. These are compiled-resource observations, not measured bandwidth
or occupancy. Score A/B paired ratios are1.0843/1.0870; pipeline ratios are
1.0503/1.0325. The improvement has the same direction in both alternating rounds.
`observed_chunk_tail64` also improves in both rounds: score1.0429/1.0409 and
pipeline1.0083/1.0177, with K_TILES16→4 for the64-row tail.

The other changed tails do not all improve. In particular FP8 tail65 score
measures0.9880 while its pipeline is0.9997, and tail64 score0.9970 has pipeline
1.0003. The non-changed single-chunk FP8 budget-exact case measures score0.9864
although its config/CUBIN is identical across sides; cache-key alignment cannot
explain a changed device program there. Small differences must not all be
attributed to the cache policy. The worst pipeline A/B is0.9942 in unchanged
single-chunk BF16 tail16k. No case crosses a2% pipeline regression in this run.

Keep is justified by verified cache/tuner-key consistency, two tail gains with
matching compiled-config evidence and no material measured pipeline regression.
It is not a claim of universal speedup or a completed optimization target.
Remaining risks include more distinct exact-value cache entries and first-call
tuning events; neither host latency nor serving-distribution cache memory has
been benchmarked. Query offsets remain outside the tuner key and JIT
specialization remains unchanged. Runtime values are not new constexprs.

## Two-round stability and duration

| Stage | Median paired-ratio spread | Maximum spread | Cases <=2% |
| --- | ---: | ---: | ---: |
| score | 0.610% | 3.848% | 27/30 |
| topk | 0.032% | 5.048% | 29/30 |
| pipeline | 0.299% | 3.432% | 22/30 |
| score_vs_previous_tuning | 0.707% | 4.073% | 26/30 |
| pipeline_vs_previous_tuning | 0.188% | 2.358% | 29/30 |

Spread is `max(paired ratios)/min(paired ratios)-1` over the two rounds; it is
not a confidence interval. The same-worker pipeline comparison is stable for
29/30 cases within2%, but the native pipeline comparison has8 cases above2%.
Retain2 alternating rounds for ordinary iterations; do not reduce to1 on this
report alone. Marginal results near0.98/1.0 and claims below1% need caution.

Full accuracy took700.75s (11m40.75s). First-to-last benchmark telemetry span
was1261.97s (21m01.97s), including274.84s of stage monitor intervals,56.04s
between stages and931.09s between workers. Monitor intervals include timer and
observer overhead; worker gaps combine startup/imports/JIT/autotune/validation,
gates and teardown and cannot be labeled compile time. The two10s waits are
within the benchmark span. The sum of accuracy duration and benchmark span is
about32m43s, excluding unmeasured setup/boundaries. The prior three-round run
had621.52s accuracy and1364.87s telemetry span; different cold paths prevent
attributing the entire elapsed difference to the rounds change. Test-time
optimization should next target the measured orchestration/cold path, not
assume all remaining time is GPU repetitions.

## Full30 native latency and speedup table

All latencies are microseconds. Each is the median of two measured rounds.
The two select-all cases stay outside these30 acceptance cases.

| Case | Score native us | Score production us | Score x | TopK native us | TopK production us | TopK x | Pipeline native us | Pipeline production us | Pipeline x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 145.424 | 126.872 | 1.1462 | 204.976 | 213.216 | 0.9614 | 344.608 | 335.608 | 1.0268 |
| perf_n1_q8192_p51200_bfloat16 | 519.536 | 491.680 | 1.0567 | 381.992 | 419.512 | 0.9106 | 891.688 | 910.288 | 0.9796 |
| perf_n1_q8192_p102400_bfloat16 | 1044.432 | 940.784 | 1.1102 | 571.424 | 709.184 | 0.8057 | 1566.128 | 1651.568 | 0.9483 |
| perf_n1_q16384_p8192_bfloat16 | 344.312 | 314.280 | 1.0956 | 456.408 | 457.920 | 0.9967 | 780.136 | 760.392 | 1.0260 |
| perf_n1_q16384_p51200_bfloat16 | 1185.080 | 1035.176 | 1.1448 | 794.656 | 892.136 | 0.8907 | 1907.232 | 1934.400 | 0.9860 |
| perf_n1_q16384_p102400_bfloat16 | 2067.320 | 1944.544 | 1.0631 | 1155.368 | 1472.912 | 0.7844 | 3155.944 | 3443.680 | 0.9164 |
| perf_n2_q8192_p8192_bfloat16 | 258.328 | 234.896 | 1.0998 | 399.696 | 414.640 | 0.9640 | 650.224 | 641.712 | 1.0133 |
| perf_n2_q8192_p51200_bfloat16 | 1087.856 | 979.632 | 1.1105 | 763.520 | 837.176 | 0.9120 | 1788.256 | 1814.896 | 0.9853 |
| perf_n2_q8192_p102400_bfloat16 | 1965.992 | 1867.344 | 1.0528 | 1137.928 | 1399.360 | 0.8132 | 3068.568 | 3289.984 | 0.9327 |
| perf_n4_q4096_p8192_bfloat16 | 225.744 | 208.144 | 1.0846 | 389.232 | 384.944 | 1.0111 | 610.392 | 584.640 | 1.0440 |
| perf_n4_q4096_p51200_bfloat16 | 1067.136 | 945.216 | 1.1290 | 752.688 | 814.736 | 0.9238 | 1763.424 | 1762.728 | 1.0004 |
| perf_n4_q4096_p102400_bfloat16 | 1888.992 | 1839.648 | 1.0268 | 1130.552 | 1388.992 | 0.8139 | 2978.448 | 3248.896 | 0.9168 |
| observed_short96_bfloat16 | 51.488 | 8.512 | 6.0489 | 8.368 | 11.392 | 0.7346 | 56.896 | 16.496 | 3.4491 |
| observed_tail16k_bfloat16 | 53.984 | 16.560 | 3.2599 | 17.136 | 18.752 | 0.9138 | 65.408 | 30.080 | 2.1745 |
| observed_tail32k_bfloat16 | 69.952 | 26.656 | 2.6242 | 21.584 | 25.968 | 0.8312 | 86.032 | 45.968 | 1.8716 |
| observed_short_batch3_bfloat16 | 52.592 | 12.704 | 4.1398 | 13.760 | 13.632 | 1.0094 | 62.112 | 22.544 | 2.7551 |
| observed_ragged54_bfloat16 | 78.528 | 73.744 | 1.0649 | 181.472 | 161.816 | 1.1215 | 249.032 | 228.392 | 1.0904 |
| observed_unequal2_bfloat16 | 599.360 | 474.256 | 1.2638 | 524.384 | 635.520 | 0.8251 | 1092.312 | 1094.632 | 0.9979 |
| observed_chunk_boundary_bfloat16 | 496.808 | 447.680 | 1.1097 | 523.680 | 538.176 | 0.9731 | 992.272 | 965.080 | 1.0282 |
| observed_chunk_tail64_bfloat16 | 627.336 | 457.272 | 1.3719 | 534.032 | 629.728 | 0.8480 | 1164.568 | 1086.928 | 1.0714 |
| fp8_first_chunk_float8_e4m3fn | 114.512 | 119.280 | 0.9600 | 351.616 | 345.184 | 1.0186 | 459.848 | 457.520 | 1.0051 |
| fp8_budget_exact_float8_e4m3fn | 290.848 | 314.256 | 0.9255 | 525.256 | 537.840 | 0.9766 | 814.120 | 855.576 | 0.9515 |
| fp8_dominant_long_float8_e4m3fn | 338.016 | 338.456 | 0.9987 | 540.560 | 562.176 | 0.9615 | 873.168 | 900.944 | 0.9692 |
| fp8_tail64_float8_e4m3fn | 385.920 | 355.968 | 1.0841 | 538.728 | 629.360 | 0.8560 | 907.280 | 968.608 | 0.9367 |
| fp8_tail65_float8_e4m3fn | 384.728 | 350.968 | 1.0962 | 533.152 | 629.056 | 0.8475 | 903.224 | 968.528 | 0.9326 |
| fp8_two_balanced_float8_e4m3fn | 394.536 | 384.624 | 1.0258 | 541.936 | 587.552 | 0.9224 | 938.576 | 976.328 | 0.9613 |
| fp8_two_skewed_float8_e4m3fn | 382.624 | 368.616 | 1.0380 | 531.480 | 625.472 | 0.8497 | 900.184 | 984.600 | 0.9143 |
| fp8_short_tail_float8_e4m3fn | 74.448 | 67.872 | 1.0969 | 93.008 | 103.840 | 0.8957 | 163.072 | 165.072 | 0.9879 |
| fp8_compact_float8_e4m3fn | 53.344 | 53.872 | 0.9902 | 181.264 | 161.584 | 1.1218 | 222.968 | 207.360 | 1.0753 |
| fp8_long100k_float8_e4m3fn | 1331.592 | 1399.688 | 0.9513 | 1162.648 | 1478.448 | 0.7864 | 2525.320 | 2868.496 | 0.8804 |

## Full30 same-worker previous10 comparison

| Case | Previous / current score x | Previous / current pipeline x | Score paired rounds | Pipeline paired rounds |
| --- | ---: | ---: | --- | --- |
| perf_n1_q8192_p8192_bfloat16 | 0.9984 | 1.0004 | 0.9948, 1.0020 | 1.0009, 0.9998 |
| perf_n1_q8192_p51200_bfloat16 | 1.0012 | 1.0000 | 1.0000, 1.0025 | 0.9996, 1.0003 |
| perf_n1_q8192_p102400_bfloat16 | 1.0034 | 1.0004 | 1.0075, 0.9992 | 0.9999, 1.0008 |
| perf_n1_q16384_p8192_bfloat16 | 0.9998 | 0.9980 | 0.9980, 1.0016 | 0.9975, 0.9985 |
| perf_n1_q16384_p51200_bfloat16 | 0.9965 | 1.0020 | 1.0098, 0.9833 | 1.0024, 1.0015 |
| perf_n1_q16384_p102400_bfloat16 | 1.0052 | 0.9998 | 1.0092, 1.0011 | 0.9990, 1.0006 |
| perf_n2_q8192_p8192_bfloat16 | 0.9978 | 1.0000 | 0.9997, 0.9960 | 0.9994, 1.0007 |
| perf_n2_q8192_p51200_bfloat16 | 1.0047 | 1.0002 | 1.0060, 1.0034 | 0.9993, 1.0011 |
| perf_n2_q8192_p102400_bfloat16 | 1.0058 | 1.0026 | 1.0122, 0.9994 | 1.0029, 1.0023 |
| perf_n4_q4096_p8192_bfloat16 | 1.0006 | 1.0012 | 1.0019, 0.9992 | 1.0020, 1.0004 |
| perf_n4_q4096_p51200_bfloat16 | 0.9998 | 1.0002 | 1.0049, 0.9947 | 0.9993, 1.0010 |
| perf_n4_q4096_p102400_bfloat16 | 1.0000 | 1.0023 | 1.0043, 0.9958 | 1.0022, 1.0024 |
| observed_short96_bfloat16 | 1.0057 | 0.9990 | 0.9923, 1.0191 | 0.9981, 1.0000 |
| observed_tail16k_bfloat16 | 0.9981 | 0.9942 | 0.9961, 1.0000 | 0.9958, 0.9926 |
| observed_tail32k_bfloat16 | 1.0051 | 0.9997 | 1.0000, 1.0103 | 1.0035, 0.9959 |
| observed_short_batch3_bfloat16 | 1.0000 | 1.0000 | 1.0025, 0.9974 | 1.0014, 0.9986 |
| observed_ragged54_bfloat16 | 0.9965 | 1.0007 | 1.0030, 0.9901 | 1.0046, 0.9968 |
| observed_unequal2_bfloat16 | 0.9951 | 0.9990 | 0.9923, 0.9979 | 0.9974, 1.0006 |
| observed_chunk_boundary_bfloat16 | 1.0021 | 1.0000 | 1.0009, 1.0033 | 1.0058, 0.9942 |
| observed_chunk_tail64_bfloat16 | 1.0419 | 1.0130 | 1.0429, 1.0409 | 1.0083, 1.0177 |
| fp8_first_chunk_float8_e4m3fn | 1.0023 | 1.0005 | 0.9989, 1.0058 | 1.0015, 0.9994 |
| fp8_budget_exact_float8_e4m3fn | 0.9864 | 0.9987 | 0.9849, 0.9879 | 0.9973, 1.0000 |
| fp8_dominant_long_float8_e4m3fn | 1.0007 | 1.0005 | 0.9848, 1.0168 | 1.0017, 0.9993 |
| fp8_tail64_float8_e4m3fn | 0.9970 | 1.0003 | 1.0051, 0.9891 | 0.9993, 1.0013 |
| fp8_tail65_float8_e4m3fn | 0.9880 | 0.9997 | 0.9874, 0.9885 | 1.0001, 0.9994 |
| fp8_two_balanced_float8_e4m3fn | 1.0856 | 1.0413 | 1.0843, 1.0870 | 1.0503, 1.0325 |
| fp8_two_skewed_float8_e4m3fn | 1.0083 | 0.9997 | 1.0020, 1.0147 | 0.9998, 0.9997 |
| fp8_short_tail_float8_e4m3fn | 0.9941 | 1.0025 | 0.9745, 1.0142 | 0.9909, 1.0143 |
| fp8_compact_float8_e4m3fn | 0.9988 | 1.0005 | 1.0048, 0.9929 | 1.0003, 1.0006 |
| fp8_long100k_float8_e4m3fn | 1.0184 | 0.9985 | 1.0210, 1.0157 | 0.9895, 1.0076 |

## Next separate trial

Keep the30 fixtures, TopK implementation and full-pipeline acceptance metric
frozen. The remaining goal is to reduce score latency enough to offset the
existing TopK cost in the upstream long-prefill and FP8 cases. A follow-up
should inspect useful versus empty request/query programs per chunk and actual
Q/K traffic/resource use before selecting one general scheduling/data-reuse
change. Do not add NVIDIA-only instructions or conflate a new tuning space
with this cache-key trial. This report does not implement that next trial.
