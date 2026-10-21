# QSA packed request/query grid trial, 2026-10-21

Pre-coding plan. Current best is clean production1aeb7aa (documented by
f323659), source SHA256 f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
Report1d9ccf7 validates cache alignment but still has12 pipeline ratios below0.98.

## Project truth

| Field | Choice |
| --- | --- |
| Editable | Private packed-grid candidate/kernel/launcher, benchmark selection/trace and same-current-production pipeline comparison, focused CPU/GPU tests, documentation |
| Read-only | All production sources, LibEntry/LibTuner, tuning YAML, native reference,30 fixtures, timer, numerical tolerances and persistent_topk |
| Build/test | CPU syntax/contracts/actual scheduler coverage and pre-commit locally; full production accuracy plus candidate BF16/FP8 gates on H800 |
| API/reference | Existing QSA API; native commit7a877ae34dcf4a7d46de08eb16db4699e1e0047c; existing dtype/layout/metadata contract |
| Active metric | Same30 scored-prefill cases; kernel-mode score, TopK and sequential pipeline versus native, plus score/pipeline versus current production |
| Goal | Pipeline per-case >=1 or near1 (report floor0.98); equal-weight arithmetic mean >1.1; experimental result explicitly distinct from formal production acceptance |
| Toolchain | Same FlagTree, ptxas13.4 and CUDA13.4 compat for all sides, two alternating rounds, warmup200ms/rep100ms,80C idle limit and existing PID/clock guards |
| Profiling evidence | Prior actual tile choices and chunk lengths; useful/empty query-program counts; new per-chunk compiled config/resources and route trace; no hardware-counter claims |
| Hypothesis | Avoid empty request/query programs in partially covered multi-request chunks, without changing useful per-request query tiles or dot orientation |
| Fallback | Existing compact/select-all/empty/rectangular paths; no Torch compute fallback, no host reads of tensor values, no new workspace or GPU preprocessing launch |

## Semantic contract

| Field | Preserved behavior |
| --- | --- |
| Dtype/layout | BF16/FP8 H/D and nonoverlapping padded/strided Q/cache/table, nonzero query base |
| Score | Exact current rectangular arithmetic from HEADS_PADDED onward: keys x query, FP32 dot result, ReLU per real head, reduction and masks |
| Request boundaries | Binary search monotone clipped-start slots, including empty requests and request/chunk boundaries; no query duplication or omission |
| Output/TopK | Same int32 output and -1 padding, ties/tolerances, visible lengths and persistent_topk object |
| Host | Same validation/workspaces/chunks/order/public function; only private score-launch routing and grid change |
| Unsupported | Same inference-only/resource/device bounds; no new semantic relaxation |

## Paths and evidence

| Path | Trial | Gate |
| --- | --- | --- |
| Existing compact | Unchanged, retains priority | Same compact gate/body/configs; isolated outer caches |
| Sparse rectangular grid | Monotone-slot scheduler with exact current rectangular body/configs | Metadata capacity >=2x chunk rows; conservative TILE_R64 guard must predict fewer query slots; no shape/dtype list |
| Other rectangular | Isolated clone of current production | Same cache alignment, exact JIT/body/configs and key strategy |
| Packed mapping | At most floor(rows/TILE_R)+requests slots, binary search clipped starts | CPU execute actual mapping across all32 fixtures and randomized empty/nonzero-base/boundary cases; on-card full-row FP32 and native checks |
| Cache/autotune | Same10 rectangular and6 compact configs, original keys/default strategy; inherit current local aligned entry | Independent candidate/production entries and actual per-chunk resources; no manual fixed config |
| Comparison | Separate baseline output/workspace and actual public chunk traces | Current production source pinned, score/TopK/pipeline before/after checks, same-worker production pipeline A/B |
| Keep/revert | Production remains unchanged until report | Keep only validated targeted benefit without material active-set regression; reject broad conclusions from sub-percent differences |

The TILE_R64 guard is the maximum frozen rectangular tile, not a hardware
constant. Static config checks enforce that bound. Exact counts for each chosen
TILE_R are recorded outside timing; they count issued query slots, not executed
loads or FLOPs. Existing compact scheduling already pays binary-search overhead;
this trial independently tests that mapping with current key-major arithmetic
on multi-request chunks previously excluded by the compact gate.

## Frozen-route forecast and validation

Forecast below uses report1d9ccf7 chosen TILE_R values. The candidate will
autotune on its actual grid; these are predicted query-slot counts, not a
latency forecast. Existing compact routes retain priority.

| Case | Chunk offset | Rows | Prior TILE_R | Rectangular query slots | Packed query slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| perf_n2_q8192_p51200_bfloat16 | 9039 | 7345 | 32 | 460 | 231 |
| perf_n2_q8192_p102400_bfloat16 | 0 | 4854 | 32 | 304 | 153 |
| perf_n2_q8192_p102400_bfloat16 | 4854 | 4854 | 32 | 304 | 153 |
| perf_n2_q8192_p102400_bfloat16 | 9708 | 4854 | 32 | 304 | 153 |
| perf_n2_q8192_p102400_bfloat16 | 14562 | 1822 | 32 | 114 | 58 |
| perf_n4_q4096_p51200_bfloat16 | 9709 | 6675 | 32 | 512 | 212 |
| perf_n4_q4096_p102400_bfloat16 | 0 | 5041 | 32 | 512 | 161 |
| perf_n4_q4096_p102400_bfloat16 | 5041 | 5041 | 32 | 512 | 161 |
| perf_n4_q4096_p102400_bfloat16 | 10082 | 5041 | 32 | 512 | 161 |
| perf_n4_q4096_p102400_bfloat16 | 15123 | 1261 | 64 | 80 | 23 |
| fp8_two_balanced_float8_e4m3fn | 16256 | 73 | 8 | 20 | 11 |
| fp8_two_skewed_float8_e4m3fn | 0 | 16304 | 32 | 1020 | 511 |
| fp8_two_skewed_float8_e4m3fn | 16304 | 80 | 8 | 20 | 12 |

All production files remain byte-for-byte unchanged; only private tools and
test/benchmark plumbing are edited. CPU tests compare exact kernel signature,
decorators and arithmetic AST, normalize the single extra launcher branch,
execute the actual scalar scheduler prefix for all32 diagnostic fixtures and
100 deterministic random boundary cases at TILE_R4/8/16/32/64, and check
isolated caches/current-source pin, diagnostics, CLI and experimental gates.

The on-card gate runs the full production suite plus8 candidate tests:6
BF16/FP8 full-row score checks across chunk/request/empty-request boundaries,
and2 actual full-entry native/FP32 comparisons with padded inputs and nonzero
query base. The30 benchmark cases retain before/after validation for all five
stages. Baseline/candidate output and score workspace pointers must differ.

Reports explicitly label the objective as experimental, never formal production
acceptance. They include packed route/config/PTX/CUBIN resources, issued/useful/
empty query slots and actual three-dimensional issued-program counts. The
useful-slot metric only accounts for request/query clipping; it does not count
visible-key early exits or executed FLOPs.

Local CPU results and pre-commit are recorded before delivery; H800 compilation,
correctness, performance and keep/revert remain pending.

Local validation completed: `PYTHONPATH=tools python3 -m unittest discover -s
tools -p 'test_qsa*.py'` ran198 tests (197 passed,1 skipped), changed-file
pre-commit passed, Python AST parsing and frozen-file byte comparisons passed.
No local Torch/Triton/CUDA installation is available, so no GPU result is
claimed. H800 delivery uses `--candidate packed-grid --suite scored-prefill
--mode kernel --rounds 2` and the existing automatic report-upload wrapper.

## First H800 attempt and reference-fixture repair

Report `codex/qsa-h800-results-20261021-030451`, commit
`1c675e2366b5b448e4ca1892888913a71f9166a4`, tested clean code629976a.
All15 report blobs and62 source snapshots were verified against the tested
commit and recorded hashes. The run stopped in the added BF16 full-entry test:
227 passed,1 skipped,1 failed in717.52s. This includes221 passing production
tests and all6 new BF16/FP8 packed score-boundary tests. FP8 full-entry did not
run because pytest used `-x`. No benchmark cases were timed.

The full-entry fixture used `padded=True`, so page-table columns have stride2
and alternate with padding -1. Pinned native `_qsa_mqa_paged_prefill_kernel`
accepts `stride_table_req` but reads `request*stride_table_req+logical_page`,
with no column-stride parameter. Logical column1 therefore reads padding -1
instead of the next physical-page index. Its query-base normalization and Q/K
row/head/token strides already match the fixture; column indexing is the
specific unsupported native input here. The candidate supports stride2.
The preexisting full30 report records column stride1 in every benchmark case.

Repair: keep the candidate's padded Q/cache/table and nonzero query base;
provide native with an identical logical page table whose columns are
contiguous, constructed and equality-checked outside timing. Independently
check native selection against full FP32 logits before candidate comparison.
Add baseline/candidate plus query-row labels to pipeline validation errors.
Run the8 small candidate gates before the complete production suite, retaining
all tests and numerical tolerances. The existing CPU accuracy-worker test now
checks that order explicitly. Candidate kernels, production, tuning YAML,
TopK, all30 fixtures and timing remain unchanged. H800 full-entry validation
and performance are still pending after this fixture repair.

## Completed H800 trial: reject for production integration

Report branch `codex/qsa-h800-results-20261021-034927`, report commit
`07993dab60aa2d284545e287d3cda1a117cfd0ae`, tested clean code4e0cde7.
Accuracy:229 passed,1 skipped, including all8 candidate checks. Verified75
report blobs,62 source snapshots against the tested commit,150 stage rows,
600 raw latency samples and450 guard telemetry records. All30 measurements
passed before/after validation under one verified FlagTree toolchain.
Sampled H800 temperature32–48C, SM1965–1980MHz; every timing guard passed
and recorded only permitted worker processes. No continuous-exclusivity claim.

| Group | Cases | Score/native | TopK/native | Pipeline/native | Score/current production | Pipeline/current production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.0960 | 0.8990 | 0.9781 | 0.9977 | 0.9993 |
| Observed BF16 | 8 | 2.6505 | 0.9065 | 1.8140 | 1.0040 | 1.0018 |
| FP8 | 10 | 1.0175 | 0.9238 | 0.9624 | 1.0024 | 1.0008 |
| All | 30 | 1.4844 | 0.9093 | 1.1958 | 1.0010 | 1.0005 |

All group values are equal-weight arithmetic means of per-case speedups.
The actual6 switched cases average0.9957 score/current-production and0.9978
pipeline/current-production; all30 pipeline mean1.0005 does not demonstrate
a stable gain. No switched case has a material consistent improvement.
The N4/q4096/p102400 BF16 case is0.9822 score and0.9845 pipeline; its paired
pipeline rounds are0.9962 and0.9727. Fewer issued slots did not establish a
latency benefit. Extra indexing and compiler resource/layout effects would
require profiling to separate; slot counts alone do not prove the bottleneck.

Decision: **do not integrate packed-grid**. The validated production operator
retains the prior cache-aligned source SHA f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
Private trial files remain available for reproducibility. The native pipeline
mean1.1958 exceeds1.1, but11 cases are below0.98, minimum0.8677, so the full30
goal remains unmet. TopK is unchanged and still reported separately.

Accuracy took713.06s; first-to-last benchmark telemetry span1264.65s. Their
sum is32m57.71s, excluding setup/boundaries. See [accuracy profiles](qsa_prefill_accuracy_profiles.md)
for the new40-production/48-with-candidate iteration selection. This completed
run used the full suite; no rerun is needed merely to adopt the future profile.

### Full30 latency and same-current-production comparison

Latencies below are actual medians of two rounds, in microseconds. Pipeline
is timed directly, not computed from the separate score/TopK medians.

| Case | Score native/ trial us | Score x | TopK native/ trial us | TopK x | Pipeline native/ trial us | Pipeline x | Score/production x | Pipeline/production x |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 144.408 / 126.464 | 1.1419 | 204.672 / 212.928 | 0.9612 | 343.792 / 340.752 | 1.0089 | 1.0010 | 0.9997 |
| perf_n1_q8192_p51200_bfloat16 | 519.648 / 489.200 | 1.0622 | 383.096 / 422.016 | 0.9078 | 894.032 / 909.944 | 0.9825 | 0.9925 | 1.0008 |
| perf_n1_q8192_p102400_bfloat16 | 1047.000 / 937.888 | 1.1163 | 571.584 / 709.312 | 0.8058 | 1562.216 / 1657.976 | 0.9422 | 1.0051 | 1.0003 |
| perf_n1_q16384_p8192_bfloat16 | 344.760 / 314.608 | 1.0958 | 453.760 / 455.120 | 0.9970 | 781.328 / 760.224 | 1.0278 | 1.0006 | 1.0092 |
| perf_n1_q16384_p51200_bfloat16 | 1187.312 / 1036.264 | 1.1458 | 788.608 / 886.224 | 0.8899 | 1914.136 / 1934.456 | 0.9895 | 0.9997 | 1.0014 |
| perf_n1_q16384_p102400_bfloat16 | 2067.248 / 1945.632 | 1.0625 | 1162.528 / 1478.584 | 0.7862 | 3156.784 / 3444.672 | 0.9164 | 0.9994 | 1.0000 |
| perf_n2_q8192_p8192_bfloat16 | 254.304 / 234.656 | 1.0837 | 399.760 / 414.544 | 0.9643 | 650.304 / 649.648 | 1.0010 | 1.0002 | 0.9980 |
| perf_n2_q8192_p51200_bfloat16 | 1086.320 / 974.272 | 1.1150 | 763.376 / 836.968 | 0.9121 | 1785.856 / 1816.136 | 0.9833 | 0.9993 | 1.0006 |
| perf_n2_q8192_p102400_bfloat16 | 1968.952 / 1864.064 | 1.0563 | 1138.912 / 1399.936 | 0.8135 | 3068.216 / 3298.128 | 0.9303 | 0.9945 | 0.9983 |
| perf_n4_q4096_p8192_bfloat16 | 226.488 / 207.520 | 1.0914 | 389.096 / 384.864 | 1.0110 | 610.184 / 587.200 | 1.0391 | 0.9991 | 1.0003 |
| perf_n4_q4096_p51200_bfloat16 | 1067.904 / 945.104 | 1.1299 | 757.648 / 819.560 | 0.9245 | 1759.760 / 1762.640 | 0.9984 | 0.9984 | 0.9989 |
| perf_n4_q4096_p102400_bfloat16 | 1893.472 / 1800.848 | 1.0514 | 1130.648 / 1388.736 | 0.8142 | 2976.448 / 3242.344 | 0.9180 | 0.9822 | 0.9845 |
| observed_short96_bfloat16 | 51.440 / 8.128 | 6.3287 | 8.160 / 11.264 | 0.7244 | 56.304 / 15.904 | 3.5402 | 1.0168 | 0.9980 |
| observed_tail16k_bfloat16 | 53.776 / 16.600 | 3.2395 | 17.280 / 18.864 | 0.9160 | 66.168 / 30.320 | 2.1823 | 0.9990 | 1.0000 |
| observed_tail32k_bfloat16 | 69.984 / 26.400 | 2.6509 | 21.440 / 25.888 | 0.8282 | 85.104 / 45.888 | 1.8546 | 0.9964 | 1.0059 |
| observed_short_batch3_bfloat16 | 52.368 / 12.480 | 4.1962 | 13.888 / 13.664 | 1.0164 | 62.048 / 22.448 | 2.7641 | 1.0077 | 1.0057 |
| observed_ragged54_bfloat16 | 78.096 / 73.536 | 1.0620 | 180.416 / 160.544 | 1.1238 | 249.032 / 228.480 | 1.0900 | 1.0056 | 1.0032 |
| observed_unequal2_bfloat16 | 595.344 / 473.856 | 1.2564 | 525.952 / 639.840 | 0.8220 | 1089.656 / 1097.248 | 0.9931 | 1.0006 | 0.9993 |
| observed_chunk_boundary_bfloat16 | 489.736 / 444.816 | 1.1010 | 523.480 / 538.024 | 0.9730 | 995.544 / 980.656 | 1.0152 | 1.0053 | 1.0001 |
| observed_chunk_tail64_bfloat16 | 625.128 / 456.656 | 1.3689 | 533.552 / 629.176 | 0.8480 | 1163.616 / 1085.136 | 1.0723 | 1.0009 | 1.0018 |
| fp8_first_chunk_float8_e4m3fn | 114.032 / 118.592 | 0.9615 | 351.640 / 345.296 | 1.0184 | 461.072 / 457.800 | 1.0071 | 0.9995 | 0.9938 |
| fp8_budget_exact_float8_e4m3fn | 290.176 / 318.856 | 0.9101 | 525.576 / 538.192 | 0.9766 | 813.200 / 856.056 | 0.9499 | 1.0132 | 1.0010 |
| fp8_dominant_long_float8_e4m3fn | 338.704 / 337.624 | 1.0032 | 540.592 / 562.208 | 0.9616 | 874.080 / 900.032 | 0.9712 | 1.0143 | 1.0003 |
| fp8_tail64_float8_e4m3fn | 385.376 / 351.072 | 1.0977 | 538.744 / 629.264 | 0.8561 | 925.408 / 984.160 | 0.9403 | 1.0124 | 0.9994 |
| fp8_tail65_float8_e4m3fn | 384.128 / 349.928 | 1.0977 | 534.688 / 629.280 | 0.8497 | 902.080 / 966.072 | 0.9338 | 0.9864 | 0.9991 |
| fp8_two_balanced_float8_e4m3fn | 394.632 / 384.984 | 1.0251 | 545.424 / 591.288 | 0.9224 | 937.696 / 973.256 | 0.9635 | 1.0000 | 1.0024 |
| fp8_two_skewed_float8_e4m3fn | 382.400 / 364.024 | 1.0505 | 534.952 / 629.744 | 0.8495 | 907.912 / 979.488 | 0.9269 | 0.9995 | 1.0021 |
| fp8_short_tail_float8_e4m3fn | 73.920 / 67.952 | 1.0878 | 92.880 / 103.744 | 0.8953 | 163.664 / 165.552 | 0.9886 | 1.0016 | 1.0073 |
| fp8_compact_float8_e4m3fn | 53.464 / 53.904 | 0.9918 | 181.408 / 161.616 | 1.1225 | 223.032 / 207.456 | 1.0751 | 0.9950 | 1.0005 |
| fp8_long100k_float8_e4m3fn | 1334.624 / 1405.232 | 0.9498 | 1162.984 / 1478.736 | 0.7865 | 2523.496 / 2908.400 | 0.8677 | 1.0025 | 1.0021 |
