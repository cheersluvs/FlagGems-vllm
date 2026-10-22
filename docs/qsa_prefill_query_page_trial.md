# Combined BF16 page-table and joint-query trial

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Op/reference | Existing qsa_select_paged_prefill and pinned native vLLM source; no single Torch API |
| Editable | New private combined kernel/factory, candidate GPU/CPU tests, runner, this document; candidate routing/accuracy/source archive in benchmark tools |
| Read-only | Canonical production source/exports, all existing private kernels/factories, all YAML groups,30 scored-prefill cases, original timer/guard/reference/tolerances and TopK |
| Build/validation | Local syntax/AST/math/cache/YAML/acceptance/profile tests and pre-commit; full canonical production correctness file plus combined candidate boundary/mode/all-config/full-entry checks on H800 |
| Active set | Frozen30:12 upstream BF16,8 observed BF16,10 FP8; two score-skipping cases remain excluded |
| Metric | Same-compiler --mode kernel us; native baseline/candidate; separately score,TopK,sequential pipeline; same-run current-production score/pipeline and fixed-query diagnostics retained |
| Aggregation | Equal-weight arithmetic mean of30 native pipeline ratios>1.1 and every case>=1/near1 (proposed floor0.98); experimental only, not formal production acceptance |
| Runtime |2 rounds,warmup200ms/rep100ms,initial80C; case1800s,full correctness7200s,cooldown900s/10s; automatic report upload |
| Autotune | Reuse unchanged qsa_select_paged_prefill_query_joint_score17 choices, original keys/strategy/entry/pruner; BF1610 full-query choices,FP8 all17; compact original6. No new config/YAML parameter |
| Profiler | Compiled resources/static instruction sites only; no hardware counters/occupancy proof |
| Fallback | No Torch compute fallback or new production path; no new workspace/asm/cache policy/TopK change |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn,H/D,query/cache/table strides,integer metadata,nonzero query bases,chunks,invalid page/live masks |
| Math | Full-D FP32 dot per head,ReLU before real-head reduction,unchanged score scaling and tolerances; no split of reduction D |
| Outputs | Same int32 selection/-1 padding,caller output identity and None return;tied indices follow unchanged native normalization |
| Promotion/scope | Existing FP32 scores and integer outputs;forward only,unchanged exceptions/unsupported/autograd/special-value coverage |
| Host | Exact canonical score/public function bytecode;independent entries/caches;canonical public export remains canonical |
| Torch | Only existing test/benchmark references perform compute; production never imports this private trial |

## Paths

| Path | Trigger | Implementation/config | Validation and risks |
| --- | --- | --- | --- |
| BF16 page fetch | not IS_FP8 and BLOCK_N<=PAGE_SIZE | Exact previously measured two scalar page-ID loads and broadcast, with original masks | Actual page-branch AST emulation; GPU page16/64/196,nondivisible boundaries,strided tables,all10 choices |
| General page fetch | FP8 or BLOCK_N>PAGE_SIZE | Exact original vector fetch | AST specialization recovers query-joint for FP8; GPU fallback/page-boundary tests |
| Full query | QUERY_PARTS1 or tiny/BF16 | Exact prior joint full-query path | Forced modes and all eligible configs; BF16 recovers prior page-table candidate |
| Split query | FP8,TILE_R>=32,QUERY_PARTS2 | Exact prior two-half full-D query epilogues,one K fetch | Forced modes,H/D and tail coverage,all17 FP8 choices |
| Compact/early returns | Existing dispatch | Exact canonical compact/identity paths | Full canonical suite plus combined compact/full-entry checks |
| Report | Existing query-joint diagnostics | Main5 stages,FP8 fixed mode1/2 comparison,BF16 short_batch3 swap,combined page-gate metadata | Experimental acceptance validates source/config/page metadata; full correctness before integration |

The only kernel change from query-joint is replacing the vector physical-page
load with the exact BF16-only page-table branch. Neither optimization alters the
other dtype's active math. No fixture-name dispatch is introduced. The runner
uses --accuracy-profile full: canonical production tests keep their real public
export, while new candidate tests explicitly bind the private combined module.
This validates the unchanged production contract and the modified candidate
paths; it does not replace production exports or claim production integration.

## Implementation and validation

The private score kernel reuses the existing17-choice joint YAML group. Removing
only its BF16 page-load branch recovers the exact joint score AST. Disabling the
FP8-only split path, removing the mode parameter and restoring the decorator
plumbing recovers the exact previously measured BF16 page-table score AST.
The actual page/mask AST was executed with checked fake memory for both dtypes,
page sizes1/7/16/64/96/196/256,block widths64/128,table strides1/2/3,invalid page
IDs and live/dead columns. No production or YAML file is changed.

The factory preserves canonical host bytecode and TopK identity, with separate
rectangular/compact entries and caches. Candidate selection is explicitly
`query-joint-page`. Reports retain native score/TopK/pipeline and current-production
score/pipeline comparisons, selected-tile full/split diagnostics and the one BF16
buffer swap. The combined metadata records source pin, page gate, config snapshot
and unchanged TopK; acceptance checks reuse the existing joint gates and require
this metadata too. Private results cannot claim formal production integration.

The full GPU profile runs two files: the unchanged complete canonical production
suite (through the real public export), and40 combined candidate tests. The latter
include6 forced query-mode/head-dim/full-entry nodes,8 page-boundary/full-entry
nodes,8 rectangular every-config nodes,6 compact every-config nodes and12 public
candidate-entry nodes across both dtypes,512/1/0MiB workspace budgets and both
rectangular/compact envelopes. All17 FP8 and10 BF16 rectangular choices are
explicitly checked, alongside all6 compact choices, padded strides,int64 metadata,
nonzero query bases,request/chunk tails and untouched invalid-column sentinels.
The full candidate-entry checks also compare canonical output and FP32 reference
without rebinding the real production export. These40 nodes are required to pass;
missing,duplicate,skipped or incomplete candidate families fail the full gate.

This is full canonical contract validation plus targeted full coverage of the
modified candidate paths, not a claim that every canonical parameterized test
was rerun through the private module. A separate iteration profile remains54
nodes for later work (40 canonical +14 candidate), but the delivered runner
explicitly requests full,not iteration. The30 benchmark definitions and timing
parameters remain frozen. No focused control cycles are repeated in this run.

Local validation:98 related CPU tests passed, including7 new composition/gate
checks. New kernel syntax, source pins, unchanged YAML/production/TopK identities,
profile/CLI plumbing,shell syntax,diff checks and changed-file pre-commit passed.
No local Torch/Triton/CUDA is available: H800 JIT,numerics,selected configurations
and all30 performance results remain pending. Keep the candidate private until
those results have been audited; no production integration is part of this change.

Run `bash tools/run_qsa_prefill_query_page_h800.sh --flagtree /opt/flagtree` from
a clean checkout under the existing QSA vLLM0.31 activation and QSA_STOCK_PYTHON.
The verified FlagTree/ptxas13.4/CUDA13.4 compatibility profile is routed explicitly;
no compiler is installed into or replaced in the virtualenv. The existing runner
selects an idle H800 and uploads complete or partial JSON/TXT/TSV reports to a
result branch under a `flagtree134-query-page-*` report directory. Run the shell
entry in a child bash so test/upload failures preserve the interactive container.

## Audited H800 result: 2026-10-22 combined candidate

Report branch `codex/qsa-h800-results-20261022-015356`, report commit
`ea000eb5e1b125a301dc63abeed47ed217c386cd`, tested clean code commit
`b096e86df692da2d1c9a275c309eb5e7f9660c03`.
All75 archived files (45 unique blobs) match GitHub sizes/blob hashes. All96
source snapshots match their recorded SHA256 and tested commit; the native
source matches its pinned reference. All181 stage TSV rows agree with case JSON
and summary. All640 baseline/candidate timing samples,480 telemetry observations,
fixed-mode parameter/resource records and combined metadata were checked. The
acceptance result was recomputed. Original timing mode,AB/BA order,compiled
configuration stability,common TopK input fingerprints and same runtime identity
passed. Every observed GPU/process guard was clean; sampled temperature32–51C
and SM clock1950–1980MHz. Guards do not guarantee identical cache/timing state
between stage pairs or explain the short-batch anomaly below.

Both sides loaded the same verified FlagTree/ptxas13.4/CUDA13.4 compatibility
profile. The actual library was
`/tmp/qsa-cuda134-compat.rl9fpx_n/root/usr/local/cuda-13.4/compat/libcuda.so.615.71.09`,
SHA256 `85a8cc8cf00419ceb295ec62c481d05b4cdbdab10f25c5375d6d24bdbe00b4d9`,
Driver API13040. The random temporary-directory suffix changes the extraction
path,not the verified library version/content. No mixed-compiler baseline was used.

### Correctness scope

261 passed,1 skipped in808.63s. Canonical production contributed221 passes and
one skip; the40 combined-candidate nodes all passed. The skipped test is
`test_noncurrent_cuda_device`,which requires a second SM90 device. That device
guard scenario remains unvalidated under the single visible GPU. Candidate
forced modes,page boundaries,all17 FP8/all10 BF16 rectangular choices,all6 compact
choices and12 budget/envelope full-entry nodes passed the exact coverage gate.
Canonical tests retain the real production public export. These results are full
canonical validation plus the40 targeted candidate nodes; they do not imply that
every canonical parameterized test was replayed through the private candidate.

Benchmark telemetry spans1465.00s (24m25s). Adding the correctness duration gives
37m53.63s,excluding setup and any gaps outside telemetry. The40-node full candidate
suite is appropriate for this combination gate; unchanged-kernel diagnostics can
use the iteration profile later. Production/YAML/exports/TopK remain unchanged.

### Performance

All ratios below are baseline median/candidate median. Groups use equal-weight
arithmetic means of individual case ratios; native sequential pipeline is timed
separately,not obtained by adding isolated stage times. Native TopK is the vLLM
CUDA baseline; candidate TopK remains repository persistent_topk.

| Cases | Count | Score/native | TopK/native | Pipeline/native | Score/current production | Pipeline/current production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.1159 | 0.8987 | 0.9895 | 1.0260 | 1.0104 |
| Observed BF16 | 8 | 2.6943 | 0.9069 | 1.6556 | 1.0233 | 0.9921 |
| FP8 | 10 | 1.0776 | 0.9238 | 0.9814 | 1.0578 | 1.0245 |
| All active | 30 | 1.5241 | 0.9093 | 1.1644 | 1.0358 | 1.0102 |

Native pipeline mean1.1644 exceeds1.1,but16/30 cases remain below1 and10 below
the proposed0.98 floor. Minimum0.8901 is FP8 long100k. The compound objective is
not met. The private result is measurement-valid under the existing report
checks,but is not formal production acceptance. Do not delete the anomalous
short_batch3 result or substitute its swapped-buffer replay in these aggregates.

Same-run current-production comparisons retain the useful gain evidence:
BF16's20 cases average score1.0249/pipeline1.0030; FP8's10 average
score1.0578/pipeline1.0245. Across30,score1.0358/pipeline1.0102. Twenty-three
cases improve in both paired rounds for score/current-production,and23 for
pipeline/current-production. These are combined kernel plus independently tuned
configuration results,not an isolation of the page-load instruction change alone.
Thirteen cases select different non-mode tile geometry from current production.
FP8 first_chunk and short_tail select mode2 again; first_chunk's fixed comparison
is1.0029 with rounds1.0132/0.9926,and short_tail0.9879 with0.9768/0.9991.
Small-mode selection remains sensitive; no fixture-specific dispatch is justified.

The prior joint-only native pipeline mean1.1878 versus this1.1644 must not be
interpreted as an incremental combination regression. For example,short96's
native pipeline is56.992us versus candidate16.240us in that prior run; here it is
56.416us versus26.816us. Its same-run current production is26.336us versus
candidate26.496us (0.9940),so the large cross-run absolute shift is also present
in the unchanged production path. This demonstrates cross-run state sensitivity,
not its cause. Always keep the within-run comparisons and raw timings visible.

### BF16 short_batch3 anomaly recurred

At fixed TR8/BN64/KT4/W4,canonical and combined share the same tile geometry
(the candidate adds QUERY_PARTS1) and TopK object. Score/current-production is
1.0420 in both paired rounds (1.0445/1.0396),but the normal pipeline comparison
is0.8843 and the swapped replay1.0136. Actual samples are:

| Comparison | Baseline us round1/2 | Candidate us round1/2 | Ratio of medians | Paired ratios |
| --- | --- | --- | ---: | --- |
| Native pipeline | 62.144/62.080 | 22.240/22.368 | 2.7848 | 2.7942/2.7754 |
| Current production,normal buffers | 27.328/22.816 | 28.992/27.712 | 0.8843 | 0.9426/0.8233 |
| Current production,swapped buffers | 22.720/22.752 | 22.432/22.432 | 1.0136 | 1.0128/1.0143 |

The normal paired ratios span14.49% (max/min minus1). Candidate latency shifts
from22.304us in the native comparison to28.352us in the normal production
comparison,and22.432us on the swapped replay; these use the same candidate source.
This requires investigation before production integration. Last round's focused
controls exercised joint-only code,not this combined candidate,and cannot clear
its regression. Earlier joint-only controls failing to reproduce the large loss
also do not prove it was noise or identify its cause now.

Canonical/candidate score resources at this tile are66/58 registers,0/0 spills
and43008/43008 shared bytes. Static PTX global-load sites drop27 to13; this is a
static site count,not a measured memory transaction/bandwidth or occupancy claim.
Buffer shape/stride/alignment metadata agree,but addresses/order/cache effects
are not separately established. The normal and swapped stages execute at
different times. No compiler,cache,address or TopK root cause has been proven.

### Cases below the proposed native pipeline floor

| Case | Native pipeline us | Candidate pipeline us | Speedup |
| --- | ---: | ---: | ---: |
| perf_n1_q8192_p102400_bfloat16 | 1563.192 | 1644.192 | 0.9507 |
| perf_n1_q16384_p102400_bfloat16 | 3161.984 | 3413.784 | 0.9262 |
| perf_n2_q8192_p102400_bfloat16 | 3074.720 | 3291.296 | 0.9342 |
| perf_n4_q4096_p102400_bfloat16 | 2980.976 | 3171.888 | 0.9398 |
| fp8_budget_exact_float8_e4m3fn | 813.296 | 834.608 | 0.9745 |
| fp8_tail64_float8_e4m3fn | 922.104 | 947.424 | 0.9733 |
| fp8_tail65_float8_e4m3fn | 901.392 | 932.032 | 0.9671 |
| fp8_two_skewed_float8_e4m3fn | 912.096 | 945.816 | 0.9643 |
| fp8_short_tail_float8_e4m3fn | 162.696 | 166.464 | 0.9774 |
| fp8_long100k_float8_e4m3fn | 2505.208 | 2814.632 | 0.8901 |

The remaining floor failures include all four upstream102400-context cases and
six FP8 cases. TopK is slower than native in many of these shapes,while this
iteration deliberately leaves TopK unchanged. Stage ratios cannot be added to
predict pipeline ratios or used to establish a score-only optimization ceiling.

### Decision and next step

Retain the combined private candidate as a promising direction,but do not
integrate it into the canonical production operator yet. Full correctness passed
within the declared scope; the per-case performance objective and the recurring
short_batch3 comparison still need work.

The next useful diagnostic is a focused run on the combined candidate,extending
the existing balanced buffer/order controls with canonical/canonical and
candidate/candidate self-comparisons on identical buffers. This distinguishes
repeatability of code differences from the measurement context without changing
the existing kernel timer or suppressing original data. Include short96 if
investigating the cross-run absolute pipeline shifts. Reuse the unchanged kernel
and iteration correctness gate instead of rerunning the full suite and30 cases
for this diagnostic. Keep two paired rounds and record all cycles,TopK and
pipeline samples. Do not change TopK or add case-name dispatch while the cause
remains uncertain. Further production integration requires audited representative
and30-case results after that issue is understood.

### All30 native stage measurements

Times are microsecond medians. Each stage cell lists native / candidate / ratio.
The last two columns retain same-run current-production score/pipeline ratios.

| Case | Score us/us/ratio | TopK us/us/ratio | Pipeline us/us/ratio | Score/current production | Pipeline/current production |
| --- | --- | --- | --- | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 144.928 / 120.752 / 1.2002 | 204.688 / 212.976 / 0.9611 | 344.080 / 329.296 / 1.0449 | 1.0821 | 1.0174 |
| perf_n1_q8192_p51200_bfloat16 | 520.024 / 489.056 / 1.0633 | 383.280 / 422.208 / 0.9078 | 892.360 / 904.944 / 0.9861 | 1.0132 | 1.0044 |
| perf_n1_q8192_p102400_bfloat16 | 1047.056 / 924.816 / 1.1322 | 571.456 / 709.264 / 0.8057 | 1563.192 / 1644.192 / 0.9507 | 1.0144 | 1.0044 |
| perf_n1_q16384_p8192_bfloat16 | 344.360 / 309.984 / 1.1109 | 453.728 / 455.128 / 0.9969 | 781.040 / 749.656 / 1.0419 | 1.0154 | 1.0262 |
| perf_n1_q16384_p51200_bfloat16 | 1187.784 / 1034.080 / 1.1486 | 789.632 / 886.592 / 0.8906 | 1907.288 / 1918.688 / 0.9941 | 1.0065 | 1.0068 |
| perf_n1_q16384_p102400_bfloat16 | 2101.248 / 1886.528 / 1.1138 | 1163.256 / 1478.800 / 0.7866 | 3161.984 / 3413.784 / 0.9262 | 1.0339 | 1.0085 |
| perf_n2_q8192_p8192_bfloat16 | 253.560 / 230.480 / 1.1001 | 401.936 / 417.208 / 0.9634 | 646.944 / 635.528 / 1.0180 | 1.0237 | 1.0063 |
| perf_n2_q8192_p51200_bfloat16 | 1086.224 / 958.240 / 1.1336 | 763.600 / 837.024 / 0.9123 | 1792.272 / 1793.528 / 0.9993 | 1.0133 | 1.0112 |
| perf_n2_q8192_p102400_bfloat16 | 1973.304 / 1844.672 / 1.0697 | 1130.984 / 1393.968 / 0.8113 | 3074.720 / 3291.296 / 0.9342 | 1.0112 | 1.0023 |
| perf_n4_q4096_p8192_bfloat16 | 227.232 / 203.040 / 1.1191 | 389.104 / 384.816 / 1.0111 | 609.904 / 589.728 / 1.0342 | 1.0179 | 1.0075 |
| perf_n4_q4096_p51200_bfloat16 | 1031.856 / 934.352 / 1.1044 | 752.304 / 814.304 / 0.9239 | 1759.296 / 1751.280 / 1.0046 | 1.0114 | 1.0058 |
| perf_n4_q4096_p102400_bfloat16 | 1893.872 / 1730.224 / 1.0946 | 1130.720 / 1389.096 / 0.8140 | 2980.976 / 3171.888 / 0.9398 | 1.0688 | 1.0236 |
| observed_short96_bfloat16 | 51.456 / 8.272 / 6.2205 | 8.256 / 11.360 / 0.7268 | 56.416 / 26.816 / 2.1038 | 0.9904 | 0.9940 |
| observed_tail16k_bfloat16 | 53.584 / 15.920 / 3.3658 | 17.248 / 18.848 / 0.9151 | 66.032 / 29.296 / 2.2540 | 1.0515 | 1.0056 |
| observed_tail32k_bfloat16 | 70.104 / 24.896 / 2.8159 | 21.440 / 25.968 / 0.8256 | 85.168 / 44.336 / 1.9210 | 1.0766 | 1.0396 |
| observed_short_batch3_bfloat16 | 52.688 / 12.144 / 4.3386 | 13.936 / 13.664 / 1.0199 | 62.112 / 22.304 / 2.7848 | 1.0420 | 0.8843 |
| observed_ragged54_bfloat16 | 78.192 / 74.160 / 1.0544 | 181.376 / 161.664 / 1.1219 | 249.296 / 228.464 / 1.0912 | 0.9985 | 1.0016 |
| observed_unequal2_bfloat16 | 597.904 / 479.248 / 1.2476 | 527.552 / 639.472 / 0.8250 | 1089.360 / 1101.600 / 0.9889 | 0.9946 | 0.9980 |
| observed_chunk_boundary_bfloat16 | 491.328 / 431.648 / 1.1383 | 526.960 / 541.792 / 0.9726 | 997.688 / 969.760 / 1.0288 | 1.0316 | 1.0123 |
| observed_chunk_tail64_bfloat16 | 627.824 / 457.056 / 1.3736 | 533.776 / 629.320 / 0.8482 | 1163.488 / 1085.240 / 1.0721 | 1.0008 | 1.0012 |
| fp8_first_chunk_float8_e4m3fn | 114.112 / 118.672 / 0.9616 | 351.600 / 345.264 / 1.0184 | 461.456 / 456.688 / 1.0104 | 0.9889 | 1.0024 |
| fp8_budget_exact_float8_e4m3fn | 288.600 / 298.104 / 0.9681 | 525.232 / 537.872 / 0.9765 | 813.296 / 834.608 / 0.9745 | 1.0545 | 1.0252 |
| fp8_dominant_long_float8_e4m3fn | 338.888 / 319.728 / 1.0599 | 540.176 / 561.920 / 0.9613 | 872.056 / 875.560 / 0.9960 | 1.0683 | 1.0320 |
| fp8_tail64_float8_e4m3fn | 386.064 / 319.576 / 1.2081 | 538.680 / 629.328 / 0.8560 | 922.104 / 947.424 / 0.9733 | 1.1037 | 1.0385 |
| fp8_tail65_float8_e4m3fn | 385.816 / 319.376 / 1.2080 | 534.928 / 629.424 / 0.8499 | 901.392 / 932.032 / 0.9671 | 1.1035 | 1.0392 |
| fp8_two_balanced_float8_e4m3fn | 395.392 / 365.384 / 1.0821 | 545.904 / 591.600 / 0.9228 | 938.560 / 951.920 / 0.9860 | 1.0587 | 1.0221 |
| fp8_two_skewed_float8_e4m3fn | 382.840 / 319.744 / 1.1973 | 534.616 / 629.496 / 0.8493 | 912.096 / 945.816 / 0.9643 | 1.1347 | 1.0550 |
| fp8_short_tail_float8_e4m3fn | 74.240 / 68.800 / 1.0791 | 92.320 / 103.088 / 0.8955 | 162.696 / 166.464 / 0.9774 | 0.9960 | 1.0031 |
| fp8_compact_float8_e4m3fn | 53.256 / 53.648 / 0.9927 | 181.272 / 161.504 / 1.1224 | 223.336 / 207.736 / 1.0751 | 0.9967 | 1.0009 |
| fp8_long100k_float8_e4m3fn | 1328.776 / 1303.424 / 1.0195 | 1162.344 / 1478.200 / 0.7863 | 2505.208 / 2814.632 / 0.8901 | 1.0726 | 1.0265 |
