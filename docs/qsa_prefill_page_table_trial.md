# QSA scalar page-table fetch trial, 2026-10-21

Pre-coding plan. Baseline is current production source SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c,
validated by full H800 report07993da. Packed-grid is rejected; its scheduler
is not part of this experiment.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Editable | Private page-table fetch candidate/kernel, current-production A/B plumbing,8 candidate gates and profile nodes, CPU proofs, docs |
| Read-only | All production code, grid/host/chunks, ten rectangular and six compact config choices, math, dtype/layout contract, TopK, native source,30 fixtures, timer/tolerances and GPU guards |
| Hypothesis | Explicitly fetch at most two unique physical page IDs per key tile instead of one ID per key column, then broadcast them across the existing column lanes |
| Scope | Rectangular score only, both BF16 and FP8; unchanged compact path. Compile-time BLOCK_N<=PAGE_SIZE enables two-page fetch; otherwise retain general per-column fetch |
| Evidence | Actual production BLOCK_N64/128 and benchmark page196 imply at most two logical pages per tile; previous empty-grid reduction did not yield stable benefit. No profiler counter establishes page-table bottleneck yet |
| Validation | CPU execute actual load/mask branch over page boundaries, table strides/bounds/invalid IDs; exact unchanged AST outside lookup;8 H800 candidate gates plus40 production iteration nodes; full profile retained |
| Benchmark | Same30 scored-prefill cases, same FlagTree/ptxas13.4/CUDA13.4 compat; kernel mode, two alternating rounds, warmup200ms/rep100ms; score/TopK/pipeline versus native plus score/pipeline versus actual current production |
| Goal | Full-pipeline per-case >=1 or near1 (proposed0.98 floor), equal-weight arithmetic mean >1.1; experimental/iteration reports do not assert formal release validation |
| Fallback | No Torch compute fallback; no new allocation, GPU preprocessing or page-table conversion in candidate. Extra reference normalization remains test-only |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs | Current BF16/FP8, H/D and page sizes, padded/strided Q/cache/table, nonzero query base and chunk boundaries |
| Indexing | Identical logical_page/page_offset per column; physical IDs masked by page-table bounds and num_pages, including invalid IDs |
| Math | Exact key x query FP32 dot, ReLU before real-head reduction and existing visible-column stores |
| Output | Existing int32 selected indices/-1 padding, tie/tolerance rules, caller output and None return |
| Host/TopK | Exact current public/score launcher bytecode, chunk order/workspaces and persistent_topk identity |
| Compiler specificity | Existing Triton scalar load/where and constexpr branch only; no assembly, cache policy or NVIDIA instruction |

## Paths and gates

| Path | Change | Gate |
| --- | --- | --- |
| BLOCK_N<=PAGE_SIZE | Load first_page and optional first_page+1 with scalar address masks; tl.where broadcasts the proper ID to each column | CPU equality for every live lane and no out-of-bounds unmasked table read; H800 page64/196 with BLOCK_N64 and128 |
| BLOCK_N>PAGE_SIZE | Exact original vector fetch | H800 page16 and page64/BLOCK_N128; CPU arbitrary small page sizes |
| Invalid/tail columns | Preserve all original per-column live and physical-page masks | Short visible rows, table last page, negative/too-large physical IDs and page stride2 |
| Cache/autotune | Existing aligned rectangular entry class and original ten keys/configs; compact clone uses original six | Independent caches; source pin and actual returned per-chunk config/PTX/CUBIN snapshots |
| A/B | Isolated current-production output/workspace and real public traces | Same five-stage before/after correctness and native frozen common logits; actual pipeline timings, never sums |
| Keep/revert | Production stays current until measured | Reject unstable sub-percent effects or material active regressions; static source load counts are not memory-transaction/latency measurements |

## Implementation and delivery scope

The explicit private kernel copies the exact current rectangular signature,
cache/tuner decorators, request/query/key grid and all arithmetic/masks outside
the physical-page load. The host score launcher and public API are rebound
from the original bytecode, with independent rectangular/compact caches and
the same TopK object. The new kernel is not wired into the public export.

CPU proof executes the actual new scalar/general load branch with checked fake
memory across arbitrary page sizes, both frozen block widths, partial visible
ranges, padded table strides and invalid page IDs. Replacing just that branch
with the old load recovers exact original kernel AST. The page-table variant
uses the same30 experimental full-pipeline objective and isolated current
production A/B as packed-grid; no packed scheduler or its gate is included.

H800 candidate coverage comprises6 BF16/FP8 score tests at page16/64/196,
each checking BLOCK_N64 and128 across three request/chunk splits, plus2 real
full-entry BF16/FP8 native/FP32 comparisons. Full-entry keeps padded candidate
inputs and nonzero query base, with native's logically identical page table
normalized to contiguous columns outside timing. The default iteration profile
therefore runs48 exact nodes (40 production+8 new candidates); `full` remains
available for complete production validation before integration.

Timing remains two alternating rounds with the same stage timer, thermal/PID
guards and automatic result upload. Actual compiled configs/PTX/CUBIN/static
resources for candidate and current production are captured outside timing.
There is no claim that scalar source loads reduce actual memory transactions,
register use or latency until the H800 report is inspected.

Local validation completed:208 CPU contract/orchestration tests ran (207
passed,1 skipped). Changed-file pre-commit, Python syntax, frozen-file byte
comparisons and CPU-only CLI listing of exactly48 iteration nodes passed.
There is no local Torch/Triton/CUDA installation; H800 compilation, numerical
checks, speedup and keep/revert remain pending. Delivery uses the existing
safe child-bash runner and automatically uploads reports on success/failure.

## H800 result: BF16 promising, do not integrate the unrestricted candidate

Report branch `codex/qsa-h800-results-20261021-070337`, report commit
`9dd9767c2e29937235e765f5b0971f375dbb215a`, clean tested code
`1a805f6a43e6b3795eec8507bf180eee45710db4`.
All48 iteration correctness nodes passed in183.68s (40 production+8 trial).
This is reduced iteration validation, not the full production suite and not
formal production acceptance. All30 benchmark cases completed under the same
FlagTree/ptxas13.4/CUDA13.4 compatibility runtime. Verified75 Git blobs,
70 source snapshots against the tested commit,150 stage rows,600 latency
samples and450 guard telemetry records. Every sampled guard passed with only
permitted worker PIDs; temperatures32–51C and SM1965–1980MHz. These snapshots
do not establish continuous machine exclusivity.

| Group | Cases | Score/native | TopK/native | Pipeline/native | Score/current production | Pipeline/current production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.1122 | 0.8989 | 0.9886 | 1.0204 | 1.0094 |
| Observed BF16 | 8 | 2.6906 | 0.9066 | 1.6688 | 1.0204 | 1.0105 |
| FP8 | 10 | 1.0139 | 0.9236 | 0.9633 | 0.9973 | 1.0002 |
| All | 30 | 1.5003 | 0.9092 | 1.1616 | 1.0127 | 1.0066 |

Values are equal-weight arithmetic means of per-case ratios of two-round
latency medians. Pipeline is timed directly through the unchanged public
function bytecode, not summed from isolated score/TopK measurements. TopK
remains unchanged and is reported, not optimized.

All12 upstream BF16 cases beat current production in both rounds for both
score and pipeline. Their mean score gain is2.04%, pipeline0.94%. Observed
BF16 means are2.04% and1.05%; tail16k, tail32k and short_batch3 show positive
score/pipeline in both rounds. Conversely FP8 budget_exact score is0.9691x
(rounds0.9741/0.9641), first_chunk0.9859x (0.9833/0.9886), dominant_long0.9808x
(0.9763/0.9853), and long100k0.9575x (0.9595/0.9555). FP8 mean pipeline1.0002x
does not offset these score regressions. Long100k pipeline rounds1.0073/0.9828
are inconsistent, so do not claim a stable pipeline regression for that case.
The two compact cases use the unchanged compact path; their small timing
changes are controls, not evidence of this optimization.

Ten cases select different rectangular configurations between trial and
current production. Gains are therefore for the candidate plus its unchanged
allowed autotune set; they do not isolate instruction changes at a fixed
configuration. Some same-configuration cases also show consistent effects:
BF16 tail16k improves, FP8 budget_exact/long100k slow down. Compiled resources
for budget_exact show123->130 registers/thread, long100k226->236; shared bytes
are unchanged. Static global-load sites drop26->20 and38->32 respectively.
These are compiler resource/static-site observations, not measured traffic,
occupancy or a proven causal explanation for the latency regression.

Decision: **retain the BF16 direction for follow-up; do not integrate this
unrestricted BF16/FP8 candidate**. Current production source remains
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
Next suggested experiment is a dtype-gated BF16 path, leaving FP8 on the
current lookup, followed by same-current-production A/B and full validation
before integration. Do not hard-code individual benchmark case names.
A fixed-configuration comparison can distinguish tuning-selection effects.
No such follow-up code is part of this report update.

The30-case native pipeline mean1.1616 exceeds1.1, but11 cases remain below
the proposed0.98 floor (16 below1), minimum0.8641 at FP8 long100k. The full
performance goal is unmet. Avoid treating the prior packed-grid report's
native mean1.1958 as a direct before/after baseline: shortest BF16 native
pipeline stays near56us, but this run's unchanged current-production pipeline
is25.28us versus the prior candidate's15.90us; this run's new trial is25.44us.
The same-run pipeline/current-production ratio0.9937 does not support assigning
that historical shift to the page-table edit. Its cause is unproven; use the
same-worker current-production comparison for the optimization decision.

Accuracy shrank from713.06s in the preceding full trial to183.68s, about74.2%
less. First-to-last benchmark telemetry span was1395.73s (23m15.73s), so the
combined accuracy+benchmark span is26m19.41s, excluding setup/boundaries;
benchmark work itself has not become shorter.

### All30 measured latencies and current-production comparisons

Microseconds; two-round medians. Final two columns use independently measured
same-worker current-production comparisons. Full raw samples and compiled
configuration/resource diagnostics remain in the report branch.

| Case | Score native/trial us | Score x | TopK native/trial us | TopK x | Pipeline native/trial us | Pipeline x | Score/production x | Pipeline/production x |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 146.288 / 118.752 | 1.2319 | 204.768 / 212.928 | 0.9617 | 344.592 / 332.736 | 1.0356 | 1.0580 | 1.0201 |
| perf_n1_q8192_p51200_bfloat16 | 520.920 / 487.712 | 1.0681 | 384.680 / 422.264 | 0.9110 | 892.040 / 904.568 | 0.9862 | 1.0125 | 1.0042 |
| perf_n1_q8192_p102400_bfloat16 | 1030.832 / 928.048 | 1.1108 | 571.392 / 709.000 | 0.8059 | 1563.632 / 1643.368 | 0.9515 | 1.0065 | 1.0024 |
| perf_n1_q16384_p8192_bfloat16 | 339.016 / 309.880 | 1.0940 | 453.936 / 455.376 | 0.9968 | 780.760 / 749.600 | 1.0416 | 1.0133 | 1.0179 |
| perf_n1_q16384_p51200_bfloat16 | 1189.120 / 1015.264 | 1.1712 | 787.768 / 887.472 | 0.8877 | 1906.216 / 1918.816 | 0.9934 | 1.0262 | 1.0076 |
| perf_n1_q16384_p102400_bfloat16 | 2065.584 / 1970.432 | 1.0483 | 1162.504 / 1478.288 | 0.7864 | 3151.224 / 3428.320 | 0.9192 | 1.0189 | 1.0022 |
| perf_n2_q8192_p8192_bfloat16 | 253.664 / 230.384 | 1.1010 | 399.488 / 414.624 | 0.9635 | 647.528 / 636.480 | 1.0174 | 1.0236 | 1.0090 |
| perf_n2_q8192_p51200_bfloat16 | 1085.616 / 965.824 | 1.1240 | 758.160 / 831.504 | 0.9118 | 1783.512 / 1805.112 | 0.9880 | 1.0113 | 1.0139 |
| perf_n2_q8192_p102400_bfloat16 | 1968.736 / 1846.696 | 1.0661 | 1137.664 / 1398.896 | 0.8133 | 3066.744 / 3266.624 | 0.9388 | 1.0142 | 1.0082 |
| perf_n4_q4096_p8192_bfloat16 | 226.240 / 204.992 | 1.1037 | 390.304 / 386.304 | 1.0104 | 611.280 / 583.312 | 1.0479 | 1.0274 | 1.0059 |
| perf_n4_q4096_p51200_bfloat16 | 1050.320 / 927.968 | 1.1318 | 755.488 / 817.264 | 0.9244 | 1759.960 / 1752.216 | 1.0044 | 1.0076 | 1.0045 |
| perf_n4_q4096_p102400_bfloat16 | 1892.832 / 1727.944 | 1.0954 | 1131.080 / 1389.392 | 0.8141 | 2982.240 / 3173.384 | 0.9398 | 1.0249 | 1.0164 |
| observed_short96_bfloat16 | 51.488 / 8.256 | 6.2364 | 8.288 / 11.328 | 0.7316 | 56.352 / 25.328 | 2.2249 | 1.0039 | 0.9937 |
| observed_tail16k_bfloat16 | 54.432 / 15.840 | 3.4364 | 17.056 / 18.672 | 0.9135 | 65.824 / 29.376 | 2.2407 | 1.0382 | 1.0217 |
| observed_tail32k_bfloat16 | 70.256 / 24.800 | 2.8329 | 21.456 / 25.824 | 0.8309 | 84.976 / 43.984 | 1.9320 | 1.0598 | 1.0421 |
| observed_short_batch3_bfloat16 | 52.560 / 12.464 | 4.2169 | 14.000 / 13.872 | 1.0092 | 62.160 / 22.400 | 2.7750 | 1.0391 | 1.0213 |
| observed_ragged54_bfloat16 | 78.376 / 73.968 | 1.0596 | 181.408 / 161.568 | 1.1228 | 248.960 / 228.864 | 1.0878 | 1.0011 | 0.9945 |
| observed_unequal2_bfloat16 | 592.880 / 480.272 | 1.2345 | 524.384 / 635.504 | 0.8251 | 1089.952 / 1099.376 | 0.9914 | 0.9932 | 0.9971 |
| observed_chunk_boundary_bfloat16 | 489.360 / 431.520 | 1.1340 | 526.736 / 541.936 | 0.9720 | 996.896 / 969.736 | 1.0280 | 1.0275 | 1.0110 |
| observed_chunk_tail64_bfloat16 | 627.320 / 456.584 | 1.3739 | 537.248 / 633.616 | 0.8479 | 1162.848 / 1086.048 | 1.0707 | 1.0009 | 1.0021 |
| fp8_first_chunk_float8_e4m3fn | 114.160 / 119.824 | 0.9527 | 351.504 / 345.072 | 1.0186 | 463.600 / 460.768 | 1.0061 | 0.9859 | 0.9958 |
| fp8_budget_exact_float8_e4m3fn | 289.952 / 326.704 | 0.8875 | 522.176 / 534.720 | 0.9765 | 814.752 / 866.240 | 0.9406 | 0.9691 | 0.9897 |
| fp8_dominant_long_float8_e4m3fn | 338.704 / 346.720 | 0.9769 | 536.816 / 558.240 | 0.9616 | 873.208 / 906.528 | 0.9632 | 0.9808 | 0.9944 |
| fp8_tail64_float8_e4m3fn | 385.792 / 345.320 | 1.1172 | 538.800 / 629.488 | 0.8559 | 915.776 / 962.424 | 0.9515 | 1.0236 | 1.0082 |
| fp8_tail65_float8_e4m3fn | 384.976 / 347.120 | 1.1091 | 532.912 / 629.152 | 0.8470 | 904.560 / 956.960 | 0.9452 | 1.0356 | 1.0084 |
| fp8_two_balanced_float8_e4m3fn | 394.496 / 389.504 | 1.0128 | 545.600 / 591.536 | 0.9223 | 938.768 / 971.224 | 0.9666 | 0.9934 | 1.0015 |
| fp8_two_skewed_float8_e4m3fn | 381.424 / 359.344 | 1.0614 | 534.944 / 629.776 | 0.8494 | 918.848 / 988.184 | 0.9298 | 1.0207 | 1.0080 |
| fp8_short_tail_float8_e4m3fn | 74.680 / 67.776 | 1.1019 | 93.040 / 103.792 | 0.8964 | 164.224 / 166.200 | 0.9881 | 1.0090 | 1.0006 |
| fp8_compact_float8_e4m3fn | 53.488 / 53.744 | 0.9952 | 181.424 / 161.664 | 1.1222 | 223.528 / 207.472 | 1.0774 | 0.9976 | 1.0007 |
| fp8_long100k_float8_e4m3fn | 1334.320 / 1442.992 | 0.9247 | 1162.360 / 1478.296 | 0.7863 | 2503.648 / 2897.280 | 0.8641 | 0.9575 | 0.9950 |
