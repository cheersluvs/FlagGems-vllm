# QSA BF16-only page-table fetch trial, 2026-10-21

Pre-coding plan. Follow-up to report9dd9767: all12 upstream BF16 cases
improved in both rounds, while several FP8 score cases regressed. This trial
changes only the dtype scope of the previous private page-table candidate.
Production stays source SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
`optimization.md` and `deep_opt.md` are absent from this checkout; the existing
workflow, measured stage/config/resource evidence and keep/split decision apply.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Op/reference | qsa_select_paged_prefill, existing pinned native vLLM QSA source; no corresponding single torch API |
| Editable | New private BF16 kernel, shared factory/benchmark selection and labeling,8 candidate GPU gates, CPU contracts and docs |
| Read-only | Production source/exports/YAML/TopK, previous unrestricted kernel and GPU gates,30 workloads, reference/tolerances, timers, GPU guards |
| Build | Python source only; local syntax/CPU tests/pre-commit; real JIT compilation on H800 |
| Hypothesis | Preserve measured BF16 scalar-page-ID gains and avoid the same lookup edit for FP8, without case-name dispatch |
| Active set/metric | Same30 scored-prefill cases, native and actual current-production baselines; kernel mode, us, baseline/candidate ratio; pipeline primary, equal-weight arithmetic mean >1.1 with all cases >=1 or near1 (proposed0.98 floor); score/TopK reported |
| Autotune | Original rectangular ten/small compact six configs and YAML groups, exact original keys/strategy/entry class; no new performance parameter or key. Independent caches; actual selected configs recorded, not forced |
| Validation |48 exact iteration nodes (40 production+8 candidate), full profile retained and required before integration; CPU actual-branch/AST proofs |
| Runtime/timeouts | Same FlagTree+ptxas13.4+CUDA13.4 compatibility runtime; two alternating rounds, warmup200ms/rep100ms, initial80C; case1800s, accuracy7200s, cooldown900s/10s |
| Profiler | Existing collected compiled resources/static PTX only; no claim of hardware-counter evidence or memory bottleneck |
| Fallback/integration | No Torch compute fallback, allocations/preprocessing/assembly/backend policy; private experiment only, no public export changes |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs/dtypes | Existing BF16 and FP8 e4m3fn Q/cache contract, H/D parameters, page sizes, padded strides/nonzero query base, chunks and runtime bounds |
| Math/output | FP32 dot accumulation, per-head ReLU then real-head sum, current int32 selection/-1 padding, out buffer and None return |
| Special values/unsupported | Existing masked/invalid-page handling and numeric/tie/tolerance rules, existing validation errors, forward-only; no new NaN/Inf/autograd promise |
| Host/API | Same original launcher/public bytecode, grid, dispatch, workspaces and persistent_topk identity |
| Torch use | Test/reference only; production and private candidate computation remain Triton |

## Paths and gates

| Path | Trigger | Implementation/config | Test/benchmark and risk |
| --- | --- | --- | --- |
| Scalar lookup | not IS_FP8 and BLOCK_N<=PAGE_SIZE | Same previous first/optional-second scalar page-ID loads, broadcast; frozen ten configs | BF16 page64/196 boundaries/stride2; same30 workload and current-production A/B. Gains may include independent autotune selection |
| Vector lookup | IS_FP8 or BLOCK_N>PAGE_SIZE | Exact original vector load and remaining original arithmetic; constexpr gate removes scalar branch for FP8 | FP8 page16/64/196 plus full-entry, BF16 small pages, CPU AST restores exact old function; compiler resources still measured |
| Compact/empty/select-all | Existing host eligibility | Existing compact kernel/early returns, unchanged configuration | Production40 coverage plus unchanged30 set; no benefit attributed to unchanged compact cases |
| Keep/split/revert | Measured result only | No integration until full validation and same-runtime data | Avoid universal FP8 rollout and case-name overfitting; unrestricted prior trial remains reproducible |

## Local validation and delivery

214 CPU contract/orchestration tests ran:213 passed,1 skipped. This includes
inherited actual lookup/mask execution over both dtype gates, page sizes and
strides; exact original AST outside the load; exact previous BF16 branch; host
bytecode/TopK identity, independent aligned entry caches, candidate acceptance,
CLI and48-node selection checks. The full-profile gate tests also passed after
adding BF16-only coverage/missing-coverage cases (14 tests).
All tracked production/benchmark files, original GPU tests and the prior
unrestricted/packed kernel sources were compared byte-for-byte to HEAD.
Changed/new Python syntax, diff whitespace and changed-file pre-commit passed.
No local Torch/Triton/CUDA is available. H800 compilation/numerics, selected
configurations, latency and keep/revert are pending; no speedup is claimed.
The command uses a fresh checkout and automatic report upload in a child shell.

## Completed H800 iteration: keep BF16 direction, full validation still required

Report branch `codex/qsa-h800-results-20261021-075147`, commit
`a65f693062c390968f25f2e4155be5fb77e89554`, tested clean code
`f667c26e7eeeaaa7e20c4da3dfd654ac140eea14`. All48 exact iteration nodes passed
(40 production+8 candidate) in181.57s. This is not a full correctness run and
does not qualify as formal production acceptance. No production integration
is made in this results-only update.

Verified75 report blobs,74 source snapshots against the tested commit,
150 stage rows,600 raw latency samples,450 telemetry records and complete
30-case geometry/order. Pinned native reference SHA256 remains
4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b.
Every case's actual runtime identity matches preflight/accuracy under the same
FlagTree/ptxas13.4/CUDA13.4 compatibility runtime. Before/after numeric checks,
common logits and alternating timing-order checks passed for all five stages.
Sampled guards record only permitted worker PIDs, temperature32–50C and
SM1965–1980MHz; these samples do not prove continuous GPU exclusivity.

| Group | Cases | Score/native | TopK/native | Pipeline/native | Score/current production | Pipeline/current production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.1104 | 0.8985 | 0.9904 | 1.0187 | 1.0102 |
| Observed BF16 | 8 | 2.6624 | 0.9096 | 1.8184 | 1.0207 | 1.0085 |
| FP8 | 10 | 1.0213 | 0.9238 | 0.9624 | 1.0013 | 1.0003 |
| All | 30 | 1.4946 | 0.9099 | 1.2019 | 1.0134 | 1.0064 |

All values are equal-weight arithmetic means of per-case ratios of two-round
latency medians. Pipeline is independently timed through the unchanged public
function bytecode, never inferred by adding score and TopK medians. TopK is
unchanged and remains reported.

### Keep/split decision and remaining uncertainty

All20 BF16 cases average1.0195x score/current-production and1.0095x pipeline;
the19 rectangular BF16 cases average1.0207x and1.0101x. The remaining BF16
case is unchanged compact-path control. Upstream12 means are1.0187x score
and1.0102x pipeline, reproducing the prior trial's approximately2% score/1%
pipeline benefit. This run has10/12 upstream cases positive in both score
rounds and11/12 positive in both pipeline rounds. Do not claim every case
improves consistently: N1/q16384/p102400 score rounds1.0183/0.9832 straddle1;
observed_unequal2 score0.9919x and pipeline0.9964x regress in both rounds, and
observed_short96 pipeline0.9981x is also negative in both rounds.

FP8 mean1.0013x score/current-production and1.0003x pipeline is essentially
flat. Previous budget_exact score0.9691x is now1.0065x, long100k0.9575x is
now1.0138x; their material previous score regression is not reproduced with
the original vector-lookup branch. Long100k pipeline rounds1.0132/0.9878
remain variable; do not call the scalar edit a universal FP8 solution.
The two_skewed FP8 score0.9884x is negative in both rounds0.9871/0.9898;
its tail chunk chooses TILE_R16 in the candidate and8 in current production.
The association is observed, not a proven explanation of the slowdown.

Eight cases choose different configurations overall. Nine of ten FP8 cases
have identical chosen configs in corresponding chunks, registers/shared/spills,
warp counts and static PTX instruction-family/cache-policy counts. The one
exception is two_skewed's tail config. FP8 rectangular PTX/CUBIN hashes still
differ, so this evidence is not a claim of binary identity. The compact control
hashes match. Same-config observations plus AST branch proofs support keeping
FP8 on the original vector path; independent tuning/timing variation remains.

Decision: **keep the BF16-only candidate for full validation before production
integration**. Avoid broadening it to FP8 or adding benchmark-case-name gates.
Production remains source SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
These modest gains do not finish optimization: native pipeline mean1.2019
exceeds1.1, but16 cases are below1 and11 below the proposed0.98 floor;
minimum0.8823 is FP8 long100k. Native score mean1.4946 and unchanged TopK
mean0.9099 must both remain visible. Use current-run production A/B (pipeline
mean1.0064x) to assess this edit, rather than attributing the historical native
mean shift1.1616->1.2019 to this dtype gate. For example short96 trial pipeline
is16.544us in this run versus25.344us in the preceding run; this run's current
production is16.512us. Such cross-run movement does not establish a gate gain.

Correctness took3m1.57s; benchmark first-to-last telemetry span23m12.99s.
Their combined span26m14.56s excludes setup/boundaries. No additional GPU run
is necessary to analyze this report; complete-suite correctness is the next
gate before integration, while performance coverage stays frozen at30.

### All30 measured latencies and current-production comparisons

Microseconds, two-round medians. Final two columns use independent same-worker
production A/B timings. Full raw samples and actual configs/resources are in
the report branch.

| Case | Score native/trial us | Score x | TopK native/trial us | TopK x | Pipeline native/trial us | Pipeline x | Score/production x | Pipeline/production x |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 144.336 / 120.856 | 1.1943 | 206.016 / 214.560 | 0.9602 | 344.968 / 328.784 | 1.0492 | 1.0622 | 1.0212 |
| perf_n1_q8192_p51200_bfloat16 | 519.472 / 489.112 | 1.0621 | 382.128 / 419.672 | 0.9105 | 890.792 / 904.736 | 0.9846 | 1.0135 | 1.0046 |
| perf_n1_q8192_p102400_bfloat16 | 1026.368 / 933.384 | 1.0996 | 571.728 / 709.496 | 0.8058 | 1565.784 / 1632.640 | 0.9591 | 1.0070 | 1.0121 |
| perf_n1_q16384_p8192_bfloat16 | 342.384 / 309.744 | 1.1054 | 456.656 / 458.336 | 0.9963 | 781.312 / 750.832 | 1.0406 | 1.0122 | 1.0229 |
| perf_n1_q16384_p51200_bfloat16 | 1168.480 / 1036.912 | 1.1269 | 793.512 / 892.320 | 0.8893 | 1907.912 / 1916.560 | 0.9955 | 1.0078 | 1.0083 |
| perf_n1_q16384_p102400_bfloat16 | 2101.736 / 1954.048 | 1.0756 | 1163.352 / 1479.072 | 0.7865 | 3160.840 / 3420.144 | 0.9242 | 1.0005 | 1.0081 |
| perf_n2_q8192_p8192_bfloat16 | 257.936 / 229.456 | 1.1241 | 399.696 / 414.544 | 0.9642 | 649.512 / 645.792 | 1.0058 | 1.0227 | 1.0099 |
| perf_n2_q8192_p51200_bfloat16 | 1084.864 / 967.088 | 1.1218 | 763.632 / 837.096 | 0.9122 | 1785.216 / 1807.864 | 0.9875 | 1.0214 | 1.0059 |
| perf_n2_q8192_p102400_bfloat16 | 1972.312 / 1852.768 | 1.0645 | 1130.880 / 1393.984 | 0.8113 | 3067.280 / 3260.320 | 0.9408 | 1.0191 | 1.0110 |
| perf_n4_q4096_p8192_bfloat16 | 226.464 / 202.352 | 1.1192 | 390.384 / 387.496 | 1.0075 | 612.872 / 583.376 | 1.0506 | 1.0203 | 1.0074 |
| perf_n4_q4096_p51200_bfloat16 | 1064.240 / 937.776 | 1.1349 | 757.504 / 819.320 | 0.9246 | 1759.872 / 1751.784 | 1.0046 | 1.0089 | 1.0039 |
| perf_n4_q4096_p102400_bfloat16 | 1891.168 / 1724.224 | 1.0968 | 1129.968 / 1388.656 | 0.8137 | 2982.560 / 3163.944 | 0.9427 | 1.0286 | 1.0069 |
| observed_short96_bfloat16 | 51.472 / 8.448 | 6.0928 | 8.560 / 11.440 | 0.7483 | 56.768 / 16.544 | 3.4313 | 1.0151 | 0.9981 |
| observed_tail16k_bfloat16 | 53.488 / 16.000 | 3.3430 | 17.024 / 18.672 | 0.9117 | 65.600 / 29.328 | 2.2368 | 1.0391 | 1.0204 |
| observed_tail32k_bfloat16 | 69.824 / 24.752 | 2.8209 | 21.456 / 25.856 | 0.8298 | 85.456 / 44.208 | 1.9330 | 1.0695 | 1.0367 |
| observed_short_batch3_bfloat16 | 52.160 / 12.304 | 4.2393 | 13.984 / 13.712 | 1.0198 | 62.368 / 22.640 | 2.7548 | 1.0273 | 1.0034 |
| observed_ragged54_bfloat16 | 78.160 / 73.936 | 1.0571 | 181.440 / 161.760 | 1.1217 | 249.072 / 228.824 | 1.0885 | 0.9961 | 0.9989 |
| observed_unequal2_bfloat16 | 595.840 / 480.376 | 1.2404 | 524.616 / 635.696 | 0.8253 | 1092.264 / 1101.168 | 0.9919 | 0.9919 | 0.9964 |
| observed_chunk_boundary_bfloat16 | 490.936 / 433.408 | 1.1327 | 526.832 / 541.744 | 0.9725 | 995.856 / 956.920 | 1.0407 | 1.0265 | 1.0121 |
| observed_chunk_tail64_bfloat16 | 627.312 / 457.000 | 1.3727 | 533.664 / 629.216 | 0.8481 | 1161.448 / 1085.664 | 1.0698 | 0.9999 | 1.0023 |
| fp8_first_chunk_float8_e4m3fn | 114.400 / 116.632 | 0.9809 | 351.488 / 345.232 | 1.0181 | 464.176 / 460.632 | 1.0077 | 1.0036 | 1.0010 |
| fp8_budget_exact_float8_e4m3fn | 288.584 / 313.728 | 0.9199 | 525.360 / 537.888 | 0.9767 | 813.600 / 851.456 | 0.9555 | 1.0065 | 0.9995 |
| fp8_dominant_long_float8_e4m3fn | 338.304 / 340.600 | 0.9933 | 540.360 / 562.072 | 0.9614 | 874.696 / 903.184 | 0.9685 | 0.9941 | 0.9993 |
| fp8_tail64_float8_e4m3fn | 385.408 / 353.520 | 1.0902 | 538.376 / 629.232 | 0.8556 | 920.032 / 982.584 | 0.9363 | 0.9919 | 0.9993 |
| fp8_tail65_float8_e4m3fn | 386.288 / 351.136 | 1.1001 | 533.128 / 627.400 | 0.8497 | 901.504 / 968.104 | 0.9312 | 1.0041 | 0.9993 |
| fp8_two_balanced_float8_e4m3fn | 394.496 / 385.760 | 1.0226 | 545.424 / 591.232 | 0.9225 | 937.568 / 975.232 | 0.9614 | 1.0009 | 1.0005 |
| fp8_two_skewed_float8_e4m3fn | 382.784 / 363.408 | 1.0533 | 531.728 / 625.744 | 0.8498 | 918.216 / 996.344 | 0.9216 | 0.9884 | 0.9994 |
| fp8_short_tail_float8_e4m3fn | 74.448 / 68.448 | 1.0877 | 93.088 / 103.808 | 0.8967 | 163.432 / 165.904 | 0.9851 | 1.0066 | 1.0039 |
| fp8_compact_float8_e4m3fn | 53.456 / 53.712 | 0.9952 | 181.376 / 161.728 | 1.1215 | 223.128 / 207.736 | 1.0741 | 1.0036 | 1.0000 |
| fp8_long100k_float8_e4m3fn | 1341.544 / 1383.192 | 0.9699 | 1162.608 / 1478.408 | 0.7864 | 2523.312 / 2859.792 | 0.8823 | 1.0138 | 1.0005 |
