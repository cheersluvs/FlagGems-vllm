# QSA query subtiles trial, 2026-10-21

Pre-coding plan. Current production remains pinned to SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
The BF16 page-table candidate remains separate pending full validation. This
experiment isolates query subtiles against actual current production, without
stacking that candidate. workflow.md applies; optimization.md/deep_opt.md are
absent. No GPU profiler measurements are available locally.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Operator/reference | qsa_select_paged_prefill; existing pinned native vLLM, no single torch API |
| Editable | Private query-split kernel/factory, fixed-config comparison helper, benchmark selection/diagnostics, CPU/GPU gates and docs |
| Read-only | Production/exports/YAML, all previous kernels, TopK,30 scored-prefill fixtures, native/reference tolerances and timers/guards |
| Evidence/hypothesis | FP8 TILE_R64/K_TILES16 compiled at226 registers/thread; split FP32 dot results into two query halves with immediate per-half head reduction/store, aiming to reduce simultaneously live accumulators. Register pressure is not a proven bottleneck |
| Autotune | Existing ten rectangular/six compact choices, original keys/strategy/aligned entry class; no new performance parameter. Fixed-config stage inherits each actual baseline chunk's configuration, then independent original autotuning remains for main comparisons |
| Active set/metric | Same30, FlagTree/ptxas13.4/CUDA13.4 compatibility runtime, kernel mode/us/baseline divided by candidate, pipeline primary: equal-weight mean>1.1, per-case>=1 or near1 (proposed0.98 floor); score/TopK always reported |
| Validation/build | Local AST and executable CPU contracts, Python syntax and pre-commit; GPU JIT/numerical checks on H800.40 production iteration nodes plus candidate score/full-entry nodes; full profile before integration |
| Timing | Two alternating rounds, warmup200ms/rep100ms, initial80C, case1800s/accuracy7200s/cooldown900s interval10s; current five stages plus fixed-config score comparison for FP8 rectangular calls eligible to split |
| Profiler | Same collected register/spill/shared/static PTX diagnostics; no hardware-counter/occupancy/traffic claim |
| Fallback/integration | No Torch compute fallback or GPU preprocessing/extra workspace; private experiment, no production export change |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn, parameterized H/D, query/cache/table strides, pages, nonzero query base/chunk offsets and tail rows |
| Math | Each output retains one full-D FP32 dot, ReLU per real head before head sum; no splitting/reordering D accumulation or changing tolerances |
| Output | Existing int32 indices/-1 padding, caller output and None return; invalid-page/live/store masks and unspecified tie ordering |
| Special/unsupported | Existing validation/NaN/Inf/tie treatment, forward-only/no autograd promises, no new dtype or layout scope |
| Host | Original launcher/public bytecode, grids/chunks/workspaces and persistent_topk identity |
| Torch | Test/reference only; none in candidate compute |

## Paths and gates

| Path | Condition | Implementation/config | Coverage/risk |
| --- | --- | --- | --- |
| Split | IS_FP8 and TILE_R>=32 | Two half-query loads outside key loop, keys loaded once per key tile; two sequential full-D dot/ReLU/head-sum/store epilogues; same whole TILE_R grid and ten configs | Direct fixed configs TILE_R32/64, uneven request/chunk tails, padded H/D and stride2; compiler can reschedule, so register savings not guaranteed |
| Original | BF16 or TILE_R<32 | Exact original query load/dot/epilogue, original vector page lookup | BF16 controls and FP8 TILE_R8; original path AST proof |
| Compact/early return | Existing gate | Exact existing kernel/API | Existing production checks; no benefit attributed |
| Fixed-config score | FP8 rectangular baseline has eligible tile | Rebind identical original host bytecode with per-chunk fixed entry launchers, baseline original JIT and candidate split JIT, exact same configs, separate existing workspaces | Untimed resource capture and before/after existing FP32 score checks; compare to independently tuned main stages, never conflate |
| Keep/revert | Full validated same-runtime data | Preserve only stable measured gains without material regressions; full correctness before integration | Two live query operands, extra instructions/layout moves or compiler scheduling may erase gains |

## Implementation and local validation

The split path loads two contiguous half-query operands outside the key loop.
The existing visible vector is reshaped/transposed/split into matching halves,
without extra global visibility reads. Each key tile is fetched once and used
by both full-D dots. Each half immediately applies the original ReLU/head-sum
and visible/invalid-page stores. Source ordering is not a guarantee that the
compiler will keep accumulator lifetimes short; resource/latency data decide.
The original BF16/tiny-tile branches are AST-identical after constexpr path
selection. The compact kernel, original host/public bytecode, config set/keys,
TopK, workspaces, chunking, shape set and numeric tolerances are unchanged.
No Torch compute/copy/cast fallback was introduced (the kernel `.to(tl.int64)`
is the existing Triton address cast).

The supplemental `score_matched` stage uses a separate pair of modules with
the exact original host bytecode and per-chunk fixed launchers. Both sides
inherit actual baseline public-entry configurations and their existing
independent workspace buffers. The baseline uses original JIT and candidate
uses split JIT. Unknown chunks, omitted preparation, config mismatches or new
compilation keys after freezing fail the trial. Fixed kernels are warmed and
resource-snapshotted outside timing, FP32-checked before/after timing, and the
fixed comparison is measured first when applicable. Main five independently
autotuned stages remain available in the same run to avoid another remote
round trip; promotion requires auditing the fixed and tuned results together.
No fixed-config score improvement is asserted before that audit.

Iteration correctness uses46 exact nodes:40 existing production+4 direct score
nodes+2 full-entry nodes. Direct score launches cover existing TILE_R8/32/64
configs, BLOCK_N64/128, K_TILES4/16, BF16 H4/D128, FP8 H4/D128,H1/D32,H3/D96,
three chunks across uneven request boundaries, nonzero query base and padded
page table/cache/query. Full-entry forces existing TILE_R64/K_TILES16/w4 so
the new FP8 path is actually executed, and reuses native/FP32 whole-output
selection gates with logically identical normalized native page table.
`--accuracy-profile full` still selects the complete production test file and
all new candidate nodes; iteration is not formal production validation.

Local result:221 CPU contract/orchestration tests ran,220 passed,1 skipped.
Proofs cover unchanged original path AST and per-half full-D dot/epilogue,
actual query address/mask execution, visibility-half mapping, host/TopK/cache
identity, fixed-config exact-key replay/freezing, report coverage and CLI
profiles. Frozen production/benchmark/previous test and kernel sources were
compared byte-for-byte, new Python syntax and diff checks passed, changed-file
pre-commit passed. No local Torch/Triton/CUDA is available. H800 JIT/numerics,
register/spill/shared/static PTX observations and performance remain pending.
The safe child-shell command uses a fresh checkout and automatically uploads
the result on success/failure. Production integration remains pending measured
benefit and complete validation.

## Completed H800 trial: useful long-FP8 gains, retain as a candidate

Report branch `codex/qsa-h800-results-20261021-085333`, report commit
`5045a0ffd10357ad47d8fae0cf7ff424ac59ccef`, clean tested code
`0ea8d13710a0895abcbfad871e26c643289132a3`. All46 exact iteration nodes
passed (40 production+6 new), including forced split full-entry FP8 and the
H1/D32,H3/D96 direct score gates. Accuracy took173.93s; this is reduced
iteration validation, not the complete production suite/formal acceptance.

Verified75 report Git blobs and80 source snapshots against tested code,
180 TSV rows (159 timed,21 fixed-stage N/A),636 raw latency samples and477
telemetry points. Runtime identity, geometry/order, source pins, independent
caches/workspaces, frozen configs, before/after numeric checks and common
native logits passed. Pinned native SHA remains
4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b.
Same verified FlagTree/ptxas13.4/CUDA13.4 compatibility runtime on both sides.
Every sampled guard passed with only permitted worker PIDs; temperature32–53C,
SM1965–1980MHz. Samples do not prove continuous GPU exclusivity.

### Arithmetic means and unchanged TopK

| Group | Cases | Score/native | TopK/native | Pipeline/native | Score/current production | Pipeline/current production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.0919 | 0.8985 | 0.9790 | 1.0002 | 1.0009 |
| Observed BF16 | 8 | 2.6510 | 0.9077 | 1.6493 | 0.9976 | 0.9969 |
| FP8 | 10 | 1.0766 | 0.9240 | 0.9804 | 1.0591 | 1.0237 |
| All | 30 | 1.5026 | 0.9094 | 1.1582 | 1.0191 | 1.0074 |

Equal-weight arithmetic means of per-case ratios of two-round latency medians.
Pipeline is independently measured, not a sum of isolated stage medians.
TopK is the unchanged persistent_topk, still reported versus native CUDA.
All30 goal remains unmet: pipeline mean1.1582>1.1, but18 cases<1 and10<0.98,
minimum0.8890 at FP8 long100k. Current-production A/B, not historical native
mean shifts, assesses this edit: FP8 score1.0591x/pipeline1.0237x; all30
score1.0191x/pipeline1.0074x. BF16 page-table trial is not stacked here.

### Fixed-config evidence and per-case effects

Each fixed row inherits all corresponding baseline chunk configurations;
register/shared/spill records are taken from the actual direct JIT launches.
Compact/tiny tails remain unchanged. The9 applicable fixed comparisons average
1.0345x, comprising5 improvements and4 regressions. This average does not imply
uniform benefit. All5 improving fixed comparisons are positive in both rounds;
first_chunk/budget_exact/dominant_long regress in both rounds, short_tail is
mixed. The table compares fixed score and separately tuned score/pipeline to
current production, not to native.

| FP8 case | Fixed score x | Fixed paired rounds | Tuned score x | Tuned pipeline x |
| --- | ---: | --- | ---: | ---: |
| fp8_first_chunk_float8_e4m3fn | 0.9956 | 0.9951/0.9962 | 0.9924 | 1.0010 |
| fp8_budget_exact_float8_e4m3fn | 0.9797 | 0.9698/0.9896 | 1.0636 | 1.0251 |
| fp8_dominant_long_float8_e4m3fn | 0.9746 | 0.9793/0.9698 | 1.0550 | 1.0276 |
| fp8_tail64_float8_e4m3fn | 1.0813 | 1.0869/1.0758 | 1.1071 | 1.0406 |
| fp8_tail65_float8_e4m3fn | 1.0910 | 1.0896/1.0923 | 1.1056 | 1.0386 |
| fp8_two_balanced_float8_e4m3fn | 1.0570 | 1.0554/1.0586 | 1.0566 | 1.0241 |
| fp8_two_skewed_float8_e4m3fn | 1.0794 | 1.0818/1.0770 | 1.1414 | 1.0516 |
| fp8_short_tail_float8_e4m3fn | 0.9926 | 0.9791/1.0060 | 0.9949 | 0.9906 |
| fp8_long100k_float8_e4m3fn | 1.0598 | 1.0575/1.0621 | 1.0718 | 1.0347 |

TILE_R64/K_TILES16/w4 main blocks drop226->144 registers/thread (36.3%);
shared57344B stays constant and spill counts are0->0. Balanced fixed score
1.0570x and long100k1.0598x establish useful same-config effects, but do not
prove an occupancy bottleneck or attribute all benefit solely to registers.
TILE_R32/K_TILES4/w4 blocks drop123->80 registers, shared40960->36864B, yet
first_chunk fixed0.9956x and budget_exact0.9797x show that fewer registers
alone are insufficient. Dominant_long fixed0.9746x includes a w4 main block
and a w8 tail; it is a whole-chunk-sequence result, not a per-kernel diagnosis.
The w8 main blocks of tail64/tail65/two_skewed drop110->80 registers; their
fixed comparisons improve1.0813x/1.0910x/1.0794x, with unchanged tiny tails.
No hardware counters or measured occupancy are available.

Independent tuning changes configurations in6 cases,5 FP8. Budget_exact moves
from TILE_R32/K_TILES4/w4 to64/16/w4; dominant_long's main block also moves to
64/16/w4. These tuned gains1.0636x/1.0550x coexist with slower fixed32/w4
results. Tail64/tail65/two_skewed choose K_TILES16 instead of4 on32/w8 main
blocks. Do not label all tuned gains as an isolated reduction in registers.
First_chunk tuned score0.9924x is negative in both rounds; short_tail0.9949x
is negative in both score rounds, while its pipeline0.9906x has mixed rounds.

A control also needs attention before integration: unchanged BF16 short_batch3
pipeline/current-production is0.9747x in both rounds0.9664/0.9833, medians
26.496/27.184us. Chosen score config TILE_R8/BLOCK_N64/K_TILES4/w4, register66,
shared43008B, no spills and static PTX family counts match. Score ratio1.0000x
is identical in both rounds. The split branch is not used there. This does not
establish the cause of the pipeline difference; it must not be silently omitted
or called proven harmless noise. BF16/compact controls cannot earn split gains.

Decision: **retain the query-subtile direction as a private candidate, do not
integrate the unconditional FP8 TILE_R>=32 policy**. Next suggested step is
jointly tuning full-query versus two-subtile execution with the existing tile
choices, preserving a full-query option for regressing regions rather than
hard-coding benchmark case names or a backend-specific register limit. Audit
that mixed-path result, rerun complete correctness before integration, and
recheck the BF16 pipeline control. Production remains SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
This results-only update does not implement the next trial or change TopK.

Benchmark telemetry span1410.34s (23m30.34s), plus accuracy2m53.93s gives
26m24.27s excluding setup/boundaries. No additional run is needed to audit
this report. Native aggregate changes across different runs do not alone
establish an optimization regression/gain; use the current paired A/B.

### All30 native latencies and current-production ratios

Microseconds, two-round medians. Raw samples and matched/tuned compiled
config/resource records remain in the report branch.

| Case | Score native/trial us | Score x | TopK native/trial us | TopK x | Pipeline native/trial us | Pipeline x | Score/production x | Pipeline/production x |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| perf_n1_q8192_p8192_bfloat16 | 144.992 / 125.664 | 1.1538 | 206.080 / 214.544 | 0.9605 | 344.136 / 339.664 | 1.0132 | 0.9989 | 1.0020 |
| perf_n1_q8192_p51200_bfloat16 | 519.168 / 493.112 | 1.0528 | 384.344 / 422.080 | 0.9106 | 892.432 / 909.136 | 0.9816 | 1.0039 | 0.9997 |
| perf_n1_q8192_p102400_bfloat16 | 1031.112 / 940.320 | 1.0966 | 571.392 / 709.248 | 0.8056 | 1562.592 / 1654.200 | 0.9446 | 0.9989 | 0.9999 |
| perf_n1_q16384_p8192_bfloat16 | 339.640 / 314.312 | 1.0806 | 455.120 / 458.128 | 0.9934 | 780.896 / 768.368 | 1.0163 | 0.9914 | 1.0029 |
| perf_n1_q16384_p51200_bfloat16 | 1168.656 / 1051.632 | 1.1113 | 790.216 / 889.680 | 0.8882 | 1909.176 / 1933.936 | 0.9872 | 1.0097 | 0.9993 |
| perf_n1_q16384_p102400_bfloat16 | 2100.960 / 1946.736 | 1.0792 | 1163.760 / 1479.256 | 0.7867 | 3151.392 / 3446.104 | 0.9145 | 1.0007 | 1.0000 |
| perf_n2_q8192_p8192_bfloat16 | 254.176 / 236.192 | 1.0761 | 401.856 / 417.176 | 0.9633 | 649.040 / 650.288 | 0.9981 | 0.9998 | 1.0004 |
| perf_n2_q8192_p51200_bfloat16 | 1087.728 / 981.840 | 1.1078 | 763.592 / 837.048 | 0.9122 | 1785.280 / 1813.984 | 0.9842 | 1.0025 | 1.0012 |
| perf_n2_q8192_p102400_bfloat16 | 1967.352 / 1872.352 | 1.0507 | 1138.256 / 1399.712 | 0.8132 | 3074.336 / 3291.960 | 0.9339 | 0.9990 | 1.0011 |
| perf_n4_q4096_p8192_bfloat16 | 228.624 / 207.776 | 1.1003 | 391.344 / 387.536 | 1.0098 | 611.304 / 585.776 | 1.0436 | 1.0000 | 1.0017 |
| perf_n4_q4096_p51200_bfloat16 | 1067.216 / 944.800 | 1.1296 | 755.344 / 817.248 | 0.9243 | 1756.576 / 1758.592 | 0.9989 | 0.9981 | 0.9996 |
| perf_n4_q4096_p102400_bfloat16 | 1890.816 / 1776.800 | 1.0642 | 1130.912 / 1389.392 | 0.8140 | 2979.984 / 3196.696 | 0.9322 | 0.9991 | 1.0031 |
| observed_short96_bfloat16 | 51.280 / 8.096 | 6.3340 | 8.144 / 11.248 | 0.7240 | 56.368 / 25.904 | 2.1760 | 0.9902 | 0.9944 |
| observed_tail16k_bfloat16 | 53.888 / 16.592 | 3.2478 | 17.232 / 18.816 | 0.9158 | 65.840 / 30.160 | 2.1830 | 0.9990 | 1.0058 |
| observed_tail32k_bfloat16 | 69.984 / 26.592 | 2.6318 | 21.616 / 26.032 | 0.8304 | 86.064 / 46.032 | 1.8697 | 0.9970 | 1.0028 |
| observed_short_batch3_bfloat16 | 52.344 / 12.464 | 4.1996 | 13.744 / 13.424 | 1.0238 | 62.384 / 22.272 | 2.8010 | 1.0000 | 0.9747 |
| observed_ragged54_bfloat16 | 77.984 / 73.856 | 1.0559 | 181.344 / 161.664 | 1.1217 | 248.832 / 229.056 | 1.0863 | 1.0000 | 0.9976 |
| observed_unequal2_bfloat16 | 594.416 / 475.400 | 1.2503 | 524.784 / 635.768 | 0.8254 | 1094.024 / 1096.600 | 0.9977 | 0.9980 | 1.0020 |
| observed_chunk_boundary_bfloat16 | 492.256 / 441.320 | 1.1154 | 527.008 / 541.776 | 0.9727 | 993.072 / 983.664 | 1.0096 | 0.9972 | 0.9977 |
| observed_chunk_tail64_bfloat16 | 627.704 / 457.112 | 1.3732 | 536.832 / 633.144 | 0.8479 | 1163.280 / 1086.080 | 1.0711 | 0.9996 | 1.0004 |
| fp8_first_chunk_float8_e4m3fn | 113.856 / 118.344 | 0.9621 | 351.424 / 345.040 | 1.0185 | 463.744 / 459.680 | 1.0088 | 0.9924 | 1.0010 |
| fp8_budget_exact_float8_e4m3fn | 289.712 / 297.840 | 0.9727 | 525.184 / 537.760 | 0.9766 | 815.456 / 834.520 | 0.9772 | 1.0636 | 1.0251 |
| fp8_dominant_long_float8_e4m3fn | 337.824 / 320.200 | 1.0550 | 540.456 / 562.064 | 0.9616 | 872.856 / 876.904 | 0.9954 | 1.0550 | 1.0276 |
| fp8_tail64_float8_e4m3fn | 386.096 / 320.848 | 1.2034 | 535.720 / 625.584 | 0.8564 | 908.624 / 935.440 | 0.9713 | 1.1071 | 1.0406 |
| fp8_tail65_float8_e4m3fn | 384.864 / 320.392 | 1.2012 | 534.592 / 629.128 | 0.8497 | 900.064 / 933.376 | 0.9643 | 1.1056 | 1.0386 |
| fp8_two_balanced_float8_e4m3fn | 394.936 / 364.888 | 1.0823 | 545.264 / 591.136 | 0.9224 | 938.240 / 953.632 | 0.9839 | 1.0566 | 1.0241 |
| fp8_two_skewed_float8_e4m3fn | 382.848 / 320.752 | 1.1936 | 531.808 / 625.952 | 0.8496 | 919.496 / 948.464 | 0.9695 | 1.1414 | 1.0516 |
| fp8_short_tail_float8_e4m3fn | 74.560 / 69.008 | 1.0805 | 92.256 / 102.928 | 0.8963 | 161.416 / 166.384 | 0.9701 | 0.9949 | 0.9906 |
| fp8_compact_float8_e4m3fn | 53.216 / 53.888 | 0.9875 | 181.392 / 161.632 | 1.1223 | 223.056 / 207.520 | 1.0749 | 1.0024 | 1.0026 |
| fp8_long100k_float8_e4m3fn | 1339.400 / 1303.008 | 1.0279 | 1162.560 / 1478.288 | 0.7864 | 2505.280 / 2818.184 | 0.8890 | 1.0718 | 1.0347 |
