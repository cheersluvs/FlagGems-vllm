# QSA joint full-query/subtile tuning trial, 2026-10-21

Pre-coding plan. Report5045a0f shows useful fixed TILE_R64 and32/w8 gains but
some32/w4 regressions. The next hypothesis is choosing the algorithm jointly
with the existing tiles, preserving every original full-query choice. Current
production source stays SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
No BF16 page-table candidate is stacked. workflow.md applies; optimization.md
and deep_opt.md remain absent in this checkout.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Op/reference | qsa_select_paged_prefill; existing pinned native vLLM, no single torch API |
| Editable | New private query joint kernel/factory/policy, new YAML group only, benchmark selection/fixed-comparison/control diagnostics, CPU/GPU gates, docs |
| Read-only | Production source/exports, all existing YAML groups, old trial kernels/tests, TopK,30 scored-prefill cases, numeric tolerances/reference and timer/guard implementations |
| Hypothesis | QUERY_PARTS=1/2 jointly tuned per existing runtime key chooses full-query for slow regions without case-name or backend-specific register limits |
| Autotune | New qsa_select_paged_prefill_query_joint_score group: original10 with QUERY_PARTS1 plus7 TILE_R>=32 twins with PARTS2; BF16 early prune retains only original10 PARTS1; same original keys/strategy/aligned entry class; no empty pruned sets |
| Validation/build | Local AST/math equivalence, YAML/pruner/cache/host contracts, syntax and pre-commit; GPU direct PARTS1/2 score and forced full-entry tests plus40 production iteration nodes; full suite before integration |
| Active/metric | Same30, same FlagTree/ptxas13.4/CUDA13.4 compatibility runtime; kernel us, baseline/candidate, equal-weight native pipeline arithmetic mean>1.1 and every case>=1 or near1 (proposed0.98 floor); score/TopK/current-production A/B retained |
| Fixed comparison | FP8 chosen joint tile is replayed with PARTS1 versus2, exact same other parameters and separate existing workspaces; labeled distinctly from old production-config comparison |
| BF16 control | Existing short_batch3 receives an additional pipeline buffer-swap replay with the same timer/2 rounds; record actual configurations and buffer layouts, before/after FP32 output checks; diagnose timing/buffer sensitivity without assuming a cause |
| Runtime/timeouts | Two rounds/warmup200ms/rep100ms/initial80C; case1800s/accuracy7200s/cooldown900s interval10s; automatic report upload |
| Profiler | Actual compiled resources/static PTX only, no hardware counters/occupancy or proven bottleneck claim |
| Fallback/integration | No Torch compute/copy/cast fallback, new workspace, custom assembly or production export; private experiment only |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs | BF16/FP8 e4m3fn with existing H/D, pages, query/cache/table strides, nonzero query bases/chunk offsets, invalid-page/live masks |
| Math | PARTS1 recovers original path; PARTS2 recovers previous split source for FP8 eligible tiles. Full-D FP32 dot per output, ReLU before real-head sum, no source split of D accumulation or tolerance change |
| Output | Existing int32 selection/-1 padding, caller output and None return; tie order unspecified |
| Unsupported | Existing validation errors and forward-only/autograd/special-value limitations; no extra dtype/layout promise |
| Host | Original score/public bytecode, grid/chunks/workspaces and persistent_topk identity; controls use existing buffers |
| Torch | Only existing tests/reference use Torch compute |

## Paths and gates

| Path | Trigger | Implementation/config | Coverage and risk |
| --- | --- | --- | --- |
| Full | QUERY_PARTS1, or BF16/tiny tile | Original query/dot/epilogue; original10 geometries remain available | Forced GPU PARTS1, AST original proof, BF16 controls; more candidates do not guarantee tuner always finds faster result |
| Split | FP8,TILE_R>=32,QUERY_PARTS2 | Exact previous two-half queries and immediate epilogues; Q outside key loop, one K fetch | Both modes at fixed32/64 tiles, H1/D32,H3/D96,H4/D128, padded strides/request/chunk tails; existing compiler scheduling risk |
| Compact/early returns | Existing host condition | Exact unchanged compact/identity path | Existing production40; no split gain attributed |
| Fixed diagnostic | Selected FP8 rectangular tile eligible | Compare PARTS1/2 on chosen joint tile, normalize only QUERY_PARTS for exact tile matching | Separate supplemental score_matched label; no silent conflation with old fixed production-config metric |
| Control replay | Existing BF16 short_batch3 only | Swap score/TopK workspace/output ownership between canonical and candidate pipeline functions; same math and timer | Record config equality ignoring PARTS1, buffer shape/stride, paired rounds and FP32 validation; does not establish root cause automatically |
| Keep/revert | Audited H800 result | No production merge before stable gains/full correctness and control review | Extra cold compilation/tuning cost; no promised runtime or performance gain |

## Implementation and local delivery

QUERY_PARTS is a new constexpr performance parameter in the private kernel.
The added YAML group explicitly lists17 choices,10 full-query and7 eligible
split twins. The early pruner keeps all17 for FP8 and exactly the10 original
full-query geometries for BF16. Existing groups are byte-identical: removing
only the inserted experimental YAML block restores the complete original file.
The rectangular factory preserves the pruner when isolating its aligned entry;
compact keeps its original six choices/JIT/entry behavior. Runtime trace checks
reject ineligible selected modes. Mode search can still be noisy; the existence
of full-query choices is not a promise the tuner always selects a faster one.

Supplemental score_matched now compares QUERY_PARTS1 versus2 using each chosen
joint tile. Both sides use the same joint JIT and exact same other parameters;
small/compact tails stay original. It is explicitly labeled joint_tile_full_query
and query_parts_pair[1,2], distinct from the prior trial's original-production
configuration comparison. Main native and true current-production score/TopK/
pipeline comparisons retain their original meaning. Fixed launchers reject
unknown/unprepared keys, check every non-mode parameter, and freeze compiled
records outside timing. Old fixed-config behavior remains the default for old
trial selections.

The one BF16 control uses canonical code with candidate buffers versus joint
code with canonical buffers, including corresponding TopK scratch/output. Its
pipeline_buffer_swap row has before/after full FP32 selection checks using the
correct swapped output owners, plus the same paired timer/guard. Reports contain
workspace/output shapes, strides, alignment, actual per-chunk configs, equality
of all non-mode config fields and identical TopK binding. Metadata reads do not
hash uninitialized scratch. Config equality or a change on swapping does not
alone establish a causal explanation for the earlier0.9747x observation.
The supplemental control is required by the experimental report gate.

The46 iteration nodes are40 original production plus6 new candidate nodes.
Each FP8 direct node explicitly executes both modes at TILE_R8/32/64 across
request/chunk tails, query_base97, padded strides and H1/D32,H3/D96,H4/D128.
Both full-entry FP8 modes are forced through the existing64/16/w4 choice and
native/FP32 checks; BF16 checks mode1. The full profile keeps all production
tests before integration. No numerical tolerance, workload or TopK was changed.

Local validation:228 CPU contract/orchestration tests ran,227 passed,1 skipped.
AST proof recovers the exact prior kernel after only removing the new parameter,
mode predicate and tuner plumbing. YAML/config pruning, actual host/TopK/cache
identity, matched-mode records/freezing, control buffer ownership and selection
validation output mapping, acceptance gates and CLI profiles passed. Production
source SHA, all old YAML groups, frozen benchmark/previous GPU tests/kernels,
new Python syntax, diff whitespace and changed-file pre-commit passed.
No local Torch/Triton/CUDA is available. H800 JIT/numerics, selected modes,
performance, additional cold tuning time and the BF16 control remain pending.
Private candidate only; no production exports or TopK implementation changes.
The child-shell runner uses a fresh checkout,2 rounds and automatic report upload.

## Audited H800 result: 2026-10-21 joint query mode

Report branch `codex/qsa-h800-results-20261021-102057`, report commit
`0ab48375ecb1f86a5d5d170a2941614e17e80bba`, tested clean code commit
`6505f47e0c157bcaf2aee19045984973b88402fc`.
The report's75 archived files (45 unique blobs) were verified against GitHub
sizes/blob hashes;
all86 source snapshots match their recorded SHA256 and tested commit (native
source matches the pinned reference SHA256). All181 TSV stage rows agree with
per-case JSON/summary, including21 inapplicable fixed comparisons. All640 timed
baseline/candidate samples and480 telemetry observations were checked, and the
acceptance result was independently recomputed. Cases, production SHA, same
compiler/runtime identity, stage baseline labels, pair order, config selections,
matched non-mode parameters, compiled resource identities, source bindings and
before/after output checks passed. Every observed process guard was clean;
observed GPU temperature32–52C and SM clocks1965–1980MHz. These guards validate
observed conditions; they do not rule out timing/cache sensitivity between stages.

All46 iteration correctness nodes passed in177.76s, including forced FP8 modes1
and2 and the existing40 production iteration nodes. Full production correctness
was not run; this remains an experimental result. Benchmark telemetry spans
1450.89s (24m10.89s); correctness plus that span is27m8.65s, excluding setup and
any gaps outside telemetry. Production source/exports and TopK remain unchanged.
The BF16 page-table candidate is not stacked in this experiment.

Speedup is baseline median divided by candidate median; groups use equal-weight
arithmetic means of individual case ratios. Native score, TopK and sequential
pipeline are separately timed, not sums or averages of stage speedups.

| Cases | Count | Score/native | TopK/native | Pipeline/native | Score/current production | Pipeline/current production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream BF16 | 12 | 1.0855 | 0.8979 | 0.9750 | 1.0057 | 1.0011 |
| Observed BF16 | 8 | 2.6556 | 0.9074 | 1.7613 | 1.0006 | 0.9879 |
| FP8 | 10 | 1.0796 | 0.9231 | 0.9844 | 1.0589 | 1.0225 |
| All active | 30 | 1.5023 | 0.9088 | 1.1878 | 1.0221 | 1.0047 |

The native pipeline mean1.1878 exceeds1.1, but17 cases are below1 and10 below
the proposed0.98 floor. Minimum0.8894 is FP8 long100k. The full objective remains
unmet. Keep the BF16 short-batch anomaly in all official aggregates above; do
not silently remove it or replace its main result with the supplemental control.
The earlier split trial's native mean1.1582 and this1.1878 are separate runs;
that difference alone is not evidence of an incremental joint-tuning gain.
The same-run FP8 current-production comparisons are the useful gain evidence:
score1.0589, pipeline1.0225. They are close to the prior split trial's1.0591 and
1.0237, so this run does not establish an extra aggregate gain from joint search.

### Chosen query modes and fixed-tile counterfactual

Every BF16 rectangular selection used mode1. Eight FP8 cases used mode2 on at
least one chunk; short_tail used mode1 and compact used its unchanged compact
kernel. All tiny FP8 tails stayed mode1. Six FP8 cases and one BF16 upstream
case selected different tile geometry from their independently tuned current
production counterparts; compare full parameter traces rather than attributing
all tuned A/B differences to mode alone.

Below, full/split compares mode1 versus2 on the chosen joint tile, with all
other constants identical. It is not the previous experiment's fixed original
production tile comparison. Main/tail modes follow chunk order. TR/KT/W mean
TILE_R/K_TILES/num_warps, with BLOCK_N64, stages2 and ctas1 throughout these
rectangular selections. Fixed comparison includes unchanged tiny tails.

| FP8 case | Joint chunks: TR/KT/W/mode | Fixed full/split score | Score/current production | Pipeline/current production |
| --- | --- | ---: | ---: | ---: |
| first_chunk | 32/4/4/2 | 0.9869 | 1.0058 | 0.9948 |
| budget_exact | 64/16/4/2 | 1.0844 | 1.0466 | 1.0261 |
| dominant_long | 64/16/4/2; 32/4/4/2 | 1.0647 | 1.0787 | 1.0141 |
| tail64 | 32/16/8/2; 8/4/4/1 | 1.1738 | 1.0921 | 1.0370 |
| tail65 | 32/16/8/2; 8/4/4/1 | 1.1796 | 1.0945 | 1.0387 |
| two_balanced | 64/16/4/2; 8/4/4/1 | 1.0534 | 1.0591 | 1.0240 |
| two_skewed | 32/16/8/2; 8/4/4/1 | 1.1783 | 1.1351 | 1.0505 |
| short_tail | 32/4/4/1 | 0.9778 | 1.0234 | 1.0074 |
| compact | compact | N/A | 0.9982 | 1.0002 |
| long100k | 64/16/4/2; 64/16/4/2; 64/16/4/2; 64/16/4/2 | 1.0726 | 1.0555 | 1.0327 |

Nine eligible fixed comparisons average1.0857; seven improve in both rounds,
while first_chunk0.9869 and short_tail0.9778 regress in both. Joint tuning
correctly selects mode1 for short_tail, but selects mode2 for first_chunk despite
its later fixed-tile loss. Its fixed paired ratios are0.9951/0.9787; tuned score
versus current production is1.0058 with paired0.9944/1.0174, and pipeline0.9948.
This is a remaining mode-selection uncertainty; having mode1 in the search
space is not a no-regression guarantee. Do not add a case-name dispatch rule.

Unlike the prior original-production tile comparison, budget_exact and
dominant_long now compare at their selected64/16/W4 tile on the main chunk.
Their fixed1.0844/1.0647 confirms benefit at those choices; it does not contradict
the prior losses on different32/4/W4 tiles. Tail64/tail65/two_skewed fixed
1.1738/1.1796/1.1783 uses selected32/16/W8. Actual register changes are226 to144
for64/16/W4, and110 to80 for32/16/W8; spills remain0 and shared memory is
57344 to57344 or40960 to36864 bytes respectively. Resource reductions support
the hypothesis but do not prove occupancy or a hardware-counter bottleneck.

### BF16 short_batch3 control remains unresolved

Both normal score kernels select8/4/W4, mode1 on the joint side,66 registers,
0 spills and43008 bytes shared memory. Tile parameters, static instruction-site
counts, buffer shape/stride/alignment metadata and TopK binding agree; PTX and
CUBIN hashes differ. The score/current-production ratio1.0013 does not show a
stable score regression. Pipeline measurements are:

| Control | Canonical us: round1/round2 | Joint us: round1/round2 | Ratio of medians | Paired ratios |
| --- | --- | --- | ---: | --- |
| Original buffer ownership | 26.720/22.656 | 26.736/27.584 | 0.9090 | 0.9994/0.8213 |
| Swapped buffer ownership | 22.720/22.752 | 22.624/22.656 | 1.0042 | 1.0042/1.0042 |

The normal paired ratios span21.68% (max/min minus1); original canonical
round2 becomes22.656us while joint remains27.584us. Swapped launches are both
about22.7us and have stable paired ratios. This shows the observed pipeline
regression is sensitive to the buffer/timing experiment; it does not establish
buffer addresses, caches, timing order or source code as the cause. Normal and
swapped stages run at different times, so swapping is not an isolated causal
intervention. The earlier0.9747 normal result must likewise not be dismissed
as noise. Do not mark the anomaly resolved or claim a durable9.1% code regression.

### Decision and next validation

Retain the query-subtile direction and this private joint-mode experiment, but
do not integrate it into production yet. First perform focused, repeated
full/split comparisons for first_chunk (same tile, alternating order, same
buffer conditions), plus normal/swapped BF16 short_batch3 replay with controlled
stage order and repeated buffer ownership. Keep existing timing/guard/numerical
contracts and report all samples. Resolve mode selection from runtime geometry
and measured behavior rather than case names or backend register thresholds.
A focused diagnostic need not rerun30 cases. Keep two rounds for broad iteration;
one round would hide the BF16 discrepancy observed here.

Only after those issues are understood should the retained BF16 page-table
candidate and query mode be combined privately, followed by full production
correctness and all30 same-compiler kernel benchmarks. No TopK optimization is
introduced; continue publishing its measurements. The30-case floor remains
unmet, especially upstream long-context BF16 and FP8 long100k. Their relatively
slow TopK measurements constrain pipeline gain, but neither isolated stage ratios
nor their arithmetic average predicts the separately measured pipeline.


### All 30 native stage measurements

Times below are medians in microseconds; each cell lists native / candidate /
speedup. Pipeline was timed separately and includes unchanged TopK.

| Case | Score: us / us / ratio | TopK: us / us / ratio | Pipeline: us / us / ratio |
| --- | --- | --- | --- |
| perf_n1_q8192_p8192_bfloat16 | 143.840 / 127.768 / 1.1258 | 205.952 / 214.464 / 0.9603 | 344.520 / 340.560 / 1.0116 |
| perf_n1_q8192_p51200_bfloat16 | 517.512 / 497.024 / 1.0412 | 384.304 / 422.728 / 0.9091 | 890.496 / 914.616 / 0.9736 |
| perf_n1_q8192_p102400_bfloat16 | 1029.200 / 944.784 / 1.0893 | 571.160 / 709.072 / 0.8055 | 1558.128 / 1664.800 / 0.9359 |
| perf_n1_q16384_p8192_bfloat16 | 342.448 / 317.208 / 1.0796 | 452.960 / 454.864 / 0.9958 | 778.192 / 772.240 / 1.0077 |
| perf_n1_q16384_p51200_bfloat16 | 1182.800 / 1042.440 / 1.1346 | 791.256 / 892.368 / 0.8867 | 1904.720 / 1939.848 / 0.9819 |
| perf_n1_q16384_p102400_bfloat16 | 2094.384 / 1961.632 / 1.0677 | 1154.312 / 1470.664 / 0.7849 | 3153.864 / 3447.504 / 0.9148 |
| perf_n2_q8192_p8192_bfloat16 | 258.464 / 236.208 / 1.0942 | 398.880 / 414.224 / 0.9630 | 647.552 / 641.440 / 1.0095 |
| perf_n2_q8192_p51200_bfloat16 | 1084.152 / 985.152 / 1.1005 | 763.280 / 836.496 / 0.9125 | 1780.960 / 1817.752 / 0.9798 |
| perf_n2_q8192_p102400_bfloat16 | 1966.344 / 1886.720 / 1.0422 | 1130.392 / 1394.464 / 0.8106 | 3063.096 / 3310.880 / 0.9252 |
| perf_n4_q4096_p8192_bfloat16 | 224.480 / 209.608 / 1.0710 | 388.416 / 384.720 / 1.0096 | 608.600 / 589.320 / 1.0327 |
| perf_n4_q4096_p51200_bfloat16 | 1050.208 / 943.544 / 1.1130 | 752.592 / 813.968 / 0.9246 | 1761.896 / 1767.808 / 0.9967 |
| perf_n4_q4096_p102400_bfloat16 | 1888.016 / 1768.816 / 1.0674 | 1121.024 / 1381.104 / 0.8117 | 2973.712 / 3194.712 / 0.9308 |
| observed_short96_bfloat16 | 51.600 / 8.160 / 6.3235 | 8.128 / 11.200 / 0.7257 | 56.992 / 16.240 / 3.5094 |
| observed_tail16k_bfloat16 | 53.664 / 16.240 / 3.3044 | 17.024 / 18.496 / 0.9204 | 65.456 / 29.680 / 2.2054 |
| observed_tail32k_bfloat16 | 69.808 / 26.160 / 2.6685 | 21.280 / 25.616 / 0.8307 | 85.296 / 45.408 / 1.8784 |
| observed_short_batch3_bfloat16 | 52.232 / 12.528 / 4.1692 | 13.856 / 13.648 / 1.0152 | 62.384 / 26.864 / 2.3222 |
| observed_ragged54_bfloat16 | 78.128 / 74.368 / 1.0506 | 181.280 / 161.496 / 1.1225 | 250.000 / 228.240 / 1.0953 |
| observed_unequal2_bfloat16 | 593.384 / 475.864 / 1.2470 | 524.288 / 635.504 / 0.8250 | 1095.200 / 1096.520 / 0.9988 |
| observed_chunk_boundary_bfloat16 | 496.936 / 447.928 / 1.1094 | 526.048 / 541.200 / 0.9720 | 994.968 / 983.240 / 1.0119 |
| observed_chunk_tail64_bfloat16 | 628.960 / 458.360 / 1.3722 | 536.288 / 632.512 / 0.8479 | 1161.136 / 1086.448 / 1.0687 |
| fp8_first_chunk_float8_e4m3fn | 115.008 / 117.136 / 0.9818 | 349.296 / 342.608 / 1.0195 | 464.368 / 459.696 / 1.0102 |
| fp8_budget_exact_float8_e4m3fn | 291.784 / 299.760 / 0.9734 | 525.120 / 537.880 / 0.9763 | 814.144 / 827.296 / 0.9841 |
| fp8_dominant_long_float8_e4m3fn | 338.640 / 321.944 / 1.0519 | 539.168 / 561.552 / 0.9601 | 873.440 / 875.992 / 0.9971 |
| fp8_tail64_float8_e4m3fn | 386.360 / 320.240 / 1.2065 | 534.320 / 624.928 / 0.8550 | 927.176 / 947.688 / 0.9784 |
| fp8_tail65_float8_e4m3fn | 386.216 / 321.600 / 1.2009 | 530.928 / 625.152 / 0.8493 | 901.504 / 934.192 / 0.9650 |
| fp8_two_balanced_float8_e4m3fn | 395.504 / 366.416 / 1.0794 | 544.624 / 590.880 / 0.9217 | 937.920 / 954.712 / 0.9824 |
| fp8_two_skewed_float8_e4m3fn | 383.320 / 319.888 / 1.1983 | 530.688 / 625.032 / 0.8491 | 923.648 / 947.864 / 0.9745 |
| fp8_short_tail_float8_e4m3fn | 74.928 / 68.544 / 1.0931 | 92.928 / 104.000 / 0.8935 | 164.192 / 166.208 / 0.9879 |
| fp8_compact_float8_e4m3fn | 53.384 / 53.696 / 0.9942 | 181.040 / 161.392 / 1.1217 | 223.584 / 207.952 / 1.0752 |
| fp8_long100k_float8_e4m3fn | 1330.880 / 1308.768 / 1.0169 | 1153.320 / 1470.504 / 0.7843 | 2504.400 / 2815.904 / 0.8894 |
