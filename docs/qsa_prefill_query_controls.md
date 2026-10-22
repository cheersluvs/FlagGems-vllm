# Focused QSA query-mode and BF16 pipeline controls

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Op/reference | Existing qsa_select_paged_prefill and pinned native vLLM; no single Torch API |
| Editable | New tools/qsa_prefill_query_controls.py, CPU contract tests, focused shell entry, this document; explicit optional benchmark hook/CLI/source archive |
| Read-only | Production source/exports, all kernels/YAML, frozen workloads, numeric tolerances, original timer/guard, correctness profile and TopK |
| Build/validation | Python syntax, pre-commit, CPU orchestration tests locally;46 existing iteration GPU nodes plus before/after diagnostic score/selection checks on H800 |
| Benchmark | Same FlagTree/ptxas13.4/CUDA13.4 compatibility profile, --mode kernel; baseline/candidate us and ratios, all samples; no30-case acceptance claim |
| Active set | Only fp8_first_chunk_float8_e4m3fn and observed_short_batch3_bfloat16 |
| Aggregation | Per condition/cycle/stage ratio of medians; list all raw paired ratios, no merging with native30-case mean |
| Runtime | Two timing rounds, warmup200ms/rep100ms;4 diagnostic cycles; initial80C, case1800s, accuracy7200s, cooldown900s/10s |
| Autotune | Existing public tune unchanged; diagnostic launchers intentionally replay one existing YAML choice per issue (FP8 TR32/BN64/KT4/W4, BF16 TR8/BN64/KT4/W4; stages2/ctas1); fixed replay is an explicit exemption, not production dispatch |
| Profiler | Compiled resource identities and metadata only; no occupancy/hardware-counter claims |
| Fallback | No production change or Torch fallback; test/reference Torch permitted only in diagnostic validation |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Exact two existing BF16/FP8 fixtures, layouts and page/request/chunk geometry |
| Math | Original full-D FP32 dots, ReLU before head reduction; query modes1/2 only for eligible FP8; unchanged masks and tolerances |
| Output | Existing int32 selection/-1 padding; ties follow existing reference normalization; same-buffer outputs validated per side before overwrite |
| Dtypes/promotion | BF16 or FP8 e4m3fn Q/K, FP32 scores, existing integer metadata/output |
| Scope | Diagnostic forward-only; no new unsupported/autograd/special-value promise; invalid options fail |
| Torch | Existing FP32 score/selection reference outside timing; no production import of diagnostic |

## Execution paths

| Path | Trigger | Implementation/config | Checks and risk |
| --- | --- | --- | --- |
| Existing run | No diagnostic flag | Original benchmark and acceptance behavior | Source/CPU regression tests |
| FP8 | first_chunk | Joint JIT at fixed32/4/W4 with mode1 versus2, same chunk geometry; score only | Independent/shared/swapped buffers; validate each score before/after; no claim tuner picks correctly |
| BF16 | short_batch3 | Canonical versus joint JIT mode1 at fixed8/4/W4; score, TopK, pipeline | Each output owner validated separately before overwrite; identical TopK object |
| Schedule |4 cycles | Balanced condition positions and predecessors; rotate stage order; each pair still original2-round AB/BA timer | Record exact order and timestamps; no assumption that cache state stays constant |
| Report | Success/failure | Original upload runner, nested JSON plus per-case controls TSV | Partial results retained; diagnostic never formal production acceptance |

The four buffer conditions are A/B, B/A, A/A and B/B for baseline/candidate.
A and B are the already allocated canonical and trial chunks, including score,
TopK scratch and output. Inputs and native common TopK logits remain identical.
No extra GPU workspace, replacement timer, synthetic cache flush or cache-policy
change is introduced. Shared-buffer checks execute and validate each side on its
own, so the second launch cannot hide an invalid first result. Pipeline validation
uses the unchanged FP32 selection checker; TopK uses unchanged native common
logits checks. All validation is outside timing.

This is a controlled follow-up, not a causal proof. In particular, buffer-order
rotation and fixed geometry reduce confounders but do not equate device cache
state across launches. Interpret repeated raw samples before choosing a dispatch
policy or integrating either private optimization.

## Delivery and interpretation

Run `bash tools/run_qsa_prefill_query_controls_h800.sh --flagtree /opt/flagtree`
from a clean checkout containing this change, using the existing activated vLLM
0.31 virtualenv and QSA_STOCK_PYTHON. The runner selects an idle H800 under the
unchanged guards, uses the verified FlagTree runtime, runs46 iteration nodes,
then only the two named benchmark fixtures and automatically uploads reports.
It never installs a compiler into the virtualenv. As before, all shell scripts
must run in a child bash, not be sourced into an interactive container shell.

Four cycles produce16 FP8 score comparisons and48 BF16 comparisons (score,
TopK and pipeline). Each comparison has the original two timing rounds in
AB/BA order, so diagnostic raw timing samples total256. The usual native/current
production stage comparisons for the two cases remain in the original stage
TSV; FP8 TopK and pipeline data are retained there. Diagnostic rows are separate
`*.controls.tsv` files, mirrored in per-case JSON under `query_controls`, with
cycle/position, buffer owners/pointers/layouts, actual fixed kernel resources,
checks before/after, time interval, timing order and every sample/paired ratio.
The result path is labeled `flagtree134-controls-*`; failures retain partial JSON
and completed TSV rows, and are uploaded by the existing runner.

Interpret the same-buffer A/A and B/B mode comparisons alongside A/B and B/A.
A stable FP8 slowdown for mode2 at the same tile would motivate investigating
tuning measurements/selection; it does not by itself specify a new dispatch
rule. For BF16, compare isolated score, common-input TopK and sequential pipeline
across orders and buffers. A change between cycles without a source/config
change indicates sensitivity requiring investigation, not proof of a compiler
or buffer-address cause. These two diagnostic fixtures cannot establish the
30-case performance objective or full production correctness.

Local validation:91 related CPU checks passed, including the new8 focused
checks. These cover all condition positions/precedences, fixed YAML eligibility,
child/CLI restrictions, both full control schedules, buffer owners and TSV/JSON
samples, immediate rejection of an invalid first shared-buffer result, and
rejection of changed fixed kernel records. Changed-file pre-commit, shell syntax,
Python syntax, production/YAML source identity and diff checks passed. No local
Torch/Triton/CUDA is available; GPU compilation, numerics and performance remain
pending H800 execution. No production/YAML/TopK change is included.

## Audited H800 result: 2026-10-22

Report branch `codex/qsa-h800-results-20261022-002430`, report commit
`cd2bc79e34cf4ea43ff4a128ec45903a331a7b38`, tested clean commit
`0dc3cf6bbe35523fdc0649b7ab606d2528de1cb0`.
All 21 archived files (19 unique Git blobs) were verified against GitHub sizes
and blob hashes. All 90 source snapshots match their recorded SHA256 and tested
commit; native reference matches its pinned source. Both case JSONs match the
summary; 13 main-stage TSV rows, 64 supplemental control rows, 304 raw timing
samples and 228 guard observations were independently checked. Fixed choices,
mode constants, chunk geometry, resource records before/after, buffer identities,
balanced schedule, AB/BA order, output checks, same compiler/runtime and original
kernel timer all match the plan. Every observed guard is clean; sampled GPU
32–47C and SM clock 1980MHz. No changed kernel or common TopK input was observed.

All 46 iteration correctness nodes passed in 179.20s. This is not the full
production suite. Benchmark telemetry spans 195.83s; their sum is about 6m15s,
excluding setup and time outside telemetry. No production source, YAML, export,
TopK implementation or private kernel changed for this follow-up.

The original overall acceptance field says measurement_valid=false/target_met=null
with exactly two reasons: planned and measured sets are not the frozen30. That
is expected for this two-case diagnostic, not a numerical/timing failure. It
must not update the previous 30-case acceptance result or mean. The runner,
individual stages and all 64 supplemental comparisons passed.

### Fixed-tile buffer controls

FP8 compares full-query mode1 against split mode2 using the same joint JIT at
TR32/BN64/KT4/W4/stages2/ctas1. BF16 compares canonical code against joint code
mode1 at TR8/BN64/KT4/W4/stages2/ctas1, with the identical TopK object.
A and B are the two existing buffer banks; table labels give baseline/candidate
owners. Each number is the arithmetic mean of four comparison ratios, one per
cycle. Each comparison ratio is its two-round baseline median divided by
candidate median. These ratios are not native-vLLM speedups.

| Buffer owners | FP8 score: full/split | BF16 score: canonical/joint | BF16 TopK: canonical/joint | BF16 pipeline: canonical/joint |
| --- | ---: | ---: | ---: | ---: |
| A/B | 0.9979 | 1.0000 | 0.9980 | 0.9998 |
| B/A | 0.9957 | 0.9972 | 0.9931 | 0.9996 |
| A/A | 0.9995 | 1.0019 | 0.9955 | 0.9988 |
| B/B | 0.9998 | 1.0013 | 0.9993 | 0.9996 |
| All 16 comparisons per column | 0.9982 | 1.0001 | 0.9965 | 0.9995 |

For FP8 first_chunk, 10/16 comparison ratios are below1 and6 above1; 20/32
individual paired ratios are below1 and12 above1. Comparison ratios range
0.9889–1.0054. Arithmetic mean0.9982 indicates a small tendency against splitting,
with no stable gain or repeated large regression. Pooling all raw samples gives
medians117.632us (full) and118.336us (split), ratio0.9941. This pooled ratio is a
different statistic from the arithmetic comparison mean; do not conflate them.
The four cycle means are0.9990/0.9980/0.9993/0.9966. Within-cycle round means
are1.0003 in AB and0.9962 in BA; small differences remain order-sensitive.
This does not justify a case-name branch or a confident universal exclusion of
the32/4/W4 split choice. Mode2 remains useful at other tiles in prior reports.

The real joint tuner selected mode1 for this fixture in this run, unlike the
previous mode2 selection. Its main tuned score/current-production ratio0.9979
and pipeline/current-production1.0051 are effectively close to parity in this
run. The existing selected-tile full/split main diagnostic is0.9930. Thus no
additional first_chunk dispatch restriction is supported yet; preserve both
modes and the current runtime-key selection for the next private experiment.
Selection flipping between runs also precludes claiming a guaranteed tuner win.

For BF16 short_batch3, all16 supplemental pipeline comparison ratios are within
0.9986–1.0014, mean0.9995. Both pooled raw medians equal22.528us. Across all four
conditions and repeated positions, neither the earlier0.9090 main result nor a
consistent disadvantage for one buffer bank reproduces. Main original-buffer
pipeline/current-production is0.9993; swapped replay is1.0000. Isolated score
mean1.0001 and TopK mean0.9965 likewise do not explain a stable9% pipeline loss.
The same TopK code has small measured differences across buffers/order; those
measurements are not evidence of a TopK implementation change. Native common
logits remain unchanged throughout.

Both BF16 fixed score kernels use66 registers,0 spills,43008 bytes shared memory
and identical static instruction-site counts; actual PTX/CUBIN hashes differ.
FP8 full/split uses123/80 registers,0/0 spills,40960/36864 shared bytes respectively.
The FP8 resource reduction does not yield a stable speed gain here; register
counts alone cannot determine occupancy or performance.

The evidence does not establish why earlier BF16 observations differed. There
is no controlled cross-run replay of the old timing state, hardware counters,
or proof of a cache/address/compiler cause. Treat the previous large loss as
not reproduced by these controls, rather than labeling its cause resolved or
deleting its data. Broad 30-case iteration remains at two rounds: one round
would hide differences that the AB/BA samples expose.

### Main native stage timings retained

These are the existing main-stage measurements for the two fixtures, separate
from the supplemental fixed-tile controls. Each cell is native us / candidate
us / native-to-candidate speedup. The sequential pipeline is timed directly.

| Case | Score | TopK | Pipeline |
| --- | --- | --- | --- |
| BF16 short_batch3 | 52.608 / 12.592 / 4.1779 | 13.984 / 13.584 / 1.0294 | 62.288 / 22.456 / 2.7738 |
| FP8 first_chunk | 114.272 / 118.944 / 0.9607 | 351.344 / 345.024 / 1.0183 | 463.328 / 460.248 / 1.0067 |

Do not mix baseline/candidate times from different comparisons or derive pipeline
latency by adding isolated stage medians. TopK remains measured and unchanged.

### Per-cycle diagnostic ratios

Every row below lists cycle1/2/3/4 comparison ratios, preserving repeated results
rather than choosing the best cycle. Complete paired samples and timestamps are
in the report TSV/JSON; all have before/after validation.

| Case/stage | Condition | Cycle1 | Cycle2 | Cycle3 | Cycle4 |
| --- | --- | ---: | ---: | ---: | ---: |
| FP8 score | separate | 0.9992 | 1.0020 | 0.9889 | 1.0015 |
| FP8 score | swapped | 0.9926 | 0.9986 | 1.0015 | 0.9900 |
| FP8 score | shared_a | 0.9986 | 0.9984 | 1.0034 | 0.9977 |
| FP8 score | shared_b | 1.0054 | 0.9930 | 1.0035 | 0.9974 |
| BF16 score | separate | 0.9987 | 1.0063 | 1.0025 | 0.9925 |
| BF16 score | swapped | 0.9975 | 1.0013 | 0.9937 | 0.9962 |
| BF16 score | shared_a | 0.9949 | 1.0128 | 0.9911 | 1.0089 |
| BF16 score | shared_b | 0.9975 | 0.9975 | 1.0064 | 1.0038 |
| BF16 topk | separate | 0.9918 | 1.0000 | 1.0000 | 1.0000 |
| BF16 topk | swapped | 0.9971 | 0.9942 | 0.9872 | 0.9942 |
| BF16 topk | shared_a | 0.9941 | 0.9942 | 0.9912 | 1.0024 |
| BF16 topk | shared_b | 0.9988 | 0.9977 | 0.9948 | 1.0059 |
| BF16 pipeline | separate | 1.0000 | 1.0014 | 0.9993 | 0.9986 |
| BF16 pipeline | swapped | 1.0000 | 0.9993 | 0.9993 | 1.0000 |
| BF16 pipeline | shared_a | 0.9986 | 0.9986 | 0.9986 | 0.9993 |
| BF16 pipeline | shared_b | 0.9993 | 1.0014 | 0.9986 | 0.9993 |

### Decision

These focused controls do not establish a stable large BF16 code regression or
an FP8 first_chunk split gain. Do not add a fixture-specific mode gate or change
TopK based on them. Preserve the private joint query-mode candidate and the
separately retained BF16-only page-table candidate; neither is integrated into
the canonical production operator by this result update.

The recommended next optimization experiment is to combine those two private
changes, keeping the FP8 page-table fallback and original TopK. Run the full
production correctness suite (not just46 iteration nodes) and all30 active
same-compiler kernel benchmarks before deciding production integration. This
also rechecks the remaining per-case floor failures. Two diagnostics cannot
prove the30-case arithmetic mean/per-case objective. The earlier source and
reports remain available for cross-checking, but comparisons of separate runs'
native means are not a substitute for same-run A/B.
