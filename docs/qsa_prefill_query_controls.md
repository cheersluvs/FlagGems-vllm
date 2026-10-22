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
| Schedule |4 cycles | Rotate/reverse condition-stage list; each pair still original2-round AB/BA timer | Record exact order and timestamps; no assumption that cache state stays constant |
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
