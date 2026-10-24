<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Restricted BF16 head grouping: 30-case regression

Pre-code boards, 2026-10-24. Report 2f7c408e reproduced the active upstream
case score ratio 1.055475 (four pairs 1.053516–1.056197). Pipeline ratio
1.020840 is still close to A/A variation. All other 11 cases used identical
original binaries. This step tests regression risk, without production changes.

|Project truth|Decision|
|---|---|
|Baseline|Production SHA8e4ad799, captured e1c188 report, same FlagTree/PTXAS13.4/runtime|
|Editable|New regression driver/manifest/runner/CPU tests and this document only|
|Read-only|Production, kernels, gated12 and unrestricted drivers, YAML, shapes, timers, numerical tolerances, GPU guards|
|Build/validation|GPU JIT; expanded all-row score/canary checks; actual 30-case pre/post sampled score/selection checks; excluded chunks require original PTX/cubin identity|
|Benchmark|Unchanged scored-prefill30 (12 upstream BF16 +8 observed BF16 +10 FP8), kernel mode; score, TopK, pipeline plus four A/A controls|
|Sampling|Two rounds, four for previously promising n4/q4096/p102400; 200ms warmup/100ms repeat; 80C initial ceiling, 120s cooldown|
|Aggregation|Per-case production/candidate ratios and equal-weight arithmetic mean, also split by group; no native-vLLM acceptance claim|
|Autotune|Fixed-choice probe exemption: captured per-route/per-distinct-row configs, no retuning or new search space|
|Timeout/profiling|1800s worker; PTX/TTGIR and resources recorded, no repeated NCU collection|

|Contract|Decision|
|---|---|
|Input/output|Unchanged public QSA BF16/FP8 contract and frozen workload tensors; FP32 logits, int32 selection|
|Math|Full-D dot, per-head ReLU, paired head sum; existing reference tolerances|
|Scope|Only rectangular BF16 H4/D128/R64/W4 uses the unchanged head-pair kernel|
|Torch/autograd|Inference; torch compute only in validation/reference; no production fallback or changes|
|Correctness limitation|Real large cases remain sampled. Expanded small fixtures check every score row, masks and output canaries; this is not the full production pytest suite|

|Path|Validation|
|---|---|
|Eligible BF16|Expanded row/tile/page/request/chunk boundaries using original and candidate kernels|
|Excluded BF16|Original JIT and unchanged args/configs; exact PTX/cubin identity|
|FP8/compact|Original JIT and binary identity across actual 30 fixtures, including FP8 query-split and compact routing|
|Head/dim fallback|Small BF16 H2/D64, H8/D128, H4/D256 fixtures use original JIT|
|Orchestration|Reuse gated12 CLI/worker/runner, nested context restores globals; child processes invoke regression driver|

Each of three boundary groups produces one passed evidence record containing
its subchecks; the existing worker evidence protocol still requires exactly
three groups. The original three 17-row fixtures are retained. Extra fixtures
cover rows 1/63/64/65/129, query offsets, request crossings, page sizes 64/196,
visible lengths 0/1/63/64/65/195/196/197/511/512/513/1024/1025, padded layouts,
and head/dim fallbacks. Inputs are finite random data; NaN/Inf semantics are not
newly claimed. Keep production unchanged until regression and integration tests
support merging.

Local verification: 29 related CPU checks pass, including all30 manifest mapping,
unchanged upstream12/mixed-tail configs, exactly four eligible real chunks,
expanded fixture grouping, stage-preserving group aggregation and scoped CLI
handoff/restoration. Shell syntax and pre-commit pass. GPU execution is pending;
no correctness or speedup is inferred from CPU orchestration tests. The fresh
runner automatically uploads existing reports on test failure and prints a
retry command when upload fails, so no benchmark rerun is needed for upload.

Report d3d182b0 passed 13 boundary fixtures and all30 pre/post sampled checks.
Only four real chunks activate grouping; 49 excluded chunks have identical
original PTX/cubin. Active score ratio1.055746, pipeline1.019499, TopK1.001143.
Raw all30 means score0.999168/TopK0.999894/pipeline1.001019; FP8 tail113 has a
second-pair timing excursion despite identical binaries and needs targeted
rechecking rather than attributing it to changed kernel code. The grouped
summary helper was omitted from the CLI coordinator hookup; this reporting-only
fix connects it and tests the actual CLI path through the wrapper. Existing
per-case data remain complete and do not require rerunning for aggregation.

## Targeted timing recheck, 2026-10-24 (pre-code boards)

|Project truth|Decision|
|---|---|
|Editable|Gated CLI round choices, regression reporting, matching CPU tests, this note|
|Frozen|Production/kernel/configs/workload/timer/reference/tolerances/GPU guards; same captured e1c188 fixtures|
|Hypothesis|FP8 tail113 second-pair excursion is transient measurement instability; cause unresolved|
|Measurement|Only fp8_tail113_float8_e4m3fn, four alternating paired rounds, score/TopK/pipeline and four A/A controls; preserve every raw sample|
|Build/validation|Existing 13 boundary fixtures once, sampled real-case checks, original PTX/cubin identity, CPU CLI and reporting checks|
|Timeout/thermal|1800s worker, 200ms warmup/100ms repeat, 80C ceiling/120s cooldown unchanged|
|Aggregation|One-case ratios only, not replacement of the original30 summary; no automatic noise filtering or acceptance declaration|
|Autotune|Same fixed-choice exemption; no candidate/config search|

|Contract|Preserved|
|---|---|
|Inputs/outputs/math|Identical frozen FP8 tail113 tensors and existing QSA score/selection contract|
|Torch/autograd|Test references only, inference, no production changes or torch fallback|
|Coverage|Boundary and sampled real checks retained; full production pytest remains separate|

|Path|Action|
|---|---|
|FP8 recheck|Exact original scorer both sides; require binary fallback proof; show raw samples and latency spread|
|Other shapes|Not run by the targeted command; CLI default remains two rounds|
|Eligible BF16|No changes to dispatch or kernel; prior four-round promising override retained|

Targeted recheck implementation: the gated CLI accepts `--rounds 4`; the default
stays2. For an exactly one-case tail113 report, the coordinator checks original
binary fallback proof and prints/saves raw samples and max/min latency ratios
for all stages and A/A controls. Raw stage results are never rewritten. Local
validation:31 related CPU checks, including actual four-round CLI handoff,
preservation of anomalous samples and rejection of absent binary proof; GPU
measurements pending. The existing runner provides automatic report upload.
