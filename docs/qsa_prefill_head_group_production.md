<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 head-pair production integration

Pre-code boards, 2026-10-24. Three fixed-config runs reproduced active-case
score ratio about1.0557 with registers232→207. Thirty-case regression passed
sampled correctness; FP8 tail113 timing excursion did not recur in four-round
recheck266de12d (score1.000750, TopK1.000087, pipeline0.999843). Original30
measurements are retained; no new30 average is inferred from the single recheck.

|Project truth|Decision|
|---|---|
|Op/reference|Actual exported QSA entry versus pinned native vLLM, same FlagTree/PTXAS13.4/runtime|
|Editable|Production rectangular scorer, new NVIDIA tuning group, tests/profiles/source pins/archive/runner and integration docs|
|Read-only|TopK, FP8/compact kernels, host routing/validation/workspace, benchmark30 tensors/timer/tolerances/guards, private proven head-group kernel|
|Build/validation|GPU JIT; original45 representative correctness nodes plus4 head-group mode/fallback checks; real30 pre/post sampled score/selection validation|
|Benchmark|scored-prefill30, kernel mode; native/production score, TopK, pipeline;2 rounds,200ms warmup/100ms repeat|
|Aggregation|Per-case ratios and equal-weight arithmetic means. NV scoring excludes TopK; TopK/pipeline still reported. No candidate/native ratio conflation|
|Timeout/thermal|1800s worker,7200s accuracy;80C initial ceiling,120s cooldown|
|Autotune|New qsa_select_paged_prefill_head_group_score group: all10 original configs with HEAD_GROUP=false plus one R64/BN64/KT16/W4/S2 HEAD_GROUP=true. Same keys/outer-cache; true mode pruned unless BF16/H4/D128/R64/W4|
|Profiler/risk|Compiled resources/configs recorded. Fixed-config benefit does not guarantee autotuned selection; actual public launch records decide|

|Contract|Preserved|
|---|---|
|Input/output|Existing BF16/FP8 QSA tensor/layout/page-table/visible-block contract, FP32 logits/int32 output|
|Arithmetic|Grouping splits heads0/1 and2/3 into full-D dots; per-head ReLU precedes pair sums; existing tolerances/selection ties|
|General fallback|All original ten configs retained; other H/D/dtypes never tune the grouped mode. No shape-name dispatch|
|Torch/autograd|Inference only; no production torch compute fallback, test references only|
|Validation limits|49-node representative suite plus sampled30 is not full production pytest coverage; full profile remains available|

|Path|Implementation and validation|
|---|---|
|Grouped eligible BF16|Compile-time branch in existing rectangular JIT, unchanged proven head-pair loads/dots/stores; true/false forced modes checked|
|Original rectangular|False mode retains original algorithm; autotune can select it even for eligible shapes|
|FP8/compact|Existing kernels and dispatcher unchanged; native30 and existing tests cover both|
|Unsupported grouping H/D|Prune true mode, keep original configs; representative H3/D24,H8/D128,H4/D256 checks|
|Formal public entry|Existing canonical identity/hash checks, capture actual tuned constants including HEAD_GROUP, no trial-module substitution|

Local verification:30 related CPU checks passed. The new transfer checks reduce
the actual compile-time branches and compare ASTs: grouped arithmetic exactly
matches the proven private kernel (renamed low-query variable, BF16 store),
false mode exactly matches frozen pre-integration rectangular arithmetic.
Every other pre-integration function remains AST-identical. Actual pruning
executes over real YAML choices, rejecting grouped modes for other dtypes/H/D
and invalid tile/warp regimes. Profile CLI resolves exactly49 selected GPU
nodes. Pre-commit and shell syntax pass; GPU compilation, numerical execution,
autotune selection and native30 performance remain pending. The original
pre-head production source is frozen byte-for-byte in tools for audit only;
production imports no probe/test/frozen module. Historical fixed-source probes
must be run at their pinned pre-integration commits, not silently repinned.
