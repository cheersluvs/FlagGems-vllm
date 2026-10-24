<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 page-table fetch deduplication on the integrated production baseline

Pre-code boards,2026-10-24. Production report78c3ca4f is the baseline. The
stage-depth probe1f7dfc7c regressed BF16 scoring on average, so retain stage2.
Current R32/W8 and grouped R64/W4 kernels load page IDs across BLOCK_N lanes,
although BLOCK_N64 and PAGE_SIZE196 imply at most two distinct page IDs per
tile. Earlier page-dedup evidence used a different scorer and cannot be carried
over without this controlled test. No new hardware-counter causality claim.

|Project truth|Frozen decision|
|---|---|
|Editable|New private JIT/probe/runner/CPU tests and this document|
|Read-only|Production0d7e80e2, YAML, head-group/stage settings, TopK, frozen six fixtures, numerical tolerances, timers and GPU guards|
|Hypothesis|One/two scalar page-ID fetches plus broadcast remove redundant metadata work without hurting key loads/dots|
|Measurement|Reuse exact six-case stage-depth manifest from78c3ca4f: five BF16 cases plus unchanged FP8 long100k control. Two alternating pairs; score/TopK/pipeline and four A/A controls|
|Build/validation|GPU JIT; existing17-row all-row boundary/canary groups cover BF16 R32/full heads, BF16 R64/grouped heads and FP8 fallback; small-page fallback checked separately. Real cases checked before/after timing via sampled logits/selection|
|Aggregation|Current production/candidate arithmetic means, BF16 and unchanged FP8 control separately. No native-vLLM or formal acceptance claim|
|Autotune|Fixed-choice probe exemption: preserve exact per-chunk production configs, including HEAD_GROUP and num_stages2. No tuning search|
|Timeout/thermal|1800s worker,200ms warmup/100ms repeat,80C initial ceiling/120s cooldown|
|Profiling|Export PTX/TTGIR/register/shared/spill evidence and A/A. Fresh NCU only if needed to explain a measured signal|

|Contract|Preserved|
|---|---|
|Math/output|Exact original dots/head reduction/masks/stores; modify only page-ID fetch expression|
|Gate|Only BF16 rectangular and 0<BLOCK_N<=PAGE_SIZE use candidate JIT. Other routes use exact original JIT and require PTX/cubin equality|
|Layouts|Same table request/page strides, shuffled physical pages, padded Q/cache/table and nonzero query/chunk offsets|
|Torch/autograd|Test references only; no production edits or torch fallback; inference only|
|Scope|Finite inputs with valid device metadata; small boundary cases check all rows, large cases sampled, no full-suite claim|

|Path|Validation|
|---|---|
|One-page tile|One masked scalar ID; second load masked off; broadcast to lanes|
|Two-page tile|First and second scalar IDs guarded by table bounds and actual visible tile extent|
|FP8/compact/small pages|Original JIT; exact original binaries. No stage1 or head-mode change|
|CPU proof|AST equality outside page fetch; actual scalar/vector load code executed under a masked pointer simulator for page/tile/visible/table bounds|
|Next gate|Only stable gain beyond A/A warrants broader regression or production integration|

Local verification:45 related CPU checks passed. New checks execute the actual
scalar/vector page-fetch AST across1512 page/block/tile/visible/dtype combinations,
assert masked pointer bounds and valid-lane page-ID equality, and limit scalar
reads to at most two. They compare the remaining private JIT body/signature to
production, exercise real metadata delegation/factory/config preservation,
reject changed configs or fallback binaries, verify page196/64/16 evidence
aggregation and subprocess handoff/restoration, and keep BF16/control means
separate. Pre-commit, CLI help and runner syntax pass. Production0d7e80e2 and
YAML unchanged. GPU JIT compilation, nine17-row all-row boundary/canary fixtures,
six-case sampled correctness/resources/timing remain pending. Results are printed
in the terminal and automatically uploaded; summary is page-dedup-summary.json.
