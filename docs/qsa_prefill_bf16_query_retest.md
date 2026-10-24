<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 large-tile query subdivision versus current head-group production

Pre-code boards,2026-10-24. Old gated query split had upstream12 evidence against
an older combined scorer (059fcb6b), not current production. Current baseline
0d7e80e2 integrates head pairs. Retest the existing full-head query subdivision
as a competing decomposition; do not silently call a new head+query composition
an old-candidate retest. No page-ID dedup or stage1 in this experiment.

|Project truth|Frozen decision|
|---|---|
|Editable|New private query-split JIT/driver/runner/CPU tests and this document|
|Read-only|Production/YAML/TopK, six-case captured78c3ca4f manifest, timers/guards/tolerances|
|Baseline|Current production stage2, actual captured tiles/warps/head-group flags|
|Candidate|Only BF16/H4/D128/R64/KT16 rectangular chunks: replace head pairing with two query halves, each computing all four heads. HEAD_GROUP=false explicitly in candidate configs; other config fields unchanged|
|Probe|Reuse six fixtures: five BF16 and one unchanged FP8 control. Two rounds, score/TopK/pipeline plus four A/A controls|
|Build/validation|GPU JIT; all-row129-row boundary/canary checks with whole and cross-request chunk schedules; sampled real pre/post score/selection; fallback PTX/cubin identity|
|Aggregation|Current production/query-split ratios; active cases and exact-binary controls separately, avoiding no-change timing bias. Not native-vLLM acceptance|
|Autotune|Fixed-choice exemption: no search; unchanged captured tile/warp/stage/budget choices|
|Timeout/thermal|1800s worker,200ms warmup/100ms repeat,80C initial ceiling/120s cooldown|
|Resources|Export PTX/TTGIR/register/shared/spill evidence before wider regression; no new NCU by default|

|Contract|Preserved|
|---|---|
|Math|Full-D dots, per-head ReLU then head sum. Query halves cover the same64 query rows and reuse the same key tile. No split-K or approximate math|
|Layout|Same page/cache/Q strides, query base/chunk offsets, visible lengths and FP32 output masks/canaries|
|Unsupported split|Non-BF16/non-H4D128/non-R64KT16/non-rectangular: exact original JIT. No runtime shape-name routing|
|Torch/autograd|Reference tests only, inference only; production unchanged, no torch fallback|
|Limitations|Large cases sampled, not full production pytest. Head+query composition and full joint retuning remain untested|

|Path|Treatment|
|---|---|
|Active captured R64|Compare current grouped-head scorer with the old full-head query-half decomposition; candidate flag change recorded explicitly|
|R32/FP8/compact controls|Original JIT and configurations with exact PTX/cubin equality|
|Boundary|BF16 R32 control, BF16 R64/grouped baseline vs full-head query split, FP8 control;129 rows, zero-length request, padded strides/page196/nonzero base; whole and split chunks exercise both halves and tails|
|Next gate|Only clear useful current-baseline benefit warrants upstream12 or30 expansion; no integration now|

Local verification:46 related CPU checks passed. Actual private query loads,
head ReLU/reduction and output stores execute against a masked NumPy pointer
model across24 start/tail combinations, checking both query halves, padded
head stride, output offsets, invalid-key -inf, masks and canaries. Request/key
mapping and signature are AST-identical to production. Actual factory, config
switch, metadata routing, freeze/binary rejection, boundary schedule and child
handoff/summary grouping are checked. Frozen manifest yields three active cases
(six chunks) and three exact-binary controls; negative controls are not counted
as optimization gains. GPU compilation, six all-row129-row boundary schedules,
real-case sampled correctness and timings remain pending. Production and YAML
unchanged; head+query composition is explicitly not part of this retest.
