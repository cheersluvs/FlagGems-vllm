<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 large-tile head grouping: restricted probe

Pre-code boards,2026-10-23. Evidence1684ce74: only BF16 R64/W4 yielded clear
score1.055534, registers232→207. Pipeline1.017897 overlapped about1.8% A/A
context/order variation. Four R32/W8 BF16 cases had no useful score benefit;
FP8 score0.813336 regressed with registers158→208. Keep production unchanged.

|Project truth|Frozen decision|
|---|---|
|Op/reference|Current QSA production8e4ad799, exact head-group kernel1edb3545, e1c188 production fixtures|
|Editable|New restricted policy/driver/manifest/runner/tests/docs only|
|Read-only|Both kernels, original unrestricted probe, production/YAML/TopK,12 benchmark shapes, timers/guards/tolerances|
|Build/validation|GPU JIT; all-row small bounds, per-case pre/post sampled score/selection checks, exact original PTX/cubin fallback proof, CPU gate/delegation/orchestration checks|
|Benchmark|Upstream12, fixed captured production configs; score/TopK/pipeline plus four A/A controls;2 rounds per case,4 rounds only for previously promising case|
|Aggregation|Incremental production/candidate ratios and equal-weight arithmetic means; no native acceptance claim|
|Timeout/profiling|1800s per worker,80C/120s cooldown; actual PTX/TTGIR/resources, no fresh NCU by default|
|Autotune|Exempt fixed-choice probe: exact per-chunk captured tiles/warps/stages, no retuning|
|Risk/next|Gate chosen by dtype/H/D/tile/warps, never benchmark case name; successful12 permits later30 regression, not production integration|

|Contract|Preserved|
|---|---|
|Input/output|Same BF16/FP8 public contract, padded strides/page196/offsets/budget, FP32 logits and int32 indices|
|Math|Full-D dot, per-head ReLU, pairwise head sum; original score/selection tolerances|
|Scope|Only BF16,H4,D128,TILE_R64,W4 chooses head-group JIT; all other metadata routes to exact original JIT|
|Autograd/torch|Inference only; no production changes or torch compute fallback; existing test references only|

|Path|Validation|
|---|---|
|Eligible BF16|Unchanged tested head-pair kernel; same configs and TopK; remeasure score/pipeline and A/A|
|Excluded BF16|Original JIT object with unchanged arguments; assert PTX and cubin equality after actual launch|
|FP8/compact|Always original JIT; first small boundary suite checks FP8 fallback numerical/canary and binary identity|
|Unseen H/D|Original JIT; CPU delegation checks unsupported head-group metadata never enters its static assertion|
|First worker|Three existing17-row boundary fixtures, reused by evidence hash/source/compiler for subsequent workers|
|Promising workload|Four alternating paired rounds including controls; case name affects measurement count only, not dispatch|

Captured LibEntry config/resource records are in first-seen row-count cache
insertion order under the pinned public observer. The expanded manifest maps
those distinct row counts to real chunks and verifies both config/resource
records agree. This preserves the mixed K_TILES16/4 case rather than assuming
one configuration per whole case. All12 frozen fixture dictionaries, input
hashes and compiler identity are checked before GPU execution.

Local validation:23 related CPU checks passed. The new checks execute the
actual metadata gate/JIT delegation, preserve all call arguments, reject changed
fallback binaries, verify the mixed-tail mapping and exactly one active workload
under the captured12-case configs, verify four-round override affects sampling
only, and exercise the CLI coordinator handoff so child processes execute the
restricted script rather than the unrestricted base driver. Existing head
mapping/canary/worker/timer and production-source contracts remain covered.
Pre-commit, CLI help and runner shell syntax passed. H800 compilation/numerics,
original-binary fallback proof and timing remain pending the returned report.
The original six-case probe, both kernels, all production files/configs and
benchmark shapes are byte-for-byte unchanged. Reports are uploaded automatically
and retain per-case effective rounds and actual per-chunk gate decisions.
