<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 grouped-head dot scheduling probe

Pre-code boards, 2026-10-24. Report a2553a16: R64 grouped BF16 has 9320/10123
barrier samples at two WGMMA waits (source453/457), while output-region shared
wavefronts equal ideal. This is a consumer dependency, not evidence that the
barrier can be removed. Test one portable source-order hypothesis.

|Project truth|Frozen choice|
|---|---|
|Editable|Five new private probe files only: kernel,driver,CPU tests,runner,this board|
|Read-only|Production source SHA0d7e80e2,YAML,TopK,all previous probes and their guards/timers/fixtures|
|Hypothesis|Issue independent high-head tl.dot before low-head ReLU/reduction; retain both full-D dots,each pair's ReLU/sum,and final low+high|
|Active set|perf_n4_q4096_p102400_bfloat16 and observed_chunk_boundary_bfloat16; fixed previously captured production configs|
|Compiler|Same pinned FlagTree,TLE,PTXAS13.4,CUDA13.4 compat and runtime as captured production report|
|Compile gate|Persist PTX/TTGIR/resources first. Require same total WGMMA count, more WGMMA instructions before first result wait,and no additional result-wait sites; require known registers/shared and zero spills. No timing if gate fails|
|Resource risk|Two simultaneously live accumulators may exceed original207register footprint; report actual values and reject spills before timing|
|Timing|Only after gate: inherited kernel-mode score,TopK,pipeline plus four identical-callable A/A controls. 2rounds,200ms warmup,100ms rep|
|GPU safety|Unchanged strict exclusive GPU admission and before/after stage PID/thermal guards;80C initial ceiling,120s cooldown; no process killing|
|Profiler|No NCU rerun in this round; compiler artifacts test schedule feasibility, small A/B tests whether feasibility helps|
|Autotune|Exempt private one-hypothesis probe; freeze original constants, no search|
|Delivery|Safe child-bash runner with automatic text/JSON/TSV/CSV report upload; print resources,wait sequence and all stage/A/A timings|

|Contract|Preserved|
|---|---|
|Dtype/API|Only BF16,H4,D128,TILE_R64,HEAD_GROUP true rectangular route activates candidate|
|Math|Unchanged FP32 full-D dot accumulations,per-head ReLU,head-pair reductions and final low+high ordering. No fast-math or tolerance change|
|Layout/bounds|Same request lookup,query/cache/table strides,page196,visible bounds,query offset,mask and logits store|
|Fallback|R32,FP8,compact and other arguments use exact original JIT; verify compiled hashes for inactive chunks|
|Boundary tests|Inherited three dtype/tile fixtures with all-row checks,visible0/1/tile/page tails,chunk offsets,padded cache/table and output canaries; grouped R64 enabled by frozen trace helper|
|Torch|Reference/fingerprint/allocation only in diagnostics; no production compute fallback|

|Path|Gate/action|
|---|---|
|AST proof|Private kernel equals canonical kernel after moving only high-dot assignment and renaming function/removing tuner decorators|
|Compile/warm|Warm and validate original/candidate; save unique case/PID/counter artifact snapshots before rejecting resources/schedule|
|Rejected hypothesis|Mark not_promising,exit0 as a completed negative probe; leave timings absent,do not manufacture speedup|
|Valid hypothesis|Run existing numeric,input,compiler,binary,source,GPU checks and interleaved timing/A/A unchanged|
|Integration|None this round. Improvement requires later repeated/regression evidence before production inclusion|

Validation before delivery: 37 applicable CPU checks passed (6 new plus existing
stage-depth,production-profile and profiling checks), targeted pre-commit and
bash syntax passed. Private-kernel AST equality proves only the high-dot
assignment moves. On the verified archived R64 PTX, the schedule parser finds
16 MMA sites,2 waits and8 MMAs before first wait. New GPU compilation, numerical
validation and timing are pending; no benefit is claimed. Historical head-group
unit fixtures pin a pre-integration production hash and are not the acceptance
suite for this current-production probe.
