<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Loop-scoped indirect-load pipeline probe

Pre-code boards,2026-10-24. Current production report a2553a16 has a two-slot K
shared buffer,but TTGIR schedules the next K async copy after dot/reduction/store.
Long-scoreboard samples occur along the indirect address/load chain. The prior
high-dot issue-order candidate f2a01f29 stays unintegrated;use original production.

|Project truth|Frozen choice|
|---|---|
|Editable|Five new diagnostic files only: private kernel,driver,CPU tests,runner,this board|
|Read-only|Production SHA0d7e80e2,YAML,TopK,old harnesses/guards/timers/fixtures and all old candidates|
|Hypothesis|Only change range(first_tile,last_tile) to tl.range(first_tile,last_tile,num_stages=2);loop-scoped pipeline can include non-dot loads unlike the launch-level setting|
|Source scope|Original production kernel,including serial grouped dots and original stores;no dot-schedule/page-ID-dedup/manual-prefetch combination|
|Active cases|perf_n1_q8192_p102400_bfloat16,R32;perf_n4_q4096_p102400_bfloat16,R64;fixed old production configs|
|Compile gate|Save artifacts first;require one identifiable main TTGIR loop,a next-K async copy before the first matrix dot where original has fewer,same total PTX WGMMA work,known resources,zero spills,and no larger shared footprint|
|Timing|Only after compile gate:2 rounds,20ms warmup/20ms rep,score excluding TopK plus TopK,pipeline,and four A/A controls;production/candidate ratios,not native-vLLM acceptance|
|GPU/runtime|Pinned FlagTree/PTXAS13.4/CUDA13.4 compat and captured inputs/compiler;strict original exclusive PID and thermal guards,80C initial ceiling/120s cooldown|
|Autotune|Private fixed-config hint probe;no search and no changes to launch constants|
|Negative result|not_promising/exit0,explicit no timing;retain artifacts and source identity. An execution/numeric/environment failure remains nonzero|
|Delivery|Safe child-bash with automatic report upload;terminal prints compiler evidence and all completed stage/A/A timings|

|Contract|Preserved|
|---|---|
|Dtype/gate|Only rectangular BF16,H4,D128,TILE_R32/64;all other routes use original JIT and require exact unchanged PTX/cubin|
|Math|Every dot,ReLU,head reduction and addition remains byte-for-byte AST-identical|
|Bounds/layout|Original vector page lookup,cache/query/table strides,visible limits,request/chunk mapping and masked stores;no extra source-level memory load|
|Boundary evidence|Inherited three all-row BF16R32/BF16R64/FP8 fixtures,padded cache/table,page196,tails,visible0 and output canaries|
|Torch|Test references/fingerprints only;no production compute fallback|

|Path|Gate/action|
|---|---|
|CPU|Whole-function AST equality after restoring only loop iterator/decorators/name;synthetic IR tests distinguish prologue,loop and post-dot copies;gate rejects missing/no-improvement/resource-regression evidence|
|Compile|Export uniquely named case/PID/counter PTX/TTGIR/resource snapshots before rejection. IR motion is feasibility evidence,not proof of reduced latency or final SASS scheduling|
|Run|Inherited pre/post numerical,input,compiled-kernel,common-logit,source/compiler and GPU checks;all stage and A/A timings retained|
|Integration|None this round. No new profiler capture unless a measurable signal needs explanation|

Official distinction between loop- and launch-level stages:
https://triton-lang.org/main/python-api/generated/triton.language.range.html

Delivery validation,2026-10-25:44 applicable CPU checks passed (7 new plus
existing dot-schedule,stage-depth,production-profile and profiling checks).
Targeted pre-commit,bash syntax and CLI help passed. The actual verified
production first chunks calibrate the IR parser: R32 has first dot atTTGIR250
and next-copy at308;R64 has first dot at255 and next-copy at324;both have zero
loop K async copies before the first dot. CPU coordinator simulation confirms
an unpromising R32 case does not prevent R64 collection,and negative cases are
excluded from measured-case arithmetic means. Unknown/missing evidence is not
turned into a speedup. GPU compilation,numerics and timings are pending.
