<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Three-stage indirect-load pipeline feasibility probe

Pre-code boards,2026-10-26. Original production0d7e80e2 remains the baseline.
Loop-scoped stages2(report52201bb7) only stages page IDs and does not advance K.
Explicit tensor lookahead(reportd9d29a37) advances K but removes cp.async and
regresses R32. Test one alternate compiler schedule while preserving async K.
optimization.md/deep_opt.md are absent.

|Project truth|Frozen choice|
|---|---|
|Editable|Five new private files: kernel,driver,CPU tests,runner,this board|
|Read-only|Production,YAML,TopK,original references,all prior probes,workloads,timers,tolerances and GPU guards|
|Hypothesis|Only original range(first_tile,last_tile) becomes tl.range(first_tile,last_tile,num_stages=3);stages can separate indirect page-ID load,K copy and compute|
|Algorithm|Original serial full-D dots/per-head ReLU/head sum/stores;no manual tensor carry,no dot-order combination,no page-ID dedup|
|Active set|Two captured BF16 cases: n1/R32 and n4/R64,long prefix;all original launch configs remain unchanged|
|Motion gate|Identify BF16 K separately from i32 page IDs;require actual K async copy before first dot where baseline has fewer;direct global reads never satisfy this gate;require cp.async.ca/cg.shared.global in PTX|
|Resource gate|Same static WGMMA work,one identifiable main loop,two K shared slots,known register/shared metadata and zero spills;permit at most1024 extra shared bytes for small metadata staging,not an extra K tile|
|Budget rationale|Previous stages2 costs256B metadata without motion;small metadata increase alone is not evidence of slowdown. Here motion and async preservation are prerequisites,with a fixed1KiB allowance. Actual latency remains decisive|
|Timing|Independent per configuration;only eligible compile results get kernel-mode score,TopK,pipeline and four A/A controls;2rounds,20ms warmup/20ms rep|
|Runtime/safety|Same pinned FlagTree/PTXAS13.4/CUDA13.4 compat and captured inputs/compiler;exclusive GPU/PID/thermal guards unchanged,80C initial ceiling/120s cooldown|
|Autotune|Private fixed-launch-config single-schedule probe;no search,new production config or backend intrinsic|
|Delivery|Terminal evidence and automatic JSON/TXT/TSV/CSV upload via safe child-bash|

|Contract|Preserved|
|---|---|
|Math/API|Same original production function signature and body except loop iterator;FP32 accumulations,head grouping order,ReLU/reduction/store masks unchanged|
|Memory/layout|Original vector page-table and cache loads,all offsets/strides/visible bounds;no additional source-level speculative read|
|Activation|Rectangular BF16,H4/D128,R32/R64,BN64;other routes/dtypes use exact original JIT and binary equality proof|
|Boundary checks|Original three BF16R32/BF16R64/FP8 all-row fixtures,padded page/table/query layouts,zero visibility,tails and output canaries|
|Reference Torch|Diagnostics only;production has no torch compute fallback|

|Path|Decision|
|---|---|
|CPU|Whole-function AST proof;typed async-copy/matrix-loop parser excludes prologue,metadata and synchronous loads;actual archived artifacts calibrate false positives;resource/gate/child-handoff/coordinator behavior tested|
|Compile|Save unique case/PID/counter artifacts before rejecting;print early async K copies,cp.async sites,K slots,register/shared/spills|
|Reject|not_promising/exit0 is a completed negative compile probe;no timings;continue other configuration|
|Execute|Compile feasibility is not final SASS overlap proof. Original numerical,input,compiler,source,compiled/common-logit and GPU checks precede/follow timings. Numeric/environment/compiler failures remain nonzero|
|Integrate|None this round. Expand only after reproducible gain beyond A/A|

Loop-level versus launch-level pipeline behavior:
https://triton-lang.org/main/python-api/generated/triton.language.range.html

## Local validation (2026-10-26)

- 62 CPU tests passed: 8 new tests plus 54 related probe/profiling tests.
  The new AST check restores only the loop iterator/name/decorators and proves
  the entire function identical to production, including all address/mask/math/store expressions.
- Typed parser calibration used 30 archived loop-hint IR/PTX pairs and 54 archived
  explicit-prefetch pairs. All baseline and stages2 K copies remain after the
  first dot. Explicit-prefetch BF16 copies move earlier but have zero async K
  copies, zero PTX cp.async sites and one K shared slot, so cannot pass this gate.
  FP8 boundary artifacts have no matching BF16 K slots and remain controls.
- Tests distinguish metadata-only async motion, synchronous K reads, missing PTX
  async instructions, unknown resources, spills, changed MMA work and extra K
  buffers. They verify child-driver routing, artifact export before rejection,
  continuation after a rejected configuration and preservation of numeric failures.
- Python CLI help, bash syntax and all applicable pre-commit checks passed.
- The new stage3 kernel has not been GPU compiled or timed locally. No claim of
  improved scheduling, latency or numerical correctness is made before the card run.
