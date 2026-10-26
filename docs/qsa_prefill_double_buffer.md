<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Producer-first two-buffer lifetime probe

## Project board (before code, 2026-10-26)

|Field|Frozen choice|
|---|---|
|Task/evidence|Audit and test two K slots for producer-first QSA.93eb254a proved scheduling;73bb8c12 proved tested correctness but score ratios0.8752/0.6979 with three slots|
|Editable|This board, compile/protocol driver, gated runtime adapter, CPU tests, upload runner|
|Read-only|Production0d7e80e2,configs,references,prior helpers/probes,timers,tolerances,GPU guards,installed compiler and input reports|
|One change|For the two captured BF16 layouts only, reduce K ring3→2 after an extracted lifetime proof;change K memdesc ring dimension and its two modulo-wrap comparator operands only|
|Active set|R32 n1/q8192/p102400 and R64 n4/q4096/p102400;all captured launch configs and inputs unchanged|
|Build|CPU model/tests/CLI/bash/pre-commit;four compile-only replays:three-buffer control and two-buffer candidate per configuration|
|Compile gates|Control PTX exact;same MMA work,early cp.async,wait counts,zero known spills;K slots2,shared drops exactly16KiB;unconditional complete-CTA barrier after last WGMMA wait before backedge,correct current-copy wait/pre-MMA barrier/epilogue drain retained|
|Runtime gate|Both candidates must pass compile/protocol gates before GPU execution. Reuse original producer-runtime boundary,reference,canary,input/source checks unchanged|
|Benchmark|Only two cases,2 alternating rounds,20ms warmup/rep;existing kernel timer;score,unchanged TopK,pipeline and four A/A controls;ratios production/candidate|
|Timeout/safety|300s per CPU child,1800s per GPU case;original exclusive PID/idle/thermal guards80C/120s unchanged;no GPU execution during compiler phase|
|Autotune|Private fixed-config exemption;no production config/search changes|
|Fallback|Unsupported IR/protocol/profile is an error;no Torch compute fallback in production|
|Delivery|Automatically run CPU compilation then gated two-case runtime;compile-only or reuse-compile-report options for diagnosis;automatic report upload on failure/success|

## Contract board

|Field|Preserved/limited|
|---|---|
|Math/API|No query/cache/page-table/output address,mask,dtype,shape,head reduction,ReLU or JIT ABI change;only shared ring size and local slot wrapping|
|Synchronization|Do not remove/reorder copy commits,async waits,WGMMA waits,fences or barriers. Recompiled PTX must retain the extracted protocol|
|Scope|Two structurally verified BF16 H4/D128,BN64,K_TILES16 profiles only;R32 has2 warp groups and R64 has1. No general compiler patch,FP8 candidate or arbitrary-layout claim|
|References|Existing FP32 scoring and selection references/tolerances;original runtime JIT/ir_override binding;production unchanged|
|Coverage|Three boundary scenarios on both configs,all17 selected rows with offset5,zero/one/tile/page/width edges,empty request,canaries and unchanged inputs;main pre/post score/TopK/pipeline checks|
|Proof limit|Finite CPU protocol model and structural checks establish slot-lifetime safety under explicitly checked synchronization assumptions;they do not replace CUDA runtime numerical validation|

## Paths board

|Path|Action|
|---|---|
|Proof|Extract actual K allocation,copy/read views,counter phi init/yields and wrap operations. Verify one-tile lookahead and current-copy token history. Check final PTX loop has no internal branches,all WGMMA groups drained and full-CTA barrier before backedge|
|Model|Explore all group interleavings for0..16 loop iterations,1/2 groups,two slots,with current-copy wait/pre-MMA barrier/post-WGMMA barrier. Include unsafe no-reuse-barrier and single-slot counterexamples|
|Rewrite|Require one K allocation and three views;add independent constant2;change only two wrap compare bound operands and K memdesc3→2. Reverse transform must recover original text exactly|
|Compile|Replay final TTGIR directly through original LLVM/PTXAS;no Python/C++ compiler adapter or installed-file edit. Enable resource log only. Separate processes/cache per case/variant|
|Execute|Only after both compile gates pass,use original production JIT with raw-candidate-PTX-verified per-launch IR override. No ABI transplant or global override|
|Reject|Safety/provenance/compiler mismatch is nonzero;unpromising performance resources stop before runtime;no timing/integration claim from compile success|
|Integrate|None until correct/useful results;no30-case expansion this round|

## Extracted lifetime argument

For iteration i,the copied tile is i+1 and the consumed tile is i. With B slots,
write=(i+1)%B and read=i%B. With B=2,the destination last held tile i-1.
That tile's DMA had completed before its consumers started. All its WGMMA reads
have completed before every participating thread crosses the previous iteration's
post-result CTA barrier. Thus a thread entering iteration i cannot overwrite a
slot still read by another warp group from iteration i-1. Current K and next K
remain in different slots. The existing current-copy wait and pre-MMA CTA barrier
publish the whole current tile;the existing epilogue drains pending copies.

R32 requires explicit attention to two warp groups. Per-group WGMMA wait0 alone
is not a full-CTA reuse guarantee. The post-result bar.sync0 is therefore a hard
proof/generation gate. The negative model omits the reuse barrier and assumes no
additional pre-copy reuse barrier:it demonstrates a potential cross-group clobber,
not that deleting just one barrier in the archived PTX necessarily produces it.

Why three slots:LowerLoops.getDefUseStageDiff adds a wrap-stage adjustment when
consumer cluster follows producer cluster;loadRequiresAdditionalBuffer adds a
further WGMMA lifetime slot. Producer-first yields2+1=3. These generic rules do
not use the later materialization waits and output-layout conversion's CTA
barrier to shorten this particular kernel's lifetime. This probe tests a narrow
post-lowering opportunity and does not declare the generic rule incorrect.

PTX synchronization semantics reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

## Local validation (2026-10-26)

The new six CPU tests and the prior 24 protocol/runtime tests pass. The model
checks 34 configurations and 4,322 reachable states. Both actual archived H800
producer IR/PTX pairs pass the structural proof and produce four replay inputs;
their archived files remain unchanged. CLI help and bash syntax checks pass.
Repository hook versions are run from the installed cache because the GitCode
mirror cannot resolve in this sandbox. No CUDA compilation or kernel execution
has been performed on the Mac. Compile success, runtime correctness and speedup
remain separate pending card results.
