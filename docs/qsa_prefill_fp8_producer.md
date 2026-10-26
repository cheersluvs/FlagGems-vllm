<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# FP8 producer-first compile probe

Pre-code boards,2026-10-26. Reportfc1a6333 verifies current FP8 P2 main loops
have all four16-byte K-copy instructions after the last MMA,not before the
first. Main long-scoreboard22-23% supports a delivery hypothesis but is not a
causal attribution. Efficient vector stores already reach about1 sector-byte/
payload-byte. Test producer-first scheduling only;no new source/kernel variant.

|Project truth|Frozen decision|
|---|---|
|Editable|New compile-only driver,CPU checks,upload runner and this board|
|Read-only|Production0d7e80e2,YAML,TopK,profiling and earlier probes,references,fixtures,installed compiler|
|Active set|Only tail113 main and long100k main,FP8/H4/D128/TR64/BN64/KT16/P2/W4/stages2;first_chunk and113-row tail unchanged/excluded|
|One change|Reverse the original0/1 loop cluster ordering in each independently captured pre-pipeline GPU IR;all operations,operands,stage numbers,math/masks and stores remain byte-identical|
|Build|Per case:full captured TTIR replay with pass dump,unmodified scheduled-IR resume control,producer-first resume candidate;six CPU compilations,explicit CUDA SM90 target,GPU hidden|
|Identity|Full replay and resume control must reproduce exact captured PTX;cache TTIR/options/cubin must match validated profile. Keep separate case options even when final old PTX hashes coincide|
|Inspection|Typed64x128 FP8 K copy,final16-byte copy ordering inside main backedge,MMA/wait counts,ring slots,actual PTXAS regs/spill bytes,shared footprint and current-copy/post-WGMMA/epilogue protocol|
|Resource gate|Zero known spills,same matrix/copy/wait work,all K fragments before first MMA,compiler ring2 or3 and shared change explained exactly by K-slot delta times8192B;unexpected costs reject before any runtime|
|Validation|CPU parser/rewrite/resource tests and real returned artifacts;no numerical GPU claim. Do not shrink any buffer or delete a wait/barrier|
|Benchmark|None this round;no GPU launch or new score/TopK/pipeline timing. Resource/ordering success requires later numeric/lifetime review and short A/B before any production change|
|Autotune/profiler|Fixed-capture diagnostic exemption;no tune changes or new NCU pass;use existing pipeline resume helper and resource parser|
|Timeout|300s per isolated compiler child;failure/success archive and auto-upload|

|Contract|Preserved/limited|
|---|---|
|API/data|Same Q/cache FP8 E4M3FN,FP32 dot/ReLU/head-sum/stores,INT32 metadata/output,head/query order and physical-page masks|
|Target|One selected main chunk per case;source cases/options preserved independently. Tail113's113-row chunk and first_chunk are not compiled by this probe|
|Compiler|Original pinned FlagTree/PTXAS13.4;reuse validated private in-memory pipeline resume adapter,no installed file edit or source JIT/hint/driver discovery|
|Buffer lifetime|Original compiler generates its own ring. One FP8 K slot is64*128*1=8192B. BF16 two-slot shrinking code/proof is not applied. Any later ring reduction needs a new FP8 lifetime proof|
|Evidence limits|No stall percentages as runtime fractions,no source-order-only overlap claim,no BF16 speedup transfer,no performance acceptance from compilation|

|Path|Action|
|---|---|
|Capture input|Require complete successful FP8 profile and exact source/input/compiler/actual-preparation/NCU identity;locate PTX-matched TTIR/options in preserved per-case cache,check cubin too|
|Replay|Dump original passes and extract unique Before TritonGPUPipeline snapshot;require exact PTX before transforming anything|
|Control|Reuse scheduled snapshot unchanged with original pipeline+tail;require exact captured PTX|
|Candidate|Only reverse cluster numbers;require reverse rewrite recovers original text. Compiler decides buffering;no manual memory rewrite|
|Decision|No actual early copy/spills/work/resource mismatch:negative probe without timing. Successful ordering/protocol/resources:pending correctness/lifetime review,not production-ready|
|Delivery|Show per-case copy order/slots/regs/shared/spills/gate;archive inputs,passes,PTX/TTGIR/cubin/proof and logs;automatic upload on all outcomes|

Local validation:36 applicable CPU checks pass (five new typed-motion/rewrite/
resource/replay-input/safety checks plus original pipeline/profile regressions).
Both actual fc1a6333 main-loop TTGIR/PTX pairs calibrate to ring2,two dots,
eight MMA instructions,four16-byte K-copy sites,zero before first MMA and four
after last MMA. No baseline or compiler source was edited. CLI help,shell syntax
and changed-file repository hooks pass. Captured GPU-side TTIR cache lookup,
exact replay,real pre-pipeline cluster shape and modified compilation remain
pending card execution. No local CUDA compilation,buffer shrink,GPU numeric
validation,timing or production acceptance is claimed.
