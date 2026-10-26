<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# R64 source-expression compile feasibility

Pre-code boards,2026-10-26. Reports fd83dd02 and151e6e5d reproduce R64 score
production/candidate ratios1.05784 and1.05231 for an IR-only producer-first
ring2 candidate. R32 regresses and is excluded. The previous explicit source
lookahead at loop stages1 uses register loads instead of cp.async and has no
stable gain. Test one remaining source-level hypothesis:allow stages2 for that
same explicit lookahead. This is a feasibility check,not a production edit.
optimization.md and deep_opt.md are absent.

|Project truth|Frozen decision|
|---|---|
|Editable|This board,new CPU compile driver,CPU tests,and upload runner|
|Read-only|Production0d7e80e2,all prior probes,source kernels,configs,TopK,reference,fixture,timer,installed compiler and returned reports|
|Active set|Only BF16 H4/D128,BN64,R64,HEAD_GROUP true,K_TILES16,num_warps4,num_stages2 from validated n4/q4096/p102400 first chunk|
|One hypothesis|Explicit source lookahead plus loop-scoped stages2 can preserve early hardware async K copies and at most the proven ring2 resource footprint|
|Controls|Compile original source;require executable PTX identity against captured production. Compile unchanged explicit stages1 source as negative control;require known215regs/80KiB/zero cp.async and spills. Then compile its stages2 variant|
|Build|CPU ASTSource compilation from Python source with explicit CUDA SM90 target;captured function ABI/specialization attributes and original options;no launcher or driver resource queries|
|Validation|CPU tests,unchanged-math source AST proof,control identities,final typed K-copy order/MMA/ring/shared/register/spill/synchronization checks|
|Benchmark|None this round;no GPU kernel execution and no performance claim|
|Timeout|300s per isolated compile child;three compilations maximum|
|Profiler|Existing IR/PTX and PTXAS resource evidence;no hardware counter claim|
|Autotune|Private single fixed-profile diagnostic exemption;no tuning search/config changes|
|Fallback|NO_TORCH_COMPUTE_FALLBACK;unsupported ABI/compiler/source scope fails explicitly|

|Contract|Preserved/limited|
|---|---|
|Math/API|Original production source control;candidate reuses prior explicit-lookahead source with only num_stages1->2 and common function-name normalization. Addresses,masks,FP32 serial dots,head ReLU/sum and stores unchanged|
|ABI|Recover runtime argument names/types/divisibility from validated IR header. Only omitted runtime argument permitted is stride_table_page=1. Derive PAGE_SIZE/H/D/FP8 constants from frozen input metadata;all block/launch options captured|
|Safety|Original explicit lookahead masks final speculative tile. Source equivalence to its numerically tested stages1 version does not replace new GPU correctness or lifetime validation|
|Compiler|Pinned FlagTree/PTXAS13.4 identity. No installed compiler patch,backend method replacement,IR override or captured transformed IR compilation|
|Production|No export/registration/config update;no integration until compiler gate and subsequent numerical/performance/regression gates|

|Path|Gate/next action|
|---|---|
|Original source control|Canonical executable PTX,register/shared/MMA work must reproduce captured production,discarding debug paths only|
|Negative source control|Reproduce known register-carried stages1 result;otherwise stop and diagnose specialization/environment|
|Stages2 source|Require early typed K async copy in IR and final PTX,two K slots,same MMA/wait work,shared<=106496,regs<=217,zero spills,and checked final PTX copy/WGMMA/CTA drain protocol|
|Reject|Completed compile with unmet feasibility conditions records not_promising and no GPU execution;compile/provenance failure is nonzero. Print all resources and reasons;upload artifacts on either outcome|
|Promising|Record pending lifetime/accuracy;do not auto-run a new source kernel before inspecting its newly lowered ring/token dataflow|

## Local validation

All45 applicable CPU checks pass using the bundled Python runtime,including
five new source/ABI/feasibility tests and the prior516 actual-AST address/mask/
loop-carry simulator scenarios. Initial system Python lacked numpy;the bundled
runtime supplies it and the unchanged simulator passes. Both source variants
are AST-identical except loop stage1->2 after function-name normalization.
The generated production control body equals the production AST exactly.
The real fd83dd02 R64 captured header/options/input metadata prepare all three
source jobs with27 signature entries and original constexpr/attribute bindings.
CLI help,shell syntax and changed-file repository hooks pass locally. There
is no local CUDA compilation;on-card control reproduction and candidate
feasibility remain pending. No production/config/reference/timer change.
