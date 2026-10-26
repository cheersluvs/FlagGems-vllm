<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Active benchmark scope:FP8 only

Pre-change boards,2026-10-26. User narrows future optimization to FP8 and removes
BF16 from the active benchmark. Retain the existing ten FP8 workload records
exactly;do not synthesize more shapes or carry over BF16 candidate gains.

|Project truth|Decision|
|---|---|
|Editable|Pytest benchmark matrix/dtype/default description,current kernel-mode CLI default,two existing matrix checks,this document|
|Read-only|Production kernels/configs/exports,all correctness tests/references,FP8 workload definitions,old workload suites/captured manifests,measurement engine and GPU guards|
|Active set|Ten existing fp8-prefill cases;zero BF16 and zero select-all performance cases|
|Validation|Existing CPU matrix/dtype/input-route checks plus CLI --list-cases;format/shell/source checks;no GPU run required to select existing fixtures|
|Benchmark metric|NV acceptance remains score-only kernel-mode native-vLLM/production ratio;continue reporting TopK and full pipeline timings/ratios separately|
|Autotune/fallback|No kernel or tuning change;existing libtuner and unsupported behavior preserved;no Torch compute fallback introduced|

|Contract|Unchanged|
|---|---|
|Inputs|FP8 E4M3FN query/cache,H4/D128,page196,ratio4,k512;same exact query lengths,prefixes,cache/table capacities,strides,visible construction and512MiB chunk budget|
|Outputs/math|FP32 scoring/accumulation and INT32 block indices;existing TopK unchanged|
|Functional coverage|BF16 public support and BF16/FP8 correctness tests remain;FP8 select-all cases remain correctness-only|
|Comparability|Ten-case FP8 aggregate is a new active-set summary;do not compare it directly to previous30-case averages|

|Path|Action|
|---|---|
|Pytest benchmark|All full/score/TopK comparisons use fp8-prefill and FP8-only dtype declaration|
|Current H800 runner|benchmark_qsa_prefill_triton_stages.py defaults to fp8-prefill;explicit suite flags retain their meaning|
|Historical diagnostics|Old BF16/combined suites and manifests remain for report replay;they are not the active optimization matrix. Stop BF16 source-lookahead/producer-first work pending no further BF16 instruction|
|Next optimization|Start from current FP8 production profiling;BF16 R64 gains do not establish FP8 gains|

Validation:30 existing CPU workload/benchmark checks pass (two pre-existing
optional-dependency skips). The default CLI --list-cases returns exactly
fp8.cases():ten FP8 cases,byte-for-byte identical workload records. Existing
matrix checks confirm zero BF16 inputs and ten FP8 inputs for full/score/TopK
benchmark classes. Historical explicit-suite geometry checks still pass.
No GPU benchmark or new speedup claim;no production or correctness-test edit.
