<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 pipeline-depth probe on the integrated head-group baseline

Pre-code boards,2026-10-24. Production report78c3ca4f passed49 representative
correctness checks and native30 measurements. Grouped R64/W4 records use207
registers/shared106496/zero spills; original R32/W8 uses63/shared73728. Production
BF16 choices all use num_stages2; FP8 already tunes pipeline depth. This probe
asks whether reducing only pipeline depth helps before expanding autotuning.
No fresh NCU collection: current compiled resources and measured stage ratios
are the starting evidence; register/shared changes and A/A are recorded here.

|Project truth|Frozen decision|
|---|---|
|Baseline|Actual integrated production0d7e80e2/code314a9de, exact fixtures/configs from78c3ca4f|
|Editable|New probe/manifest/runner/CPU tests/docs; generic base coordinator baseline-commit check only|
|Read-only|Production/JITs/YAML/TopK/benchmark fixtures/timers/guards/tolerances|
|Hypothesis|Stage1 can reduce shared-memory pressure for current BF16 scorer, including grouped heads, without changing tiling or arithmetic|
|Probe|Five BF16 cases: four upstream long-prefill pipeline laggards plus observed_chunk_boundary; FP8 long100k unchanged negative control|
|Build/validation|GPU JIT; three all-row17-row boundary/canary groups covering BF16 R32 original, BF16 R64 grouped, FP8 fallback; sampled real pre/post score/selection checks|
|Benchmark|kernel mode, two paired alternating rounds; score/TopK/pipeline plus four A/A controls|
|Aggregation|Per-case current stage2 / stage1 ratios; arithmetic means separately for five changed BF16 and unchanged FP8 control; no native acceptance claim|
|Autotune|Fixed-choice exemption: copy actual per-chunk production choices; change only num_stages2→1 on rectangular BF16. Other configs and algorithms fixed. Not a full22-choice joint autotune experiment|
|Timeout/thermal|1800s worker,200ms warmup/100ms repeat,80C initial limit/120s cooldown|
|Next gate|If stable benefit exceeds A/A, investigate adding stage1 to production's joint search; no immediate production integration|

|Contract|Preserved|
|---|---|
|Input/output/math|Same frozen BF16/FP8 tensors, pages, masks, chunks, outputs and head-group flags; kernel arithmetic unchanged|
|TopK|Same callable and common buffers; reported but not optimized|
|Torch/autograd|Inference only; test references only, no production torch fallback or changes|
|Limitations|Large cases sampled; small fixtures all rows checked. No full production pytest claim|

|Path|Treatment|
|---|---|
|BF16 rectangular|Identical current JIT on both sides; candidate changes num_stages only, preserving HEAD_GROUP=false/true and every other config field|
|FP8/compact|Original configs/JIT, exact PTX/cubin equality required for unchanged chunks|
|Mixed chunks|Freeze per first-seen row geometry within route from captured public observer; preserve mixed R32/full-head and R64/grouped choices|
|Coordinator|Reuse established workers/guards; require captured report's manifest baseline_code_commit rather than hardcoding the historical first integration|

Local verification:38 related CPU checks passed, including frozen mixed-tail
configuration mapping, stage-only copies with unchanged original traces,
actual factory JIT identity and independent globals/config maps, rejection of
extra config changes or changed FP8 binaries, grouped/fallback boundary evidence,
CLI child-script handoff/restoration, separate control aggregation, and rejection
of a baseline report from the old integration commit before toolchain loading.
Pre-commit, CLI help and shell syntax pass. GPU compilation/numerics/resources/
timing are pending. Production SHA0d7e80e2 and YAML remain unchanged. The runner
uploads JSON/TXT/TSV automatically on success or test failure, including compiled
PTX/TTGIR files and per-case stage timing tables. Summary is stage-depth-summary.json.
