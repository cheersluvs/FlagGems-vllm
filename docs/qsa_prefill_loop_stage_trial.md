# Single-stage score pipeline trial

## Project truth (before implementation)

| Field | Frozen choice |
| --- | --- |
| Current op/reference | Existing qsa_select_paged_prefill; pinned native vLLM7a877ae and canonical production SHA256 f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c; no single Torch API |
| Editable | New private stage-config factory,YAML group,GPU/CPU tests,runner,this document; optional candidate/profile/source/archive routing and combined previous-baseline identity |
| Read-only | Production/export/TopK,kernel math and all existing kernels/factories/YAML groups;30 workload definitions; original timer,guard,reference,tolerances and validation helpers |
| Build | Python syntax and pre-commit locally; existing FlagTree JIT on H800, no package rebuild/compiler installation |
| Validation |62-node iteration profile:40 canonical,14 existing mode/page boundary nodes rebound to the new factory,and8 every-config H/D nodes; all must pass. Full canonical +40 candidate profile remains available before integration |
| Benchmark | Same verified FlagTree/ptxas13.4/CUDA13.4compat,--mode kernel; score,TopK,sequential pipeline us and baseline/candidate ratios |
| Active/aggregation | Frozen30 scored-prefill cases; equal-weight arithmetic mean of30 native pipeline ratios>1.1,every case near1 (proposed floor0.98); experimental,not production acceptance |
| Same-run controls | Keep native and canonical score/pipeline comparisons; add original combined stage2 score/pipeline using existing previous-tuning machinery and separate buffers; keep existing fixed query-mode and short_batch3 swap rows |
| Timeout/runtime |2 rounds,200ms warmup/100ms rep,initial80C; case1800s,accuracy7200s,cooldown900s/10s; automatic partial/complete upload |
| Autotune | New qsa_select_paged_prefill_loop_stage_score group: exact17 original joint entries with only launch num_stages2->1; existing key/strategy/pruner,10 BF16/17 FP8 choices; compact6 unchanged |
| Profiler | Prior compiled resources: BF16 TR64/BN64/KT16/W4 uses240 registers/106496 shared bytes; FP8 split equivalent144 registers/57344 bytes. H800 resource records/configs collected again; no hardware counters locally,resource pressure is a hypothesis,not proven occupancy bottleneck |
| Fallback | No Torch compute fallback,new workspace,asm,policy or TopK modification |

## Semantic contract

| Field | Frozen behavior |
| --- | --- |
| Inputs | Existing BF16/FP8 e4m3fn,H/D,query/cache/page-table strides,integer metadata,query bases,chunks,visible bounds and invalid pages |
| Outputs | Same int32 indices,-1 padding,caller buffers and None return |
| Math/promotion | Exact unchanged combined full-D FP32 dot,per-head ReLU then real-head sum; no reduction split,scaling or tolerance change |
| Layout/alias/boundaries | Exact canonical host bytecode and dispatch; existing early returns,compact route and caller buffer rules |
| Scope | Forward only; no new autograd,special-value,dtype or unsupported guarantees |
| Torch | Existing test/benchmark reference only; no production import of trial |

## Paths

| Path | Trigger | Implementation/config | Validation/risk |
| --- | --- | --- | --- |
| Rectangular | Canonical routing | Exact audited combined kernel,independent entry/cache; new stage1 YAML group | All10 BF16/17 FP8 choices,H/D,modes,page and chunk tails; stage1 may lose memory/computation overlap |
| Compact | Canonical ragged routing | Exact canonical kernel and6 configs | Existing candidate compact checks/full-entry path,unchanged stage2 |
| Early/unsupported | Canonical metadata gates | Exact canonical host | Existing iteration/full contract checks |
| Previous baseline | Explicit combined previous-tuning choice | Original pinned combined kernel and original17 stage2 YAML choices | Same TopK,independent rectangular/compact caches,own output/logits buffers,source/config records |
| Report | New explicit candidate | Existing timing/validation/telemetry plus stage1/config/source metadata | Native acceptance unchanged; do not label experimental result production |

Only software-pipeline depth changes in this trial. The matrix geometry,query
mode set,entry key,host dispatch and kernel source stay fixed. The separate
config group prevents editing historical or production choices. Pipeline depth
is a compiler scheduling control,not a CUDA-specific primitive;benefit and
compiler interpretation must be verified separately on each backend.

The configuration uses triton.Config.num_stages,not a new tl.range attribute.
See [official Config documentation](https://triton-lang.org/main/python-api/generated/triton.Config.html)
and [range documentation](https://triton-lang.org/main/python-api/generated/triton.language.range.html)
for that distinction. Lower depth can reduce buffering but also overlap; no
performance or occupancy gain is assumed before the H800 result. The existing
compiled registers/shared/spill/static-site records provide the first check of
whether scheduling actually changed. optimization.md/deep_opt.md are absent in
this checkout; this board records the deep-optimization hypothesis and audit.

## Implementation and delivery

The factory calls the original combined factory,then replaces only its
rectangular entry with an independent LibTuner/entry using the new YAML group.
It binds the exact same JIT function,pruner,key and host globals. The compact
entry remains independently cloned by the original combined factory. Production
exports and every existing kernel/group remain unchanged. Runtime preparation
rejects a config count or any non-stage difference,checks the combined kernel
SHA256,records the actual stage1 choices and selected compiled configs,and
requires identical metadata after timing. No new compute implementation or
Torch fallback is introduced.

Explicit candidate name is query-loop-stage. The runner requires the same-run
previous baseline combined,using an independent original combined module with
its own captured logits/output banks. score_vs_previous_tuning and
pipeline_vs_previous_tuning carry baseline_kind=combined_query_page_stage2.
Their original validation/timer/AB-BA/guard logic is reused unchanged. Existing
native score/TopK/pipeline,canonical score/pipeline,selected FP8 mode replay and
short_batch3 swapped comparison remain. All sides use the verified same compiler
profile. The report column says gems_loop_stage_trial_us; it does not label the
private candidate as production.

The default62-node correctness profile includes every new rectangular config
for BF16/FP8 and H/D4/128,3/24,5/96,1/8,alongside forced mode/page/chunk
boundaries and full-entry checks. The inherited fixed-score helpers normally
request stage2; the new test wrappers explicitly override that request to stage1.
This prevents boundary checks from accidentally testing the previous scheduling.
Full canonical plus40 private-candidate correctness is available with
--accuracy-profile full before any production integration. Iteration results
remain labeled full_suite_validated=false and formal_acceptance_eligible=false.

The default entry is:

```bash
bash tools/run_qsa_prefill_loop_stage_h800.sh --flagtree /opt/flagtree
```

Use the existing QSA vLLM0.31 activation and QSA_STOCK_PYTHON. The existing parent
runner uploads complete/partial reports automatically under flagtree134-loop-stage-*.
Invoke in a child shell so failure preserves the interactive container. No
self-control cycles or extra benchmark shapes are added. Timing remains2 rounds,
200ms warmup/100ms rep,initial80C;compilation/autotune remain untimed. The report
retains TopK data although TopK is outside this optimization's scope.

The deep-optimization decision is pending: compare same-run stage1 against
combined stage2,inspect registers/shared/spills and selected geometry,and check
all30 native sequential pipeline ratios. Independently tuned tile choices can
differ between stage1/stage2;these measurements evaluate the scheduling variant
with its tuner,not a claim of isolated instruction-level causality. Revert if
there is no stable gain or important shapes regress. Keep all original timings,
including any renewed short-batch anomaly. No H800 result is claimed at delivery.

Local validation:103 related CPU checks passed,including6 new stage-config,
factory/cache/source,previous-baseline,acceptance,coverage/CLI tests. Python and
shell syntax,changed-file pre-commit and diff checks passed. Source-byte checks
confirm canonical production,combined kernel/factory,timer,guard,reference and
workloads are unchanged; a normalized YAML pin confirms every pre-existing
config group is unchanged. No local Torch/Triton/CUDA is available. Actual
stage1 JIT,numerics,compiled resources,selected configurations and30-case
performance are pending H800 execution. Status: private deep-optimization trial,
not production integration or a claimed performance gain.
