# Private score-loop unroll trial

Pre-coding scope,2026-10-22. The audited joint-stage candidate report e861c93c
has pipeline/native mean1.183323 but five cases below0.98. Hypothesis: unroll
adjacent key iterations by two to expose load/dot scheduling opportunities,
without changing arithmetic within an iteration or query/key reuse semantics.
This is a portable Triton loop transform; performance remains backend dependent.

| Project truth | Frozen decision |
| --- | --- |
| Op/editable | Private loop-unroll kernel/factory/tests/runner,explicit benchmark routing and source archive,accuracy selector and this document |
| Read-only | Canonical production,combined and stage-joint factories/kernels/configs,TopK,workloads,pinned reference,timers,guards,runtime profile |
| Build | Python/shell syntax,CPU regression tests,changed-file pre-commit; H800 JIT through existing FlagTree runner |
| Validation | Existing40 canonical iteration nodes +22 candidate nodes; every34/20 dtype-pruned config,both depths/query modes,page/chunk boundaries; full before integration |
| Benchmark | Frozen30 scored-prefill cases,kernel mode,2 rounds,200ms warmup/100ms repetition; native score/TopK/pipeline plus same-run stage-joint score/pipeline |
| Acceptance | Arithmetic mean native/pipeline >1.1,each >=1 or proposed near-one floor0.98; retain all stage data,do not optimize TopK |
| Timeout/profiler | Existing1800s case/7200s accuracy/900s cooldown; compiled resource/PTX metadata,no local CUDA profiler |
| Autotune | Reuse exact stage-joint34 YAML choices,key,strategy/pruner; fixed unroll2 is the sole algorithmic alternative,not a new tunable parameter; no new config group |
| Fallback | No torch compute fallback; unsupported behavior unchanged |

| Contract | Decision |
| --- | --- |
| Reference | Existing pinned native vLLM QSA source7a877ae; not a standalone torch API |
| Input | Existing BF16/FP8,H/D,page and query/cache/table strides,request offsets,visible bounds,invalid pages |
| Output | Existing FP32 logits,unchanged in-place index output and alias contract |
| Arithmetic | Full-D FP32 dot accumulation,ReLU before head sum,unscaled; no reassociation between keys |
| Boundaries | Dynamic loop bounds/tails and all masks unchanged; tl.range handles remaining single iteration |
| Autograd/unsupported | Forward-only; existing unsupported validation retained |
| Torch | Reference computation only in tests; production clone uses only existing allowed metadata/allocation/view paths |

| Path | Trigger | Implementation/config | Verification/risk |
| --- | --- | --- | --- |
| Empty/select-all | Existing metadata gates | Unchanged host/TopK | Existing canonical checks |
| Rectangular | Existing rectangular dispatch | Exact combined scorer,only range -> tl.range(loop_unroll_factor=2),same joint-stage34 choices | Both depths/modes/every choice,page/loop tails; register/code size growth or compiler scheduling regression possible |
| Compact | Existing compact dispatch | Unchanged canonical compact scorer | Existing compact/full-entry checks |
| Unsupported | Existing contract | Same explicit errors | No fallback introduced |

Keep/revert is pending same-run H800 correctness and performance. A superset
of configurations does not guarantee improvement. The previous stage-joint
baseline is separately instantiated with original source and independent outer
entry/output buffers; shared LibTuner per-configuration caching behavior is not
changed. Production exports remain canonical. Full candidate correctness and
production-entry validation are required before integration.

The transform uses the documented portable `tl.range` loop-unroll attribute:
https://triton-lang.org/main/python-api/generated/triton.language.range.html .
It does not set loop-level num_stages (which differs from the existing kernel
launch parameter). The old loop body is AST-identical,including its stores,
page lookup and split-query math. Existing every-config tests use visible1537
with BLOCK_N64/128,so the dynamic last loop group includes odd iteration counts
(e.g.25 key tiles at BLOCK_N64); output sentinels verify masked columns stay
untouched. Both launch depths and both eligible FP8 query modes are forced.

Local validation:95 related CPU regression checks passed,including seven new
loop-transform/factory/baseline/acceptance/coverage/report/CLI checks. Both shell
runners pass bash -n. Changed-file pre-commit and diff checks pass. AST checks
confirm the old loop body and entire GPU-test logic are retained; baseline
production,combined kernel,stage-joint factory and tuning YAML match the parent
byte-for-byte. Core geometry/numerics/environment/timer/guard helpers are
unchanged. Reports archive all new implementation and test sources. No local
Torch/Triton/CUDA execution is available: H800 compilation,numerical correctness,
register/shared/spill changes and speedups are pending. Status: private trial;
no production integration or performance gain is claimed.
