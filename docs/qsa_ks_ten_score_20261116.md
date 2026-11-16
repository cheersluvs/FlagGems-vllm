# Ten-workload whole-score diagnostic

## Project truth

| Field | Contract |
| --- | --- |
| Task | Extend fused logical KS candidate to all ten migrated INT8 workloads, all requests/chunks |
| Frozen | Native query16 C++, arithmetic/NZ packing/output expressions, existing integrated INT8 score adapter, production/TopK/harness/compiler |
| Editable | New suite planner/driver/tests, suite mode in raw runner |
| Inputs | Existing qsa_int8_ten_workloads_910b fixtures unchanged; H4/D128/page192/FP16 scales/shared per-request map |
| Dispatch | Current adapter's global chunk schedule; candidate splits crossings at request boundaries and charges every piece |
| Build | Same isolated task-mode fix/native bitcode; generated source changes only key-bound annotations to runtime scalars |
| Runtime specialization | TILES/RAW_TILES/MAX_VISIBLE become scalar parameters; ROWS/width/query16 remain constexpr; pointer formulas unchanged |
| Workspace | Reused case-sized scratch, metadata-only views per job, no full-logits materialization; no K/KS preparation cache across chunks |
| Validation | Every row/column for every chunk: full vs logical exact, both vs current1e-3/mask; K/KS exact; independent first/last-row CPU oracles; repeat all after timing |
| Timing | Entire score sweep over all requests/chunks, two reversed rounds/six active; all QW/K/Cube/output tasks charged and exact host/device task counts checked |
| Aggregation | Per-case device sums/round means/median and arithmetic mean of per-case ratios; diagnostic, not formal harness acceptance |
| Hardware | No repeated hardware counters in this coverage run; retain source/IR and representative compiled artifacts |
| Timeout | Worker7200s and automatic report upload; failures retained per case, no invalid-case averaging |
| Autotune/fallback | Diagnostic only; no production config/registration change; zero-visible candidate pieces use existing INT8 score kernel, never torch compute |

## Contract

| Field | Contract |
| --- | --- |
| Score | Frozen INT8 QK/FP16 two-Cube raw candidate, FP32 output and -inf mask; current INT8 reference tolerance1e-3 |
| Full/logical pair | Same scheduling/workspace/runtime bounds; only per-piece K vs fused K/KS preparation and full-block scale access differ |
| Mixed requests | Candidate never shares a request's mapping with another; global target offset preserved across splits |
| Empty visibility | No zero-tile raw allocation/launch; call current score on that exact slice and charge the task |
| Precondition | Validate scales/coefficient range/page IDs; invalid active mappings fail, no silent route substitution |
| Timing preparation | Host metadata/allocation outside device timing, same prepared convention as previous probes; each GPU packing launch remains inside timing |
| Precision/unsupported | Forward-only H4/D128; finite positive FP16 scales, finite weights; reject unsupported or numeric mismatches |

## Paths

| Arm | Definition |
| --- | --- |
| current | Existing PreparedSharedMapInt8Selection over its unchanged global chunks |
| full1024 | All request pieces, QW+K+native16+physical scale full/tail output |
| logical1024 | All request pieces, QW+fused K/KS+native16+logical scale full/physical tail output |
| Validity | Full coverage required, pre/post numeric checks and exact task accounting; no performance inferred from one representative chunk |

The runtime-bound launch form is new and will be revalidated on device; it is
not claimed to have identical lowering/latency to earlier constexpr chunk
probes. This suite times score_all behavior, including chunks that an overall
selection operator could bypass. It does not include TopK, CANN V1 timing,
model inference or the formal FlagGems-vllm harness. No production change.

Local validation: 64 CPU tests passed (31 probe, 24 raw, 5 core-mode,
4 workload-suite tests), plus CLI mode guard, Python/Bash syntax and focused
pre-commit hooks. Device compilation, correctness and timing remain pending.
