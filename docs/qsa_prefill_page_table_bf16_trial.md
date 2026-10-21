# QSA BF16-only page-table fetch trial, 2026-10-21

Pre-coding plan. Follow-up to report9dd9767: all12 upstream BF16 cases
improved in both rounds, while several FP8 score cases regressed. This trial
changes only the dtype scope of the previous private page-table candidate.
Production stays source SHA256
f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
`optimization.md` and `deep_opt.md` are absent from this checkout; the existing
workflow, measured stage/config/resource evidence and keep/split decision apply.

## Project truth

| Field | Frozen choice |
| --- | --- |
| Op/reference | qsa_select_paged_prefill, existing pinned native vLLM QSA source; no corresponding single torch API |
| Editable | New private BF16 kernel, shared factory/benchmark selection and labeling,8 candidate GPU gates, CPU contracts and docs |
| Read-only | Production source/exports/YAML/TopK, previous unrestricted kernel and GPU gates,30 workloads, reference/tolerances, timers, GPU guards |
| Build | Python source only; local syntax/CPU tests/pre-commit; real JIT compilation on H800 |
| Hypothesis | Preserve measured BF16 scalar-page-ID gains and avoid the same lookup edit for FP8, without case-name dispatch |
| Active set/metric | Same30 scored-prefill cases, native and actual current-production baselines; kernel mode, us, baseline/candidate ratio; pipeline primary, equal-weight arithmetic mean >1.1 with all cases >=1 or near1 (proposed0.98 floor); score/TopK reported |
| Autotune | Original rectangular ten/small compact six configs and YAML groups, exact original keys/strategy/entry class; no new performance parameter or key. Independent caches; actual selected configs recorded, not forced |
| Validation |48 exact iteration nodes (40 production+8 candidate), full profile retained and required before integration; CPU actual-branch/AST proofs |
| Runtime/timeouts | Same FlagTree+ptxas13.4+CUDA13.4 compatibility runtime; two alternating rounds, warmup200ms/rep100ms, initial80C; case1800s, accuracy7200s, cooldown900s/10s |
| Profiler | Existing collected compiled resources/static PTX only; no claim of hardware-counter evidence or memory bottleneck |
| Fallback/integration | No Torch compute fallback, allocations/preprocessing/assembly/backend policy; private experiment only, no public export changes |

## Semantic contract

| Field | Unchanged behavior |
| --- | --- |
| Inputs/dtypes | Existing BF16 and FP8 e4m3fn Q/cache contract, H/D parameters, page sizes, padded strides/nonzero query base, chunks and runtime bounds |
| Math/output | FP32 dot accumulation, per-head ReLU then real-head sum, current int32 selection/-1 padding, out buffer and None return |
| Special values/unsupported | Existing masked/invalid-page handling and numeric/tie/tolerance rules, existing validation errors, forward-only; no new NaN/Inf/autograd promise |
| Host/API | Same original launcher/public bytecode, grid, dispatch, workspaces and persistent_topk identity |
| Torch use | Test/reference only; production and private candidate computation remain Triton |

## Paths and gates

| Path | Trigger | Implementation/config | Test/benchmark and risk |
| --- | --- | --- | --- |
| Scalar lookup | not IS_FP8 and BLOCK_N<=PAGE_SIZE | Same previous first/optional-second scalar page-ID loads, broadcast; frozen ten configs | BF16 page64/196 boundaries/stride2; same30 workload and current-production A/B. Gains may include independent autotune selection |
| Vector lookup | IS_FP8 or BLOCK_N>PAGE_SIZE | Exact original vector load and remaining original arithmetic; constexpr gate removes scalar branch for FP8 | FP8 page16/64/196 plus full-entry, BF16 small pages, CPU AST restores exact old function; compiler resources still measured |
| Compact/empty/select-all | Existing host eligibility | Existing compact kernel/early returns, unchanged configuration | Production40 coverage plus unchanged30 set; no benefit attributed to unchanged compact cases |
| Keep/split/revert | Measured result only | No integration until full validation and same-runtime data | Avoid universal FP8 rollout and case-name overfitting; unrestricted prior trial remains reproducible |

## Local validation and delivery

214 CPU contract/orchestration tests ran:213 passed,1 skipped. This includes
inherited actual lookup/mask execution over both dtype gates, page sizes and
strides; exact original AST outside the load; exact previous BF16 branch; host
bytecode/TopK identity, independent aligned entry caches, candidate acceptance,
CLI and48-node selection checks. The full-profile gate tests also passed after
adding BF16-only coverage/missing-coverage cases (14 tests).
All tracked production/benchmark files, original GPU tests and the prior
unrestricted/packed kernel sources were compared byte-for-byte to HEAD.
Changed/new Python syntax, diff whitespace and changed-file pre-commit passed.
No local Torch/Triton/CUDA is available. H800 compilation/numerics, selected
configurations, latency and keep/revert are pending; no speedup is claimed.
The command uses a fresh checkout and automatic report upload in a child shell.
