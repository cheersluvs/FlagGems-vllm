# Hygon prefill final-network production trial

`optimization.md` and `deep_opt.md` are absent in this checkout. The public
operator is already exported and registered. This trial changes only the Hygon
override's dense dispatch and a new, separately generated kernel module.

## Project facts frozen before implementation

| Field | Decision |
| --- | --- |
| Current op | `flaggems_vllm.top_k_per_row_prefill`; `torch.topk` is the value oracle |
| Editable | Hygon fused override, its private final helper/source builder, trial tools and this plan |
| Read-only | Generic op, existing functional tests, official benchmark and vLLM baseline |
| Build | Source parse and generated-module source check locally; Triton JIT on BW1000 |
| Validation | Existing test file plus public-route adversarial cases, exact value multiset, bounds, uniqueness and guards |
| Benchmark | Official `benchmark/test_top_k_per_row_prefill.py --mode kernel`, vLLM C++ baseline, seven shapes; SpeedUp higher is better |
| Active set | fp32, column contiguous, row padding permitted, seven shipped shapes; network targets shape IDs 2, 4 and 5 |
| Aggregation | Per-shape paired off/on/on/off; geomean of the seven benchmark SpeedUps after correctness passes |
| Timeout | Each test/benchmark subprocess bounded; outer runner 14400 seconds |
| Profiler | Existing exact-device probe and official benchmark; rocprof is available on BW1000 but not required for this single-path trial |
| Autotune | Exempt: fixed Hygon path with existing geometry and top-k=512; NVIDIA routes unchanged |
| Fallback | `FLAGGEMS_HYGON_TOPK_FINAL_NETWORK=0` or source/import failure selects current dense VEC2; candidate count >256 uses original scalar selector |

## Input/output and path contract

| Field | Contract |
| --- | --- |
| Device/type | Hygon HIP wave64, fp32 logits, int32 starts/ends/output; forward only |
| Layout | Column-contiguous logits with arbitrary row stride; valid `[start,end)` per row |
| Output | Relative distinct indices, exact top-k value multiset; short rows padded with -1 |
| Special values | Ties, signed zeros and infinities preserved; NaNs remain outside this measured path's contract |
| Reference-only torch | Input generation and `torch.topk` in tests/tools only; production uses metadata and allocation |

| Path | Trigger | Implementation | Risk |
| --- | --- | --- | --- |
| Network dense | enabled, non-TLE, rows >=8192, top-k=512, 2048<=vocab<=5120, dense VEC2 available | Generated VEC2 copy with final network for candidate count<=64/128/256; original scalar overflow | Additional JIT module, small gains |
| Short bins | top-k=512 and vocab<=1536 | Existing 512-bin route | Must remain first in dispatch |
| Other dense/sparse | All remaining legal inputs | Current production routes | None from this trial |
| Unsupported | Existing generic operator contract | Existing behavior | Not expanded |

Previous direct/cached probes passed exact value checks. Cached full-call
ratios were about 1.04 on 12961x4100 and 1.08–1.11 on 16380x5115; the latter
host timings showed interference, so the official paired benchmark decides
keep or revert. The 16383x4095 device data were noisy. No benefit is presumed.

Run local source checks with
`python3 tools/hygon_prefill_final_production_check.py` and code-style checks.
On the dedicated Hygon audit worktree, run
`tools/hygon_prefill_next_run.sh final_production hygon_prefill_final_production_v1`.
The stage verifies on/off routing and the vLLM C++ baseline, validates the
public operator on normal/padded/tied/constant/partial/short/special inputs,
runs the existing quick functional suite, then runs the unchanged official
seven-shape benchmark in off/on/on/off subprocess order. Each subprocess has
a timeout and the full report is committed and pushed even on failure.
