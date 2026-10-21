# QSA packed request/query grid trial, 2026-10-21

Pre-coding plan. Current best is clean production1aeb7aa (documented by
f323659), source SHA256 f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c.
Report1d9ccf7 validates cache alignment but still has12 pipeline ratios below0.98.

## Project truth

| Field | Choice |
| --- | --- |
| Editable | Private packed-grid candidate/kernel/launcher, benchmark selection/trace and same-current-production pipeline comparison, focused CPU/GPU tests, documentation |
| Read-only | All production sources, LibEntry/LibTuner, tuning YAML, native reference,30 fixtures, timer, numerical tolerances and persistent_topk |
| Build/test | CPU syntax/contracts/actual scheduler coverage and pre-commit locally; full production accuracy plus candidate BF16/FP8 gates on H800 |
| API/reference | Existing QSA API; native commit7a877ae34dcf4a7d46de08eb16db4699e1e0047c; existing dtype/layout/metadata contract |
| Active metric | Same30 scored-prefill cases; kernel-mode score, TopK and sequential pipeline versus native, plus score/pipeline versus current production |
| Goal | Pipeline per-case >=1 or near1 (report floor0.98); equal-weight arithmetic mean >1.1; experimental result explicitly distinct from formal production acceptance |
| Toolchain | Same FlagTree, ptxas13.4 and CUDA13.4 compat for all sides, two alternating rounds, warmup200ms/rep100ms,80C idle limit and existing PID/clock guards |
| Profiling evidence | Prior actual tile choices and chunk lengths; useful/empty query-program counts; new per-chunk compiled config/resources and route trace; no hardware-counter claims |
| Hypothesis | Avoid empty request/query programs in partially covered multi-request chunks, without changing useful per-request query tiles or dot orientation |
| Fallback | Existing compact/select-all/empty/rectangular paths; no Torch compute fallback, no host reads of tensor values, no new workspace or GPU preprocessing launch |

## Semantic contract

| Field | Preserved behavior |
| --- | --- |
| Dtype/layout | BF16/FP8 H/D and nonoverlapping padded/strided Q/cache/table, nonzero query base |
| Score | Exact current rectangular arithmetic from HEADS_PADDED onward: keys x query, FP32 dot result, ReLU per real head, reduction and masks |
| Request boundaries | Binary search monotone clipped-start slots, including empty requests and request/chunk boundaries; no query duplication or omission |
| Output/TopK | Same int32 output and -1 padding, ties/tolerances, visible lengths and persistent_topk object |
| Host | Same validation/workspaces/chunks/order/public function; only private score-launch routing and grid change |
| Unsupported | Same inference-only/resource/device bounds; no new semantic relaxation |

## Paths and evidence

| Path | Trial | Gate |
| --- | --- | --- |
| Existing compact | Unchanged, retains priority | Same compact gate/body/configs; isolated outer caches |
| Sparse rectangular grid | Monotone-slot scheduler with exact current rectangular body/configs | Metadata capacity >=2x chunk rows; conservative TILE_R64 guard must predict fewer query slots; no shape/dtype list |
| Other rectangular | Isolated clone of current production | Same cache alignment, exact JIT/body/configs and key strategy |
| Packed mapping | At most floor(rows/TILE_R)+requests slots, binary search clipped starts | CPU execute actual mapping across all32 fixtures and randomized empty/nonzero-base/boundary cases; on-card full-row FP32 and native checks |
| Cache/autotune | Same10 rectangular and6 compact configs, original keys/default strategy; inherit current local aligned entry | Independent candidate/production entries and actual per-chunk resources; no manual fixed config |
| Comparison | Separate baseline output/workspace and actual public chunk traces | Current production source pinned, score/TopK/pipeline before/after checks, same-worker production pipeline A/B |
| Keep/revert | Production remains unchanged until report | Keep only validated targeted benefit without material active-set regression; reject broad conclusions from sub-percent differences |

The TILE_R64 guard is the maximum frozen rectangular tile, not a hardware
constant. Static config checks enforce that bound. Exact counts for each chosen
TILE_R are recorded outside timing; they count issued query slots, not executed
loads or FLOPs. Existing compact scheduling already pays binary-search overhead;
this trial independently tests that mapping with current key-major arithmetic
on multi-request chunks previously excluded by the compact gate.

## Frozen-route forecast and validation

Forecast below uses report1d9ccf7 chosen TILE_R values. The candidate will
autotune on its actual grid; these are predicted query-slot counts, not a
latency forecast. Existing compact routes retain priority.

| Case | Chunk offset | Rows | Prior TILE_R | Rectangular query slots | Packed query slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| perf_n2_q8192_p51200_bfloat16 | 9039 | 7345 | 32 | 460 | 231 |
| perf_n2_q8192_p102400_bfloat16 | 0 | 4854 | 32 | 304 | 153 |
| perf_n2_q8192_p102400_bfloat16 | 4854 | 4854 | 32 | 304 | 153 |
| perf_n2_q8192_p102400_bfloat16 | 9708 | 4854 | 32 | 304 | 153 |
| perf_n2_q8192_p102400_bfloat16 | 14562 | 1822 | 32 | 114 | 58 |
| perf_n4_q4096_p51200_bfloat16 | 9709 | 6675 | 32 | 512 | 212 |
| perf_n4_q4096_p102400_bfloat16 | 0 | 5041 | 32 | 512 | 161 |
| perf_n4_q4096_p102400_bfloat16 | 5041 | 5041 | 32 | 512 | 161 |
| perf_n4_q4096_p102400_bfloat16 | 10082 | 5041 | 32 | 512 | 161 |
| perf_n4_q4096_p102400_bfloat16 | 15123 | 1261 | 64 | 80 | 23 |
| fp8_two_balanced_float8_e4m3fn | 16256 | 73 | 8 | 20 | 11 |
| fp8_two_skewed_float8_e4m3fn | 0 | 16304 | 32 | 1020 | 511 |
| fp8_two_skewed_float8_e4m3fn | 16304 | 80 | 8 | 20 | 12 |

All production files remain byte-for-byte unchanged; only private tools and
test/benchmark plumbing are edited. CPU tests compare exact kernel signature,
decorators and arithmetic AST, normalize the single extra launcher branch,
execute the actual scalar scheduler prefix for all32 diagnostic fixtures and
100 deterministic random boundary cases at TILE_R4/8/16/32/64, and check
isolated caches/current-source pin, diagnostics, CLI and experimental gates.

The on-card gate runs the full production suite plus8 candidate tests:6
BF16/FP8 full-row score checks across chunk/request/empty-request boundaries,
and2 actual full-entry native/FP32 comparisons with padded inputs and nonzero
query base. The30 benchmark cases retain before/after validation for all five
stages. Baseline/candidate output and score workspace pointers must differ.

Reports explicitly label the objective as experimental, never formal production
acceptance. They include packed route/config/PTX/CUBIN resources, issued/useful/
empty query slots and actual three-dimensional issued-program counts. The
useful-slot metric only accounts for request/query clipping; it does not count
visible-key early exits or executed FLOPs.

Local CPU results and pre-commit are recorded before delivery; H800 compilation,
correctness, performance and keep/revert remain pending.

Local validation completed: `PYTHONPATH=tools python3 -m unittest discover -s
tools -p 'test_qsa*.py'` ran198 tests (197 passed,1 skipped), changed-file
pre-commit passed, Python AST parsing and frozen-file byte comparisons passed.
No local Torch/Triton/CUDA installation is available, so no GPU result is
claimed. H800 delivery uses `--candidate packed-grid --suite scored-prefill
--mode kernel --rounds 2` and the existing automatic report-upload wrapper.
