# Initial INT8 kernel benchmark against native QuantLightningIndexer V1

## Frozen project truth before implementation

| Field | Choice |
| --- | --- |
| Task | First kernel-mode timing after ten-shape correctness report f7e26f9c |
| Editable | Optional debug-store gate in standalone scorer; new benchmark/runner and this document |
| Read-only | Production ops, TopK/loader, native interface, ten logical workloads and quantization generator |
| Build | Existing FlagTree JIT; preflight compiles all shapes/tails before timing |
| Validation | No-debug score IR check; full-row index structure/native pairing/masks; independent CPU reference on the same80 sampled rows |
| Benchmark | Existing qsa_kernel_benchmark_910b.setup and do_bench_npu; microseconds summed device-kernel duration per full logical invocation |
| Active set | Ten migrated NV logical workloads; INT8 H4 D128 page192 ratio4 K512; TopK calls<=32 |
| Aggregation | Per-case native_fused/combined; arithmetic mean only when all ten pass; no skipped-case acceptance |
| Repetition | Default one round, one warmup and three active iterations; no temperature gate; stability not established by one round |
| Timeout | 1800 seconds per case, isolated subprocesses and streaming logs |
| Profiler | Installed Ascend kernel timer source/hash recorded; required single/triple score scaling check |
| Autotune | Exempt initial standalone baseline measurement; unchanged N32/head16 score schedule and existing TopK block256 |
| Fallback | No Torch scoring/TopK fallback in candidate jobs; existing TopK device routes unchanged; no alternative native baseline |

## Contract before implementation

| Field | Choice |
| --- | --- |
| Input | Same stored INT8 Q/K, FP16 descales and unit weights for candidate and native; frozen generator/seed |
| Output | Unique/range-valid INT32 group indices, min(visible,512) valid entries and -1 padding |
| Arithmetic | INT32 dot -> FP32 dequantization/ReLU/head sum; no debug allocation/store in timed score |
| Accuracy | Existing rtol/atol1e-3 for scores/selected-score multisets; no equality requirement on tied/near-tied index identity |
| Reference scope | All-row candidate/native selected scores evaluated on frozen candidate logits; sampled CPU full scores/TopKs independent |
| Native boundary | Primary: one full V1 invocation with one native batch per query row; auxiliary: chunk128 (long100k112) from correctness; input packing excluded |
| Unsupported | Other dtypes/layouts/modes and gradients unsupported; only --mode kernel accepted |
| Torch use | Input generation, allocation and validation only in standalone harness; no Torch compute in timed candidate jobs |
| Memory | Benchmark-only full FP32 logits staging up to1856MiB; candidate combined uses bounded scratch; no full production staging |

## Paths before implementation

| Job | Implementation | Configuration | Verification | Meaning |
| --- | --- | --- | --- | --- |
| score | Validated score arithmetic, compile-time debug store disabled | Existing score chunk and N32 tile | Sampled CPU full scores before/after timing; IR has one output store | Isolated score duration, bounded scratch |
| topk | Existing persistent_topk | Every call<=32, block256 | Full-row range/uniqueness, sampled CPU and native pairing | Isolated TopK on benchmark-only frozen logits |
| combined | Score into scratch then existing TopK | Same score chunk, TopK32 | Full-row structure/pairing and sampled CPU | Actual two-stage device-kernel sum, measured directly |
| native_fused | One full native V1 call | Same values, per-row page table and full local Q cumulative lengths | Full-row structure/pairing and sampled CPU | Primary fused native baseline; split unavailable |
| native_chunked | Chunked native V1 calls | Prior correctness chunk policy | Same validation | Auxiliary diagnostic only; not used for primary speedup |

Isolated score and TopK timings are not added to manufacture a combined
measurement. Frozen-logit TopK and bounded-scratch combined have different
cache histories. Isolated and combined TopK use identical call boundaries,
including the long100k112-row score chunks. The report keeps all five
measured durations, and leaves
native split durations/stage speedups null. Overall speedup is native_fused
duration divided by combined duration. Full native correctness and execution
are prerequisites: an unsupported full invocation or allocation failure does
not fall back to the slower chunked native timing. Native batch count equals
the Q-row count for this explicit-visibility adapter; this benchmark does not
claim an optimal model integration or alternative native packing strategy.
Full native workspace can require additional memory beyond frozen staging.

This is a standalone candidate benchmark; the production INT8 entry remains
unintegrated. Debug-enabled correctness calls keep their original default;
passing raw_dot=None disables that store at compile time. Preflight checks
the resulting TTIR for INT8-to-INT32 dot and exactly one tt.store.

Initial full-row and sampled checks precede timing. The last timed output
is checked directly without overwriting it with a fresh invocation. The
score job retains its last chunk; additional sampled rows are recomputed
outside timing for CPU comparison. Rounds alternate order;
the first order is reversed on alternate cases to reduce fixed-order bias.
No device isolation is asserted. Data from a shared/busy device is exploratory.

Run `bash tools/run_qsa_int8_kernel_benchmark_910b.sh --mode kernel --rounds 1
--kernel-warmup 1 --kernel-active 3`. Logs stream to terminal. Per-case JSON,
CSV, compiler/timer source and hardware snapshots are uploaded even on failure.
Only kernel mode is supported; copies, quantization, row-table generation and
CPU validation remain outside measured jobs. Native output allocation remains
part of its public API, while host launch gaps are excluded by the timer.
