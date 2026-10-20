# QSA prefill key-major dot trial

Pre-coding plan, 2026-10-20. This is a private experiment based on the
retained narrow compact scheduler, not a production replacement.

| Project truth | Frozen choice |
| --- | --- |
| Baseline | Production SHA256 d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73; compact gate from 0115b444; strict report c090220d |
| Reference | Unchanged vLLM 7a877ae34dcf4a7d46de08eb16db4699e1e0047c, SHA pinned by runner |
| Evidence | Upstream12 score/native geometric mean 0.74144; 11 cases choose TR32/BN64/KT4 (73728 shared bytes), one TR64/BN64/KT16 (106496). Native uses keys x query; production uses query x keys |
| Hypothesis | Transposing the dot operands and corresponding reduction/output axes may reduce compiler layout-conversion/resource costs on long prefill. Source alone does not establish the bottleneck or predict a gain |
| Editable | Private trial, correctness tests, candidate selector, untimed resource diagnostics |
| Read-only | Production source, native source, TopK, six tuning candidates, keys, tolerances, compact routing and benchmark shapes |
| Build / validation | Local AST/CPU tests, py_compile and pre-commit; H800 full production/compact/key-major correctness before timing |
| Benchmark | Verified same FlagTree/ptxas13.4/CUDA13.4 runtime; kernel mode, score excludes TopK, seven alternating paired rounds; upstream12 and observed10 separately |
| Profiler | Mac has no CUDA; previous H800 runs lacked ncu/nsys. Record availability, compiled registers/spills/shared bytes and static PTX instruction counts outside timing; these are not hardware counters |
| Autotune | Same qsa_select_paged_prefill_score six candidates and keys; new JIT/LibEntry identity. No new configuration search in this trial |
| Decision | Keep/revert after on-card correctness and paired performance. No measured improvement claimed before that run |

| Alignment contract | Preserved behavior |
| --- | --- |
| Inputs / outputs | Original public signature, BF16/FP8, parameterized H/D, strides, metadata integer types, output/padding and ties |
| Arithmetic | FP32 dot accumulation, ReLU each real head, sum real heads; zero padded heads before sum; unchanged tolerances |
| Bounds | Empty requests, nonzero query base, cut chunks, visible columns, invalid pages and padded dimensions |
| Workspace / stream | Same public bytecode, chunk geometry, logits buffer, scratch, allocations and stream |
| Torch | NO_TORCH_COMPUTE_FALLBACK; Torch reference computations occur only in tests/tools |

| Path | Implementation and coverage |
| --- | --- |
| Empty / select all | Original entry, no new kernel; score/TopK N/A for select all |
| Compact eligible | Existing narrow compact kernel unchanged; observed ragged54 is the sole active case |
| Other scored chunks | Original rectangular launch/grid, new keys x query score kernel; same six configurations |
| Baselines | Actual production entry warmed first in real chunk order, independent outer cache; native vLLM unchanged |
| Correctness | Every config, BF16/FP8, H4D128/H3D24/H5D96/H1D8, empty requests, strided int64 metadata, six chunk cuts, full native/FP32 comparisons |
| Resource evidence | Candidate/production LibEntry compiled objects and native JIT cache when supported; missing optional attributes are explicit, not zero |

The only arithmetic-layout change is `(query, keys)` to `(keys, query)`,
reshape `(rows, heads, columns)` to `(columns, rows, heads)`, and the matching
load/store axis transposes. Cache hints, loop staging and tile choices remain
unchanged. Native also uses an eviction hint and explicit loop staging; those
differences are deliberately outside this experiment. Numeric equivalence
must be established on H800 because operand orientation can affect lowering.

Run the existing upload wrapper with `--candidate key-major --suite
combined-prefill --mode kernel`. The complete gate adds 24 key-major cases to
the prior 194 passing tests; on one SM90 device the expected count is 218
passed and the existing second-device skip. This is an expectation, not a
GPU result. Both native and current-production score ratios remain in the
report. Only score/kernel mode is the NV acceptance metric; TopK and pipeline
remain auxiliary. Strict idle-card admission and automatic report upload on
test failure remain active.

`kernel_resources` in each case JSON records the compiled production,
rectangular candidate, compact candidate and native scoring variants after
warmup. Register/spill attributes may be absent in a compiler build and are
then null. PTX counts describe static instruction sites, not throughput,
executed traffic or a proven runtime bottleneck. No extra launch, new compiler
flag, SASS disassembly or external profiler runs inside the benchmark.

Local verification: 60 CPU tests ran, 59 passed and one optional Torch test
skipped because Torch is absent on the Mac; pre-commit passed. The tests check
that undoing the operand/axis transpose reproduces the entire production
kernel AST, execute the actual host bytecode with captured launch arguments,
verify independent caches and unchanged allocations/TopK, exercise the new
CLI/accuracy gate, and check resource parsing. These checks do not establish
GPU numerical correctness or a performance gain.
