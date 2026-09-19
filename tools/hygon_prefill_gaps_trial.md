# BW1000 remaining prefill probes

## Project truth (G1)

| Field | Frozen choice |
| --- | --- |
| Current op | Hygon top_k_per_row_prefill, current one-scan + dense carry |
| Editable | tools/hygon_prefill_gaps* and the existing report runner |
| Read-only | src/, benchmark/, tests/, prior reports and baseline |
| Build | Local stdlib source checks and lint; GPU JIT on BW1000 |
| Validation | Exact selected value multiset, distinct relative indices, short-row -1 padding, output guards; every candidate before timing |
| Benchmark | Device kernel microseconds; paired ABBA/reverse over seeds 42/43; ratios >1 favor candidate. Scratch also reports synchronized host end-to-end time separately |
| Active set | Seven existing shapes; geometry additionally sweeps rows 4/64/160/320/640/1280/2560 at vocab 4096/16385/129280 |
| Aggregation | Per-shape paired ratios, raw samples, source SHA256; no aggregate promotion claim |
| Timeout | Separate worker per shape/config, 1200 seconds default; bounded report wrapper |
| Profiler | Torch profiler CUDA events; exactly one non-TLE kernel per timed call; TLE separately records all kernel names |
| Autotune | Offline Hygon-only probe; no NVIDIA registration/config changes needed |
| Fallback | No torch compute fallback; torch used only for setup, references, diagnostics and timing |

## Contract (G2)

| Field | Contract |
| --- | --- |
| Backend | BW1000, warp64; runtime verification required |
| Input | fp32 rank-2 logits, int32 bounds, stride1=1, contiguous or padded rows; valid ranges |
| Output | Caller-owned int32 [rows,k], relative indices, unique, -1 for short rows |
| Semantics | Exact value multiset; ties may select different equal-valued indices. Includes infinity, signed zeros, constants, partial/short/padded rows |
| Dtypes/promotion | fp32 values, int32 counters and indices; no promotion |
| Unsupported | NaNs, other dtypes, column-strided input, autograd, concurrent sharing of one scratch workspace |
| Reference | Existing audited torch.topk oracle and check_output; installed torch/vLLM versions recorded |

## Paths (G3)

| Stage | Mechanism | Question / risk |
| --- | --- | --- |
| preflight | Import vLLM custom ops before resolving schema/dispatch; launch and check it | Baseline must resolve to compiled CUDA registration; no fallback |
| radix | Port final radix selection to global 256-counter scratch and tl.cumsum; gates 0/64/256 actual candidates | Extra clear/scan versus actual rank workload; full keys and index scratch remain intact |
| counters | Sparse found-counter scan alone and both counters; dense carry + final-counter scan | Density crossover, especially final-bin sparsity; all seven collection call sites retained |
| scratch | Identical kernel/launcher/output, fresh empty scratch versus caller-owned reusable scratch | Allocation affects host latency; does not eliminate kernel scratch traffic. Reuse is sequential only |
| compression | Untimed per-row candidate-count and ordered-key high16 agreement | Measures whether lossless uint16 residuals with a row prefix are even possible; NOT a compressed operator or speedup |
| geometry | B256/w2,w4; B512/w4,w8; B1024/w8,w16 over row sweep | Measure crossover with current production route, independent of obsolete wide-block env gate |
| tle | Explicit generic prefill with force-TLE in isolated subprocess | Prefill-specific correctness/performance, separate from old decode failure; all normal shapes plus adversarial inputs |

P0 cost-fit and refinement-count probes already exist. Compression stage records
last refinement step and final count again on the current source. P3 found-counter
prefix sum and dense carry already ship; only the remaining combinations are new.
TLE lowering already succeeded. LDS histogram atomics were slower in the old
microbenchmark, so no blanket LDS speedup is assumed.

All stages can run in one report:

```bash
tools/hygon_prefill_next_run.sh gaps hygon_prefill_gaps_v1
```

Run stages separately (stdout is JSONL plus diagnostics):

```bash
PYTHONPATH=src:$PYTHONPATH python tools/hygon_prefill_gaps.py radix
PYTHONPATH=src:$PYTHONPATH python tools/hygon_prefill_gaps.py counters
PYTHONPATH=src:$PYTHONPATH python tools/hygon_prefill_gaps.py scratch
PYTHONPATH=src:$PYTHONPATH python tools/hygon_prefill_gaps.py compression
PYTHONPATH=src:$PYTHONPATH python tools/hygon_prefill_gaps.py geometry
PYTHONPATH=src:$PYTHONPATH python tools/hygon_prefill_gaps.py tle
```

`--shape-ids`, `--rounds`, `--iters`, `--seeds` and `--timeout` limit a run.
The default matrix contains 182 workers (including preflight). Geometry IDs
refer to its own row/vocab grid, printed in every worker; the other stages use
the seven standard benchmark IDs. Radix validation explicitly covers final
counts 1/63/64/65/255/256/257 and min(2048,vocab), including remaining-k=1.
TLE adversarial checks include row 12289 for shapes taking the radix tail.
`all` runs preflight first; a missing/incorrect compiled baseline aborts the run.
Individual failed workers are reported and other independent configurations
continue. Nonzero final exit means at least one worker failed, not that its
report was lost. The wrapper commits/pushes only the new report, including
failed reports. Run on an idle HCU with HIP_VISIBLE_DEVICES set as needed.

Local checks do not establish GPU correctness or speedup. Production promotion
requires successful remote validation and a separate ordinary benchmark.
