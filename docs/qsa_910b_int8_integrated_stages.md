<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Prepared INT8 selection integration and stage comparison

## Project truth

| Item | Frozen choice |
| --- | --- |
| Target | Ascend 910B, pinned CANN 9.1 and FlagTree ascend3.5 environment |
| Scope | Prepared INT8 adapter under tools; no production registration |
| Input set | Ten existing migrated INT8 workloads, unchanged tensor generator |
| Evidence | Ten-case screen plus independent first_chunk/compact/long100k report 4e6912639a903a72b16c93f1974a432b56a56ad2 |
| Scorer | Same _quant_score_trial JIT function; grid cap20, N128, num_warps4 |
| TopK | Existing persistent_topk source; call cap32, block256, k512 |
| Native | npu_quant_lightning_indexer V1, full logical workload in one invocation |
| Timer | Explicit CPU+NPU trace, all four stage baselines rebuilt together |

## Contract

| Operand | Type / layout |
| --- | --- |
| Q | Contiguous INT8 [rows,4,128] on 910B |
| K | Contiguous INT8 [pages,192,1,128] |
| QS / KS / weights | FP16 [rows,4] / [pages,192,1] / [rows,4] |
| Table / visible | INT32 per-query page table / compressed-group visibility |
| Scores | INT32 dot then FP32 dequantization, ReLU, weights and head sum; masked scores -inf |
| Output | Caller-owned INT32 [rows,512] compressed-group indices, -1 padding |
| Scratch | Caller-owned bounded FP32 score chunk and at least 1MiB uint8 TopK workspace |
| Unsupported | Other device, H/D/page, dtype/layout/device mismatch raises; no silent fallback |

The prepared adapter accepts valid physical page indices and visibility within
width as caller preconditions, matching the frozen fixture. It does not inspect
NPU tensor contents on the host or create quantized inputs. It intentionally
restricts H4/D128/page192 to the tested envelope. Existing BF16 production paths
and broader original trial contracts are not replaced.

## Paths and verification

| Path | Implementation / verification |
| --- | --- |
| score_into / score_all | Shared launch path for preflight, isolated score and integrated selection |
| topk_all | Existing TopK on benchmark-only frozen full logits, same chunk/call boundaries |
| __call__ | Score one bounded chunk, select in <=32-row calls, reuse buffers |
| Native comparison | Full native V1 primary; native stage split unavailable |
| No torch compute fallback | Adapter uses metadata checks, view slicing and existing Triton launches only |
| Autotune | Not added: fixed previously screened configuration, integration validation only |
| Registration / exports | Not applicable to standalone prepared trial; production INT8 integration deferred |

`benchmark_qsa_int8_kernel_910b.py --integrated` uses the prepared adapter for
preflight, score-only, TopK-only and combined stages. Combined timing calls the
adapter entry directly; it does not reconstruct a different benchmark pipeline.
The original baseline mode remains available without the flag. An auxiliary
chunked-native comparison is omitted in integrated mode to reduce work.

Validation retains full output structure/visibility/uniqueness checks,
independent sampled CPU scores, full-row native/candidate selected-score
comparisons, no-debug score IR checks, and postchecks after timing. Output order
and exact tied index identity are not required. Tolerance stays rtol/atol 1e-3.

All four timed stages use synchronized CPU record-function invocation windows
in a single CPU+NPU trace. Sum device-task durations per window; host time and
launch gaps are excluded. Require every device task to belong to exactly one
window, nonempty windows, identical task-name counts across active invocations,
and at least the expected score/TopK call counts. Score-only task count is exact.
This detects empty/incomplete collection, but cannot prove device isolation or
rule out identically missing subsets of opaque fused-native tasks. Raw traces
and rejected attempts are retained. Only incomplete captures retry, at most
three total attempts. Score one/three-call calibration remains required.

Trace instrumentation can affect kernel behavior and timing. This is an
exploratory integrated comparison, not formal production performance acceptance.
TopK is measured, not optimized. Native V1 offers no isolated stage interface, so
score/TopK native speedups remain null; only native full time / combined time
is a native speedup. Separate score and TopK durations are not assumed additive.
Do not combine these timings with older library-timer raw data.

The recommended full-ten run uses one round and two active invocations after
one warmup per stage. This reduces stage captures after scoring stability was
already explored; it is not an independent multi-round stability claim. Terminal
and CSV list score_us, topk_us, combined_us, native_fused_us and overall_speedup.
An equal-weight full-ten overall mean is emitted only when all ten pass.

Local checks: black/isort/flake8, compile/shell syntax, metadata-only adapter
routing with full and tail chunks, missing-dependency report path, and synthetic
trace window completeness/partial-capture rejection. Actual compiler/device
execution, CANN alignment and timing require the card run.
