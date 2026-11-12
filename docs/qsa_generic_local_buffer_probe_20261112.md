# Generic TLE L1 operand feasibility probe

The installed al.fixpipe/al.copy wrappers explicitly require Ascend910_95.
No wrapper or compiler modifications are allowed here. Generic DSA to_buffer and
to_tensor are the remaining operator-level hypothesis, not established 910B support.

| Contract | Frozen scope |
| --- | --- |
| Workload | budget_exact longest-request tail32, original width8192, H4 D128 INT8 Q/K, page192 |
| Additional check | 16 queries, width193, visible0/1 and tail masking |
| Kernel | Existing query16 second-Cube body; replace only operand buffering |
| Change | FP16 weights/input -> tle.dsa.to_buffer(space=L1) -> to_tensor -> existing tl.dot |
| Current control | Unregistered integrated INT8 score_into, not public BF16 entry |
| Numerical contract | Existing FP32 CPU score reference and Top512 selected-score checks, rtol/atol1e-3 |
| Timing | Existing harness kernel timer, two reverse-arm rounds; score, unchanged persistent_topk, combined |
| Acceptance | Diagnostic chunk only, no full-input arithmetic mean |

Before device execution, candidate.warmup compiles the exact specialization with
one print-after-sync-solver option. Preserve full compiler call, input, available
output hashes and post-solver dump. Gate requires two Cube matmuls including a
FP16-input Cube, no FP16 GM workspace declaration and no FP16 GM->Cube ND2NZ.
Missing/truncated dumps reject the candidate. This is a conservative structural
gate, not proof of exact CANN L0C->Fixpipe->L1 equivalence. INT32 QK and FP32
output GM handoffs are allowed to remain. All current page/QK/grid/launch choices
remain unchanged; no NVIDIA autotune applies to this diagnostic tool.

Compile or IR rejection ends this experiment before execution/timing. Numerical
or selection rejection prevents timing. Device faults fail the worker and preserve
the report. Exit0 means collection completed; candidate_passed must be checked.
No persistent TopK algorithm, production code, compiler source, or installed
packages are modified. This is one generic-buffer hypothesis, not a tiling sweep.
