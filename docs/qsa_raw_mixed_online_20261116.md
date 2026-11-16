# Raw Cube and online TopK mixed-kernel feasibility pair

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Establish raw native-call composability and one/two-slot correctness before full-input double buffering |
| Frozen | Native query16 C++, passing QW/K/KS packing, isolated compiler correction, production and formal harness |
| Editable | New generated mixed wrapper, micro probe/tests/runner flag |
| Inputs | One 16-query group, H4/D128/page192, widths512/1024/4096 (1/2/8 updates), shuffled pages, varied visibility incl zero and partial updates |
| Compute | Same native two-Cube call; Vector multiplies logical KS then maintains online Top512 |
| TopK difference | Online sort/merge from existing tested UB-snapshot helper; not the standalone persistent_topk host pipeline |
| Pair | Identical generated kernel/algorithm; only SLOTS=1 versus2 changes storage and handshakes |
| Timing | Prepared QW/K/KS, Cube+Vector kernel device sums, two reversed rounds, three active, existing guard5ms and strict task checks |
| Evidence | Warmup metadata gate before execution, source/IR/binaries, numerical traces and optional long-case pipe/memory counters |
| Acceptance | Micro feasibility only; no full-input/CANN speedup or production change; speedup alone does not prove overlap |
| Autotune | Fixed diagnostic exemption |

## Contract

| Field | Contract |
| --- | --- |
| Native memory | Private TPipe only on Cube; no enclosing Triton Cube local tensor allocations; Vector UB state belongs to Vector scope |
| Ring | GM[SLOTS,8,16,64] FP32, raw writes one 64-key tile; producer owns slot until ready, consumer releases after all16 queries updated |
| Protocol | Same existing block-event ready FIX->MTE2 and free MTE3->FIX edges; initial free credits and final drain |
| Vector state | Each subvector handles8 rows; UB packed state[8,1024]; preserve known snapshot-before-gather merge helper |
| Validation | Exact received scores versus independently materialized raw score reference; unique/in-range/padding and exact selected score values; pre/post repeated |
| No fallback | Bad metadata or mismatch fails; timeout retains launch-phase record; no compiler source patches beyond existing independent correction |
| Padding | Width multiple512; invalid logical positions masked by per-row visibility; no arbitrary-row/table claim |

## Paths

| Path | Work |
| --- | --- |
| Preparation | Same raw query16 QW and fused K/KS kernels, once outside micro timing |
| Reference | Existing native Cube wrapper + logical finish materializes all logits for check only |
| Mixed serial | Same mixed wrapper, one GM ring slot; Cube waits for Vector release before next update |
| Mixed buffered | Two slots; producer may issue next update before prior Vector update finishes |
| Trace | Separate TRACE=True correctness launch stores all received/masked scores; timing TRACE=False omits it |

The completed serial harness candidate has ten-case CANN V1 mean0.505224x.
Its standalone persistent TopK cannot be embedded as a host call inside a mixed
kernel; this pair uses the same online update algorithm in both arms to isolate
the scheduling question. It must first pass local-buffer/event ownership and
raw custom-op linking checks. It is not claimed performance-equivalent to that
serial pipeline. Hardware counters are retained as diagnostics; pipeline_proven
and overlap_proven remain false pending actual instruction/interval evidence.

Local checks passed: 37 CPU tests (4 new ownership/address/helper tests,
24 raw-address, 5 compiler-mode and 4 capture guard tests), generated Python
syntax, CLI help, Bash syntax and focused pre-commit. Device compilation,
composability and execution remain pending. Counter collection failures are
recorded explicitly and do not turn numerical evidence into overlap proof.
