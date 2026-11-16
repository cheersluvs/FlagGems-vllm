# Full ten-input mixed two-slot harness candidate

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Compare CANN V1, preceding raw serial entry and two-slot mixed entry on all ten complete inputs |
| Frozen | Native two-Cube and .cube bridge, FP16 intermediate, online sort/merge helper, standalone persistent TopK, compiler, production and fixtures |
| Editable | Prepared mixed adapter/source wrapper, optional existing harness mode, summary/tests |
| Schedule | Existing global chunks and per-request pieces; mixed call owns each piece's rows/map; all pieces and all three GPU launches charged |
| Preparation | QW and K/KS repacked inside every timed piece, no reuse across chunks |
| Invisible blocks | Same causal contract; execute at least one and otherwise ceil(max visible/512) updates; no selections can come from omitted -inf suffix |
| Timing | Stock GenericBenchmark --mode kernel native/mixed whole calls; same get_latency for serial/current comparison and diagnostics |
| Validation | Existing score pre/post; complete CANN/current/serial/mixed rank checks; exact mixed trace and raw selected values; guard regions |
| Autotune | Unregistered diagnostic, no production tuning/registration |
| Acceptance | Ten-case arithmetic mean of stock CANN V1/mixed whole-call speedups only if all cases/postchecks pass |

## Contract

| Field | Contract |
| --- | --- |
| Mapping | Piece never spans request map; grid=ceil(piece rows/16), each owns its ring |
| Workspace | Reuse maximum per-chunk QW/K/KS/ring/trace/value scratch; output has whole-input guard rows |
| Key padding | Runtime max-visible and tiles; masked safe page0 access past max-visible, zero K/KS padding to512 |
| Row/key tails | Same proven boundary masks and update protocol; no event skipped for inactive rows |
| TopK identity | Mixed online512 sort/UB-snapshot merge; serial control persistent_topk auto is unchanged; identities recorded separately |
| Independent TopK data | Serial persistent TopK from staged raw logits plus standalone online512 math from same staged logits; neither is the fused Vector stage duration |
| No fallback | No torch computation in prepared execution; validation/staging only may use torch compute; maximum0 still executes one padded block |
| Precision | Exact mixed-versus-raw scores and selected scores, current FP321e-3; tied-index identity not required |

## Paths

| Mode | Behavior |
| --- | --- |
| Existing serial harness | Preserve prior CANN/raw serial comparison |
| --serial-harness --mixed-harness | CANN/mixed stock benchmark; serial/current whole-call and score/TopK diagnostics same worker |
| --summarize-only | Retain recovery support, require selected mode's explicit op/route/test identity |

The six boundary cases in43aefa386f306798243cae85438b34d0f06b906a passed,
with one/two-slot pair speedups1.216–1.557x. Those prepacked micro results do not
establish full-input benefit. This run charges packing, splits and online TopK,
and records real final selections. Production remains unchanged.

## Local validation

- 48 raw-kernel, ABI, boundary, ordering and aggregation checks passed.
- Five complete-input schedule/score-suite checks passed.
- Python compilation, Bash syntax, focused pre-commit and diff checks passed.
- No NPU execution performed on the Mac. Full ten-input correctness and timing
  remain the next card run, not an established candidate speedup.

Run through `tools/run_qsa_raw_cube_chain_910b.sh --compiler-corefix
--serial-harness --mixed-harness --device 0 --case-timeout 1800` with the pinned
910B environment and OPP. The first argument selects the previously verified
isolated compiler copy. The runner uploads reports even when a worker fails.
The console prints per-case harness results, auxiliary stage timings and the
complete ten-case arithmetic mean only when all checks pass.
