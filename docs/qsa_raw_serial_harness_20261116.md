# Serial raw score + unchanged TopK harness candidate

## Project truth

| Field | Contract |
| --- | --- |
| Goal | Validate final selections and stock harness kernel CANN V1 speedups on ten complete inputs |
| Frozen | Production, workload tensors, raw query16/empty finish math, current INT8 score, TopK source/call size and compiler correction |
| Editable | Optional callback in existing all-chunk driver, serial adapter, new benchmark/worker/master runner/tests |
| Schedule | Original global chunks; raw request pieces finish into the same global scratch before original 32-row TopK calls |
| Timing | Unmodified GenericBenchmark --mode kernel against full native CANN V1 call; separate score/TopK/current combined timings use same get_latency |
| Preparation | Existing prepared-input scope; no reuse of packed K/KS across chunks; all four raw stages charged |
| Validation | All-row score checks before/after; full native/current/raw structure and tie-aware score cutoff checks before/after; sampled independent CPU oracle retained |
| Precision | Raw FP16 intermediate unchanged; final selected values must match raw own TopK exactly and current FP32 score cutoff within1e-3; index differences recorded |
| Memory | Existing chunk scratch; one temporary full raw-score matrix for isolated TopK timing only, allocated after fused timing and freed after the isolated stage |
| Evidence | All source/compiled IR and native build/launch gates retained; same verified compiler copy for both candidate score/TopK and current control |
| Autotune | Fixed unregistered integration diagnostic; no production configs or registrations |
| Acceptance | Report stock harness CANN V1/raw whole-call arithmetic mean only when ten cases and postchecks pass; candidate, not production |

## Contract

| Field | Contract |
| --- | --- |
| Selection | Compressed-group indices[rows,512], int32; unique/in-range/-1 count checks on every row |
| TopK | Same repository persistent_topk, automatic route, 32-row batches; no new TopK optimization |
| Raw execution | Sequential QW pack, fused K/KS pack, two-Cube wrapper, empty finish, then TopK; no overlap claim |
| Bounds | Global chunk/output offsets retained; all request splits charged; zero-visible pieces use existing score kernel |
| Failure | Any malformed native output, rank mismatch, bad compiler source or incomplete harness record fails; no partial-case mean |
| Evidence limit | Native fused score/TopK cannot be split; only candidate/current stages are timed independently, non-additive |

## Paths

| Path | Implementation |
| --- | --- |
| Current control | Existing PreparedSharedMapInt8Selection, same inputs and automatic TopK |
| Serial raw candidate | Existing verified all-chunk closures with empty1024; existing select_chunk after global chunk completes |
| Native | torch_npu.npu_quant_lightning_indexer V1, same TND/PA_BSND/query ends/visible/table arguments |
| Harness | New benchmark file reuses QSA GenericBenchmark class; no monkeypatch to harness timer or Config.mode |
| Runner | Existing isolated corefix setup; new master spawns each case in separate pytest process, timeout/logs and automatic report upload |

The score-only ten-input result 3a80cbe827cc421184bd6d48e4e035f4cf775974
passed with mean current/raw1.393806x. That does not establish final selection
correctness or CANN V1 whole-operator speedup. This experiment supplies those
checks without changing production or introducing score/TopK double buffering.

Local checks passed: 77 CPU tests (34 source/bounds probes, 24 raw-address,
5 compiler-mode, 5 workload planner, 4 capture guard, 5 serial/summary/worker),
Python/Bash syntax and focused pre-commit. Device compilation, final selection
checks and stock harness execution remain pending. Stage diagnostics retain
stock harness sampling defaults; native score/TopK splitting is not attempted.
