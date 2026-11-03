# Ten NV logical workloads migrated to INT8/page192

## Project truth before implementation

| Field | Frozen choice |
| --- | --- |
| Current task | Full logical geometry correctness for INT8 scoring + existing TopK vs native V1 |
| Editable | New workload/probe/runner and this document |
| Read-only | NV workload files, validated INT8 score, TopK/loader, earlier probe helpers, production files |
| Build | Existing FlagTree JIT; local formatting, syntax and metadata checks |
| Validation | Full-row index structure, selected-score pairing and score masks; independent CPU reference on <=24 rows/case |
| Benchmark/aggregation | None; shape execution only, no kernel speedup or latency claim |
| Active set | All ten NV FP8 logical geometries; INT8 Q/K, H4 D128 ratio4 K512, page192 |
| Timeout | 900 seconds per case; fresh subprocess; optional --cases for partial reruns |
| Profiler | Deferred until full-shape correctness |
| Autotune | Standalone correctness exemption; unchanged N32 score tile and TopK block256 |
| Fallback | No BF16 conversion or Torch compute fallback for candidate; existing device TopK routes unchanged |

## Alignment contract before implementation

| Field | Contract |
| --- | --- |
| References | Frozen tools/qsa_prefill_fp8_workloads.py; native QuantLightningIndexer V1; CPU integer dot/FP32 score |
| Device | Pinned 910B CANN9.1/FlagTree environment already validated on 19 small-case rows |
| Input | Original request counts, per-request query lengths, prefixes, max context, H/D, compression and group TopK retained |
| Storage migration | FP8 -> INT8; page196 -> page192; rebuild minimal shuffled cache/pages and table; original unused capacity not retained |
| Quantization | Synthetic normal values quantized per token/head to [-127,127]; actual FP16 stored descales used by both implementations; weights one |
| Output | INT32 compressed-group indices, K512; valid count min(visible,512), unique/range-valid, padding -1 |
| Arithmetic | Candidate INT32 dot and FP32 score; independent reference uses exact stored INT8/FP16 operands |
| Visibility | floor((prefix+local_query+1)/4), including zero-visible first rows |
| Comparison | All rows: native/candidate selected scores evaluated with candidate logits, rtol/atol1e-3; CPU-sampled rows independently check full logits/dots and both TopKs |
| Boundaries | Zero visibility, select-all, mixed requests and final chunks included; no gradients or unsupported dtypes/layouts |
| Provenance | Logical metadata is inherited from prior fixtures, including synthetic lengths; these are not captured Ascend INT8 runtime tensors |
| Torch use | Generation, transfer, validation and CPU reference only in standalone probe; no candidate compute fallback |

## Paths before implementation

| Path | Trigger | Implementation | Configuration | Validation | Benchmark | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| Migration | Ten frozen workloads | Repack page/table metadata only, no original fixture edits | H4 D128, ratio4 K512 | --plan-only on Mac | None | Physical storage differs from NV |
| Candidate | Every chunk | Validated masked debug score + existing persistent TopK | Chunk <=128, score+dot scratch <=64MiB; block256 | All rows; independent sampled CPU reference | None | Debug stores prevent timing acceptance |
| Native | Every chunk | Same INT8/scales/weights/visibility through native V1 | Per-row native batch, local cumulative query lengths | Same row checks and CPU sample | None | Larger lengths and zero visibility newly exercised |
| Unsupported | Outside frozen contract | Explicit failure | None | Prior host gates | None | No silent fallback |

Logical Q rows are preserved, while both paths execute bounded chunks for
correctness. This does not freeze future benchmark batching. The long100k
case uses a smaller chunk to keep score plus raw-dot scratch within 64MiB.
Input cache, per-row table, native workspace and validation temporaries are
additional allocations; the scratch limit is not a total-memory guarantee.

The report includes an explicit sample-row list, all-row counts, paired
selected-score error, sampled independent CPU errors, index-set equality
counts, hashes of complete output streams and raw indices for sampled rows.
It does not claim full independent CPU validation of all rows. Mask/finite
checks cover every score, while exact integer-dot checking is sampled.

Run `bash tools/run_qsa_int8_ten_shapes_910b.sh --device 0 --timeout 900`.
The runner streams progress and uploads logs/JSON after all selected cases,
including failures. `--cases int8_tail112 int8_tail113` selects partial reruns.
`--plan-only` generates the ten-shape table without Torch/NPU dependencies.
No production or benchmark shape file is changed in this round.
