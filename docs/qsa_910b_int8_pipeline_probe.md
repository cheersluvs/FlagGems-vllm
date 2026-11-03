# INT8 score and existing persistent TopK integration trial

## Project truth before implementation

| Field | Frozen choice |
| --- | --- |
| Current op | Standalone INT8 QSA selection vs torch_npu.npu_quant_lightning_indexer V1 |
| Editable | New pipeline probe, runner and this document |
| Read-only | Production ops, validated INT8 scorer, native V1/score fixtures, existing TopK and its loader |
| Build | Existing FlagTree JIT; py_compile, black/isort/flake8 and bash -n locally |
| Validation | Three cases; full scores/dots vs CPU, both index sets vs CPU selected scores, paired selected scores |
| Benchmark/aggregation | None; no latency or speedup in this correctness trial |
| Active set | INT8 H4 D128 page192 K512, ratio4; 2/14/3 rows, including varying scales/weights and select-all |
| Timeout | 300 seconds per case, isolated subprocesses |
| Profiler | None until pipeline correctness; prior score IR already confirms i8 dot -> i32 |
| Autotune | Exempt standalone Ascend integration probe; reuse fixed validated score tile and production TopK block256 |
| Fallback | No Torch scoring/TopK fallback for candidate or native baseline; existing TopK device routes stay unchanged |

## Alignment contract before implementation

| Field | Contract |
| --- | --- |
| Reference API | [QuantLightningIndexer V1](https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer/README.md) |
| Device/dtype | 910B; INT8 Q/K, FP16 scales/weights, INT32 table/visibility |
| Input/output | Same contiguous tensors for native and candidate; group indices [rows,512] INT32; native [rows,1,512] |
| Arithmetic | Candidate INT32 dot then FP32 dequantization/ReLU/head reduction; exact dots and rtol/atol1e-3 scores |
| TopK semantics | Valid count min(visible,512), unique indices in [0,visible), -1 padding; order/tied indices need not match |
| Comparison | Both outputs checked independently vs CPU FP32 TopK scores, then paired sorted selected-score multisets, rtol/atol1e-3 |
| Compression | Exact per-row visible groups via one native batch per query, sparse_mode0; output remains group indices |
| Unsupported | Other shapes/dtypes/layouts not accepted by frozen scorer; no gradients; only finite scales/weights tested |
| Torch use | Input generation/transfers, reference checks and allocation in standalone probe only; candidate computation in existing Triton kernels |

## Paths before implementation

| Path | Trigger | Implementation | Configuration | Test | Benchmark | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| Score | All cases | Unchanged debug INT8 scorer with explicit output mask | N32, head16, <=8 programs | All valid dots/scores and padding | None | Debug store affects timing; do not time |
| TopK | All cases, mixed select-all/wide rows | Unchanged persistent_topk, existing loader only resolves one package import | block256, library auto route unless environment overrides | Raw output range/count/uniqueness/selected scores | None | TLE route may use library's existing radix fallback; state recorded |
| Native | All cases | Direct TorchNPU V1 call, no repair or replacement TopK | quant modes0/0, TND/PA_BSND, sparse_count512 | Independently checked vs CPU | None | Native arithmetic rounding and tied indices |
| Unsupported | Outside contract | Explicit error | None | Existing scorer metadata checks | None | No silent BF16 conversion |

## Execution and evidence

Run `bash tools/run_qsa_int8_pipeline_910b.sh --device 0 --timeout 300` with
the previously pinned QSA_ENV_ROOT and QSA_OPP_PATH. JSON contains all-row
score checks, independent native/candidate TopK checks, paired selected-score
differences, raw index arrays, source hashes, compiler path and TopK loader
identity/state. The summary omits raw indices; case JSON retains them.

TopK is not optimized in this round. No production file is modified. Passing
these three cases permits the next step of migrating NV's ten logical shapes
to INT8/page192; it is not full-shape or performance acceptance.
