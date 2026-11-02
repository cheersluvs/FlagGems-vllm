# QuantLightningIndexer V1 baseline prerequisite probe

This probe does not change the production operator, install packages on the
card, or benchmark BF16 against INT8. It checks whether the current 910B
environment can execute the native INT8 V1 API with our H=4 workload.

Upstream sources inspected on 2026-11-02:

- [V1 interface](https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md)
- [V1 registration](https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp)
- [V1 tiling validation](https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.cpp)
- [TorchNPU schema](https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml)

The schema takes query/key quant modes as required positional arguments. Both
are zero for per-token-head quantization. No BF16 or V2 fallback is used.
Presence in upstream master does not imply presence in installed CANN 9.1.

## Cases

All cases use INT8 Q/K, H=4, D=128, page192, FP16 scales and weights, and
group TopK=512. The page mapping is shuffled. Weights are one; scales vary
by head or key position and are powers of two to limit rounding ambiguity.

| Case | Original Q lengths | Original prefix lengths | Visible compressed groups |
| --- | --- | --- | --- |
| h4_topk512 | 2 | 16384 | 4096 for both rows |
| h4_compressed_boundary | 9, 5 | 2044, 4093 | 511..513 and 1023..1024 |
| h4_select_all | 3 | 28 | 7 for every row |

All cases use
`floor((prefix + local_query + 1) / 4)` to construct visibility. The boundary
case crosses both compression boundaries and the TopK select-all threshold.

V1 has no compression-ratio attribute. Each Q row is presented as one native
batch with its own page-table row and exact visible group count, using
`TND`, `PA_BSND`, and `sparse_mode=0`. This checks the adapter's compressed
causal semantics, not a nonexistent native V1 compressed-causal mode.
An intentionally high-scoring forbidden key detects missing visibility masks.

CPU reference: INT32 Q/K dot products, FP32 dequantization, ReLU, weighted
head reduction, and TopK. Checks cover output shape/dtype, valid index count,
uniqueness, index range and selected-score multisets (absolute tolerance 1e-3).
Ties need not return identical indices or ordering. Padding is required to be
-1 for this baseline contract; raw outputs are saved before checking, without
masking, repairs, or replacement TopK. A padding-contract failure needs review
before treating it as a native numerical failure.

## Run and reporting

Execute `bash tools/run_qsa_quant_v1_910b.sh --device 0 --timeout 180` with
`QSA_ENV_ROOT=/opt/qsa-topk-env.ojEsoQz5` and
`QSA_OPP_PATH=/opt/qsa-opp-910b-qw4qRghU`. The wrapper reuses the proven
environment activation and report-upload flow. It prints per-case geometry,
selected-score error, environment/schema and exceptions, then uploads raw
indices, logs and summary even when a case fails. Each case runs in a fresh
subprocess with a timeout. Exit 0 means all three checks passed; 2 means all
cases lack prerequisites; 1 means a failure, crash, timeout or mixed outcome.

After success, add and validate the matching production INT8 path before
measuring speedup. Reuse NV's ten logical workload geometries, replacing FP8
with INT8 and page196 with page192, rebuilding physical pages and page tables.
Do not duplicate H=4 into H=64 to make V2 execute: that changes the workload.
