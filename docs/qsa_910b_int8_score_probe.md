# INT8 score-only trial on Ascend 910B

## Project truth before implementation

| Field | Frozen choice |
| --- | --- |
| Current op | Standalone quantized QSA scoring, preceding integration with existing persistent TopK |
| Editable files | New INT8 score trial, probe, runner and this document under tools/docs |
| Read-only | Production BF16/NVIDIA ops, persistent TopK, native V1 fixture/validator |
| Build | FlagTree JIT on card; local py_compile, black/isort/flake8, bash -n |
| Validation | Three subprocess-isolated cases, exact INT32 dot check plus FP32 score rtol/atol 1e-3 |
| Benchmark/aggregation | None; no speedup or acceptance claim in this trial |
| Active set | INT8 H4 D128 page192; 2, 14 and 3 query rows; widths 4416, 1344 and 384 |
| Timeout | 300 seconds per case, adjustable by --timeout |
| Profiler | Not used before correctness; export compiler IR where available |
| Autotune | Exempt: independent Ascend compile/correctness probe, fixed N32/head16 tiles, <=8 programs; no NV path |
| Fallback | No BF16 conversion, torch scoring or native scoring fallback in candidate launcher |

## Alignment contract before implementation

| Field | Contract |
| --- | --- |
| Reference | CPU INT32 Q/K dot, FP32 dequantization, ReLU, weighted head reduction |
| Native semantics | [QuantLightningIndexer V1](https://gitcode.com/cann/ops-transformer/blob/master/attention/quant_lightning_indexer/README.md); native availability already verified separately |
| Device | Ascend910B; FlagTree from QSA_ENV_ROOT/flagtree |
| Input | Contiguous INT8 Q [rows,H,128], K [pages,page,1,128]; FP16 Q/K scales and weights; INT32 row page table and visible lengths |
| Scale semantics | Finite positive scales; finite weights; reference uses exactly stored FP16 values |
| Output | FP32 scores [rows,width], -inf for invisible columns; debug INT32 dots [rows,width,H], zero for invisible columns |
| Accumulation | INT8 dot accumulates INT32; dequantization, ReLU and head sum FP32 |
| Visibility | Explicit per-row group count, including floor((prefix+local+1)/4) compression boundaries |
| Autograd | Forward-only |
| Unsupported | Other dtypes, noncontiguous input, D!=128, H outside 1..64, non-NPU or non-910B device; fail explicitly |
| Torch use | CPU references, input generation and device copies only in probe; candidate host uses metadata and Triton launch only |

## Paths before implementation

| Path | Trigger | Implementation | Config | Validation | Benchmark | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| Empty | rows=0 | Return supplied output | None | Metadata | None | No launches |
| Trial | Supported contract | Paged INT8 tl.dot -> INT32 -> FP32 dequant/ReLU/reduce; debug raw dot store | N32, head16, <=8 programs | All valid scores/dots, all invalid masks | None | Installed compiler's integer dot support unverified |
| Unsupported | Outside contract | Explicit exception | None | Host checks | None | No silent conversion |
| External baseline | Native V1 | Not invoked by score-only trial, no native score output exposed | Separate validated probe | Previous report f5898880d1d0f12f9cdadef7141de05ec61e7724 | None | TopK indices do not prove full score equality |

## Workload and output

The three logical geometries reuse the unchanged V1 probe fixture. The first
uses full-range INT8 Q/K. The second uses non-power-of-two positive FP16 scales
and varying FP16 weights. The third includes a dot product at the signed INT8
extreme (-128 times -128 accumulated over D128). A high-scoring forbidden K
is retained in every case. Raw dots are checked exactly, scores use
rtol=atol=1e-3, and every invisible output is checked for negative infinity.

The score kernel has an extra debug store; it is unsuitable for timing.
Per-case JSON records numerical error, operand ranges, geometry, source
hashes, compiler path/version and available compiler IR text. No production
file or TopK implementation changes in this round.

Run `bash tools/run_qsa_int8_score_910b.sh --device 0 --timeout 300` using the
same QSA_ENV_ROOT and QSA_OPP_PATH as the passed V1 probe. The runner prints
results as each case finishes and uploads JSON, text logs and available IR.
Missing prerequisites return 2; failures/timeouts/crashes return 1; all cases
passed returns 0. Passing this trial is a prerequisite for TopK integration,
not production or performance acceptance.

## Padding diagnosis after report f6eb46ae

The first run compiled and executed all three cases. Each stopped at row zero
with nonzero invisible debug dots, after that row's valid integer dots, valid
scores and negative-infinity mask had passed. It did not check subsequent
rows. The report alone does not identify the cause of the nonzero debug data.

The follow-up runs two compile-time variants on each case: the original debug
store and an explicit `where(live, dot, 0)` debug store. Neither changes the
dot used for dequantization or the final score. The original variant retains
invisible debug values as diagnostics; the masked variant must return zero.
Both must pass exact valid-dot, FP32 score and score-mask checks for every row.
Validation collects all row failures instead of stopping at the first one.

Before each launch, output is filled with NaN and raw dots with 0x13579BDF in
the probe (not in the candidate). Reports include nonzero padding counts,
up to eight raw samples and sentinel counts to distinguish potentially
unwritten cells from nonzero dot results. IR export happens before numerical
validation and is attempted for both variants. Original padding anomalies
remain reported even when the explicit-mask variant passes. This experiment
does not establish a compiler root cause or production acceptance.
