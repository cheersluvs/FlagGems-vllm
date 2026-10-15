# QSA prefill: experimental single-query vector scoring

Trial 4 starts from retained commit
`454074da0c079cc649d139489ce11fd3141585b7`. Production remains unchanged until
the candidate passes on-card correctness and measured benefit warrants routing.
The current score kernel pads its matrix rows: H=4, TILE_R=4 uses a 16-row dot
even when a one-row chunk has only four real query-head rows. Prior one-row
fixtures spend 105–125 us of aggregate Graph time in scoring over 36 chunks;
Top-K remains a larger cost and is frozen by user instruction. The padded work
is a hypothesis for a generic vector path, not proof it will outperform tensor
cores: FP32 reduction could be slower, especially for larger H/D or FP8.

| Project field | Choice |
| --- | --- |
| Editable | Benchmark-only vector kernel and private candidate module; appended trial tuning section; trial tests and same-runtime full A/B support; docs |
| Read-only | Production operator in full, score/fill kernels, Top-K, validators, host preparation, existing six dot configs, reference tolerances and shared libentry |
| Hypothesis | For exactly one query, FP32 vector products/reductions avoid padded matrix rows and may reduce score latency |
| Build / tests | Python syntax/lint and CPU routing/source checks; full production suite plus trial Torch correctness and every vector config on H800 before timing |
| Active set | Existing 24 cases; add one-query H/D/dtype sweep and a 32/32/1 cross-request tail fixture for BF16/FP8 |
| Benchmark | Full candidate vs actual unchanged production, one process/compiler/input/budget per case; wall primary, Graph and isolated score diagnostic, paired ratios and raw rounds |
| Timeout / profiler | Existing 900 s per worker; CPU profiles outside timing; ncu availability unknown. Substitute evidence is padding ratio, dispatch/counts, separate score/Top-K timing and actual tuned binaries |
| Autotune | New trial-only qsa_select_paged_prefill_single_row_score, four BLOCK_N/warp configs with TILE_R=K_TILES=1 and num_stages=1. Same specialization dimensions as dot launcher; existing dot candidates/keys untouched |
| Fallback / integration | No Torch compute/copy/cast in candidate; noneligible chunks call the actual production score kernel. Prototype is not a new registered public operator |
| Acceptance | No production switch until useful full-entry benefit and no important regressions; rejected H/D regimes can remain on dot; no native-vLLM performance claim |
| Environment | Venv Python/Torch plus explicit external /opt/flagtree. Keep card 0 by default; the latest report shows other jobs on card 1. Long-case thermal drift stays diagnostic only. Record UUID/temperature/clocks, and rebuild both baselines if the user selects another free card |
| Docs | workflow.md/AGENTS.md apply; optimization.md/deep_opt.md absent. This is the first GPU scoring trial, not blind block-size tuning |

| Contract field | Preserved / verification |
| --- | --- |
| Scope | Exactly one query per scored chunk, positive H<=8 and D<=256; all other chunks retain production dot path |
| Numerics | Load BF16/FP8, cast to FP32, multiply/reduce D per head, ReLU per head, sum heads. Reduction order may differ; keep existing score/selection tolerances and unspecified tie order |
| Layout / metadata | Q/cache/table strides preserved, arbitrary packed request offsets and empty requests, physical page indirection and visible bounds; no host value reads |
| Bounds | Request containing query_offset alone writes; invalid physical pages map to -inf as before; mask visible columns and padded D lanes |
| Output / workspace | Same None API, int32 indices, logical logits shape/pointer, per-call preparation, bytes/budget, Top-K scratch/selection/padding/streams/lifetime |
| Validation / unsupported | Current validation and errors unchanged; existing SM90 BF16/FP8 inference-only gate and finite-input assumptions remain |
| Torch | Tests/reference only; candidate uses scalar metadata, existing empty allocations and Triton arithmetic |

| Path | Implementation / evidence |
| --- | --- |
| Empty/select-all | Actual unchanged production public bytecode; no scoring |
| One query, H<=8,D<=256 | Trial vector kernel, per-request early exit; keys loaded once per column tile, head loop bounded to eight |
| Multiple queries or larger H/D | Shared production dot kernel and original grid/configs; extra host gate cost is included in full A/B |
| Candidate host | Private module clones current public bytecode, replacing only score launcher with a pinned copy plus gate; never patches production module globals |
| Baseline | Actual production entry, exact current source hash pinned; both sides use identical validator, metadata preparation, Top-K, fill and compiler |
| Tests | Every vector config vs FP32 Torch scores, zero/short/full visible lengths, nonzero base/empty requests/padded layouts; full selection; CPU fallback/routing/identity/source checks |

Status: candidate implemented for on-card evaluation. GPU compilation,
correctness, performance and keep/revert are pending; no production routing
change or candidate performance claim.

## Implementation and local verification

`tools/qsa_prefill_single_row_trial.py` pins the complete production operator
file and six host-function source hashes to the baseline above. Its private
module uses the original public entry bytecode and shares the validators,
metadata preparation, fill and persistent Top-K callables. Only its private
score launcher chooses vector scoring for eligible chunks. No production
globals are patched; registration/export changes are unnecessary for this
benchmark-only candidate.

The vector kernel loads a column tile's keys once, converts inputs to FP32,
reduces products across D separately for each head, applies ReLU, and sums
heads. It contains no inline PTX, vendor intrinsic or Torch computation.
Triton `.to` calls are device arithmetic inside the kernel. All existing
tuning sections and production operator/Top-K sources are unchanged; only
the new trial tuning section is appended. Existing dot tuning keys/configs
and reference tolerances remain unchanged.

Local checks passed: 35 unittest methods covering host validation, chunk
setup, candidate isolation/routing, benchmark plumbing and automatic report
upload; pre-commit on all changed files; shell syntax and Python syntax.
The candidate routing checks exercise 45 row-count/budget/H/D combinations,
including H=17 and D=257 fallback controls. They verify identical launch
arguments, workspace bytes and Top-K calls where applicable. These checks
do not execute GPU arithmetic or compile the Triton candidate.

Reproduce the CPU checks from the repository root:

```bash
PYTHONPATH=tools python3 -m unittest \
  test_qsa_prefill_host_validation test_qsa_prefill_chunk_setup \
  test_qsa_prefill_single_row_trial test_qsa_prefill_stage_baseline \
  test_qsa_prefill_validation_runner
```

On-card correctness adds 22 cases to the retained production suite. Sixteen
score cases check every one of the four configurations, BF16/FP8, eight H/D
pairs, five query offsets, an empty request, nonzero query base, zero/short
visibility, padded layouts and int32/int64 metadata. Six full-entry cases
check both entries against the same reference under 512/1/0 MiB budgets.
Expected gate with one visible GPU: 161 passed, 1 second-GPU skip and 33
native-vLLM cases deselected. Actual counts/results must come from the card.

## Measurement and reproduction

Use `QSA_PREFILL_AB_KIND=single-row` with
`tools/run_qsa_prefill_validation_ab_h800.sh`. It invokes the full correctness
gate before the comprehensive A/B, then automatically commits and uploads
JSON/TXT/TSV reports to a result branch, including failed-run reports when a
summary exists. Invoke it through a child `bash`; do not enable `set -e` in
the interactive container shell.

The active set contains 38 cases: the previous 24 are retained, plus 12
single-query cases and two 32/32/1 tail cases. The H/D sweep extends the
existing parameterized fixtures to a single query with context 8192 and
adds H=17,D=33 as a fallback control. The tail fixture changes the previous
73-query tail case to query lengths (33,32), retaining contexts (4096,8192),
query base 113 and a 1 MiB budget so the final chunk has exactly one query.
BF16 and FP8 are measured separately. Newly added cases rebuild both sides
in the same worker; they are not compared against historical latency values.

`single-row-ab.tsv` compares candidate `full` against
`full_matmul_baseline`. Both use the same process, inputs, budgets and actual
FlagTree runtime. The baseline is the unchanged current production entry,
not native vLLM. In JSON, `same_public_bytecode` and shared validators/Top-K
should be true; `shared_score` should be false because launchers differ.
`shared_score_kernel` refers only to the common fallback dot kernel: the
eligible single-query path intentionally uses a different score kernel.
Per-chunk `score_path` records which route executed. `vector_score_configs`
and `score_configs` capture actual tuned configurations and binary hashes;
an unused route can have an empty config list. Runtime source/compiler
paths, tool hashes, source snapshots, raw timing rounds and host profiles
are included in reports. Profiler executable availability is recorded;
hardware profiling is not claimed by this run.

Full-entry wall latency is the primary criterion; Graph/isolated score time
help explain the outcome. Reduction order changes must pass the existing
score/selection tolerances without relaxing them. Keep CUDA device 0 in the
default command; the latest report shows jobs on device 1. The known thermal
throttling on long cases means those remain diagnostic and cannot establish
a gain until a stable-temperature repeat is available. Decide keep, split
the eligible H/D region, or reject only after reading returned correctness,
paired timings, dispatch and temperature/clock data.
