# QSA paged prefill: H800 implementation and validation

## Native score illegal-instruction diagnosis (before editing, 2026-10-14)

| Field | Choice |
| --- | --- |
| Evidence | Report e61086b: stock native QSA and both CUDA Top-K paths pass; FlagTree native score launch fails even before Gems runs. Async/sync full suites stop at BF16 H1/D32; H4/D128 also fails |
| Editable | A standalone diagnostic tool and this guide only |
| Read-only | All production operators, original source SHA256, reference tolerances, Top-K implementations and dispatch |
| Inputs | Generate and serialize one set per reproducer with stock Torch on the selected H800; identical tensor values/strides/device shared by all probes |
| Isolation | Fresh CUDA process/cache per compiler probe; score kernel only, no Top-K or Gems launch. Default probes do not import FlagGems; one explicit import-only control tests package initialization effects. Save IR/PTX/CUBIN/metadata before launching |
| Ablations | One change at a time: loop STAGES=1, compiler num_stages=1, TILE_R=16, target sm80, stock ptxas. All are diagnostic configurations, never changed production/native baselines |
| Replay | Launch the same original CUBIN under the opposite runtime without recompilation. Reassemble identical PTX with stock ptxas and observed original flags. Verify input/artifact hashes; reject unsupported launcher metadata |
| Fault location | Optional existing compute-sanitizer memcheck/synccheck; retain PC and SASS. No tools installed, no existing cache deleted, no performance measurements |
| Completion | Mac checks validate transport/routing/artifacts; hardware-specific root cause requires H800 report, not inferred from an illegal-instruction message alone |

| Cache-policy follow-up field | Choice frozen from a4685e8 before GPU follow-up |
| --- | --- |
| Evidence | Both failing CUBINs contain LDGSTS reads of desc[UR1] with no SASS write initializing UR1; PTX contains fractional evict_first createpolicy. Original Stock PTX omits async L2 hints |
| Editable | Diagnostic PTX intervention helper/mode and this guide only |
| Fixed | Original source/input hashes, two reproducer shapes, STAGES=2, pipeline, dot instructions, original ptxas flags and ABI; original report remains read-only |
| Experiments | Original failing CUBIN under Stock; strip only cp.async cache-hint qualifier/operand under Stock and FlagTree; separately change policy fraction 1.0 to 0.5 under Stock |
| Interpretation | Original binary failures stay visible. Patched PTX is explicitly labeled and cannot stand in for native-vLLM correctness/compatibility acceptance |

## Same-process compatibility (before editing, 2026-10-14)

| Field | Choice |
| --- | --- |
| Evidence | Report 2ae99ce confirms the one BF16 index-set difference is exact zero-score ties; four FP8 differences pass the established tolerance. Native BF16 H8/D256 cannot execute on H800 |
| Editable | Native test parameter list, diagnostic runner, this guide |
| Read-only | Production kernels/autotune/benchmarks, pinned native QSA and CUDA Top-K dispatch, all reference tolerances |
| Native coverage | Remove only BF16 H8/D256 from native comparisons at the user's request. Preserve FP8 H8/D256 and independent BF16 default/one-row tests. Full suite now 138 tests, including 33 native comparisons |
| Compatibility | Fresh processes per diagnostic job; each TLE QSA job alternates actual FlagGems and original native QSA in one process/context. Run both launch orders and full suite, async and launch-blocking |
| Failure/reporting | Continue other independent jobs after failure/timeout; nonzero overall exit, logs and summary. No adapter, upstream patch, skip, performance timing or installation |

## Native-vLLM direct alignment (before editing, 2026-10-14)

| Project field | Frozen choice |
| --- | --- |
| Evidence | Code 2cc74c9 passed 105 independent H800 cases in report 8e68728; 22 native comparisons excluded. Native cooperative Top-K had failed in the shared TLE process; no stage diagnostic report was returned |
| Editable | Validation-only subprocess adapter/worker, native comparisons, card runner, this guide |
| Read-only | Production QSA/Top-K kernels, six score candidates, original input builders/references/tolerances, hash-pinned vLLM implementation |
| Build/validation | Python-only. Add native-align mode: same full suite with no exclusions; native preflight before tests. Mac static/orchestration tests; H800 execution required |
| Runtime | Existing Python venv: stock Triton for native child processes; explicit FlagTree/TLE for FlagGems tests. Preserve CUDA compatibility libs and selected GPU. No package install or upstream dispatch replacement |
| Scope/timeout | Functional comparison only, no timing through IPC. Each native case runs fresh with a 300 s timeout; failures are hard failures with logs, never Torch substitution or skip |
| Benchmark/autotune | No production/benchmark/autotune changes. Separate-process results establish functional agreement, not shared-process compatibility or kernel performance |

| Contract field | Choice |
| --- | --- |
| Same inputs | Transfer actual tensor values, dtypes, shapes and strides plus scalar arguments, compression/budget/device. Reconstruct tensors on the same GPU; compare supplied-buffer output against original native output |
| Native source | SHA256-pinned file; original qsa_select_paged_prefill and CUDA cooperative/persistent dispatch. Stock Torch/Triton must import from selected prefix and match metadata |
| Comparison | Preserve all original independent and strict upstream BF16/FP8 checks. Add direct selected-score multiset comparison for synthetic cases using the same tolerances; exact BF16 upstream index sets and -1 padding remain mandatory |
| Coverage | Existing 22 native cases plus BF16/FP8 x (rows=1,K=1024/2048 and rows=65,K=512/1024/2048); native both <=64 cooperative and >64 persistent branches |
| Failure behavior | Input/source/runtime mismatch, timeout, native error or output mismatch fails the suite. Save per-case input/output, fingerprint, manifest and log under the single Reports directory |
| No Torch compute | Production unchanged; CPU/GPU serialization, copies and references belong only to validation helpers |

| Path | Change | Evidence required |
| --- | --- | --- |
| Native subprocess | Explicit opt-in adapter under native-align; no vLLM import in the TLE pytest process | Native actual source/version/hash, original CUDA ops, serialized input identity and native output |
| Existing native comparisons | Run original 22 cases; add explicit direct value comparisons where tests previously only checked each against Torch | Original exact upstream fixtures/tolerances preserved |
| Additional native coverage | K sweep and 65-row batches | Both native Top-K dispatch regimes, BF16/FP8 |
| Shared-process mode | Retain existing full/full-bench behavior | Existing compatibility diagnostic remains separate; no claim of resolving that fault |

## Parameterized heads and dimensions (before editing, 2026-10-14)

| Project field | Frozen choice |
| --- | --- |
| Evidence | Code 6d5fdcb: H800 independent full accuracy 54 passed, 14 vLLM comparisons deselected; native baseline failure remains unresolved |
| Editable | QSA score/validation, synthetic input helper/tests, separate generalized benchmark groups, this guide |
| Read-only | Hash-pinned upstream source, exact upstream fixtures/reference checks, persistent Top-K implementation, six YAML candidates, existing benchmark groups |
| Build/validation | Python-only; local syntax/style/host checks, H800 full independent suite before benchmark, native comparisons separately via full-bench |
| Benchmark/active set | Existing mean operator latency (ms) and per-case baseline/Gems SpeedUp; separate synthetic benchmark H/D sweep (1,32), (3,64), (5,96), (8,256), plus comprehensive (12,192), (17,33), BF16/FP8 |
| Environment/timeout | Mac has no GPU/Torch/Triton; user runs H800 via published branch. JIT/tests up to 30 min, benchmark up to 45 min externally; profiler availability unknown |
| Autotune | Existing score config/six candidates; add NUM_HEADS to key beside HEAD_DIM/IS_FP8. Fixed index fills remain exempt; no new export/registration needed |
| Fallback | No Torch compute fallback; same TLE persistent Top-K; optimization.md/deep_opt.md remain absent |

| Contract field | Choice |
| --- | --- |
| API | Same ten arguments and None return; H and D inferred from Q.shape, no extra public parameters |
| Inputs | Matching BF16/FP8 Q [T,H,D], H>0/D>0; cache [pages,page_size,1,D]; existing SM90/layout/metadata contract |
| Numerics | Finite Q/K and scores; zero-pad heads and reduction dimension, sum ReLU of each real head's dot in FP32, unscaled; no gradients |
| Output | Existing logical compressed indices, contiguous int32 [T,K], -1 padding, ties/order unspecified; K=512/1024/2048 |
| Limits | Metadata accepts positive H/D without a fixed whitelist; actual Triton compilation must fit device resources. This is not a claim of testing arbitrarily large dimensions |
| Reference | Frozen upstream checks unchanged. New synthetic shapes check visible scores and selected score multisets; exact ties can select different equivalent indices. FP8 retains upstream scaled-score rtol=atol=1e-3 |
| Unsupported | H=0/D=0, mismatched cache D, existing unsupported dtype/device/layout/gradient paths; no copies/casts in production |

| Path | Change | Verification/risk |
| --- | --- | --- |
| Host | Validate positive H/D and matching cache; infer launch constants | Invalid zero sizes/cache mismatch; mocked real dispatch |
| Score | Pad H to power of two, D to native dot minimum/power of two; pad small row tiles to M>=16; mask fake rows/heads | Every candidate on original H4 and new small/odd H/D; extra lanes must not write adjacent tiles |
| Empty/select all | Existing fill/return; admit generic H/D | No score/Top-K launch tests |
| Top-K | No changes | Generic H/D selection, short rows, ties and forced one-row chunks |
| Benchmark | Separate parameterized full-op versus Torch/vLLM groups | Preserve all existing shapes; no performance claims before card run |

## Persistent Top-K substitution (before editing)

| Project field | Choice |
| --- | --- |
| Evidence | Report 5f21e689 tests code b69506f: full 67 passed, no skips, 103.10 s; official Torch 2.13.0+cu129/Triton 3.7.1/vLLM 0.31.0+cu129. No new benchmark in that report |
| Editable | QSA dispatch, persistent_topk workspace initialization/TLE guard, QSA tests/stage benchmark, runner/environment support, this guide |
| Read-only | Score kernel/six tune candidates, persistent Top-K selection algorithm/dispatch, all original benchmark shapes and pinned vLLM source |
| Environment | Mac cannot run CUDA or SSH to H800; publish focused commit, card runs full accuracy before benchmark. FlagGems persistent_topk requires TLE; test preflight must check it |
| Benchmark | Same operator mean latency and per-case speedup; retain official full/score/Top-K groups and add old-vs-persistent Top-K stage comparison in the same runtime |
| Autotune | Existing score config unchanged; inherited persistent Top-K scheduling unchanged; fixed-size zero fill exempt from tuning |

| Contract field | Choice |
| --- | --- |
| API/output | Same QSA arguments, dtypes, D=8/128, logical block indices, padding and strict upstream checks |
| Top-K | Repository persistent_topk(logits, visible, output, uint8_workspace, k, max_seq_len=logits_width); next_n=1 behavior for flat visible counts |
| Workspace | Allocate 1 MiB Top-K workspace once per nontrivial QSA call and reuse across chunks; persistent kernel clears its active scratch every launch |
| No Torch compute | Replace inherited workspace slice assignment with a Triton zero fill; no cast/copy/selection fallback |
| Unsupported | TLE unavailable: explicit NotImplementedError, no return to old Top-K or native vLLM production kernel |

| Path | Change | Verification |
| --- | --- | --- |
| Empty/select all | Preserve existing fast paths | Existing no-launch checks |
| Scored selection | Remove row-start allocation/fill, call persistent_topk | Full 67-case suite plus workspace reuse/argument test |
| Scratch init | Fixed Triton byte-zero kernel | Mocked launch/host checks; existing persistent_topk tests on H800 |
| Benchmark | Persistent stage with preallocated workspace; old-stage A/B | Same inputs/runtime, no new baseline relabeling |

Status: code `6d5fdcb` passed 54 independent H800 cases (14 native comparisons deselected) in report `0cf70c5`; persistent Top-K performance was measured in the same FlagTree runtime. Native baseline alignment remains unresolved. The parameterized H/D change above awaits card validation. Historical code `b69506f` passed all 67 native-inclusive cases before the Top-K substitution; that result does not validate the new selection implementation.
The Mac has no Torch, Triton, or CUDA device. GPU correctness and performance
must not be inferred from static checks.

## Frozen reference

- API: `qsa_select_paged_prefill(...) -> None`, not an ATen operator.
- Upstream: https://github.com/vllm-project/vllm/blob/7a877ae34dcf4a7d46de08eb16db4699e1e0047c/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
- Source SHA256: `4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b`.
- Local base: `ebe39aa1c129142490b88301a2e641c2ef786e68`.
- Branch: `codex/qsa-select-paged-prefill-h800`.
- `optimization.md` and `deep_opt.md` referenced by `workflow.md` are absent
  from the checkout and were not found in the workspace. The design below
  follows the available workflow. Performance optimization is deferred at the user's request.

## FP8 and strict upstream coverage follow-up (before editing)

| Project field | Frozen choice |
| --- | --- |
| Editable | QSA prefill score/validation, test helper/tests, upstream benchmark groups, this guide |
| Read-only | Existing Top-K implementation/configuration, pinned upstream source/tests, original BF16 benchmark tuples |
| Build/environment | Python-only; Mac has no Torch/Triton/CUDA. Static/style checks locally, actual compilation and tests on the user's H800 with the existing official vLLM 0.31 environment |
| Validation | Existing quick/full runner; reproduce upstream BF16/FP8 x slack 0/1792 x default/one-row fixtures, BF16 sorted index equality, FP8 SM90 value comparison; D=8 zero-visible fixture |
| Benchmark | Preserve existing BF16 rows; add FP8 to separate upstream groups and D=8 zero-visible group. Operator mean host-inclusive latency in ms, per-case SpeedUp; no acceptance claims before card results |
| Autotune | Retain six score candidates; add HEAD_DIM and IS_FP8 to specialization/tuning keys. Fixed fills remain exempt; existing Top-K remains unchanged |
| Scope | Functional coverage only; user deferred Top-K optimization and performance work. No Torch compute fallback; no new exports or registration needed |

| Contract field | Choice |
| --- | --- |
| Reference | Same SHA256-pinned upstream commit; no reference or tolerance relaxation |
| Dtype/shape | Matching BF16 or float8_e4m3fn Q/cache; H=4, D=8/128; same strides, metadata, output and K contract as before |
| Accumulation | Native Triton dot to FP32; pad D=8 reduction dimension to >=16 (>=32 for FP8). FP8 SM90 dot can use reduced-precision internal accumulation, as upstream |
| BF16 check | Upstream fixtures require exact sorted index sets, including -1 padding, against Torch and pinned vLLM; random exact-tie fixtures retain set-valued checks |
| FP8 check | On SM90 compare selected score multisets scaled by 1/sqrt(D), rtol=atol=1e-3, matching upstream tests; production scoring remains unscaled |
| Unsupported | Other dtypes/H/D, gradients, non-SM90, unsupported layouts, nonfinite scores; explicit exceptions, no cast/fallback |

| Path | Change | Verification/risk |
| --- | --- | --- |
| Host validation | Admit matching BF16/FP8 and D=8/128 | Mixed-dtype and unsupported-shape error tests |
| Score | Parameterize D and pad loads; keep H=4 and existing tiling | Every candidate checked for BF16/FP8 and D=8/128; CUDA compilation pending |
| Select all | Existing index fill, now accepts D=8/FP8 | Exact upstream zero-visible output and positive D=8 cases |
| Top-K | Reuse unchanged repository operator | Strict BF16 and upstream FP8 value checks |

## Project truth table (before implementation)

| Field | Frozen choice |
| --- | --- |
| Current op | vLLM QSA paged prefill; public `flaggems_vllm.qsa_select_paged_prefill` |
| Editable files | New op, NVIDIA tune YAML, exports, new tests and benchmark, reference helper, this guide, card runner |
| Read-only | Existing QSA/Top-K operators, upstream reference, input/reference helpers after first validation; existing dirty FP8 gate files in the primary checkout |
| Build | Python-only change. Card: `python -m pip install --no-build-isolation --no-deps -e .`; development/validation uses `PYTHONPATH=$PWD/src` |
| Validation | `python -m pytest -q tests/test_qsa_select_paged_prefill.py --quick`, then full file; value-based Top-K and score comparison |
| Benchmark | `python -m pytest -s -q benchmark/test_qsa_select_paged_prefill_perf.py --level core --mode operator --warmup 100 --iter 500 --record json`; mean host-inclusive operator latency in ms, smaller is better |
| Active set | BF16, H=4, D=128, page=16/64, requests=1/4, total query rows=1/64/512/2048, compressed width=2048/4096/16384/65536, block K=512/1024/2048; exact tuples in benchmark |
| Aggregation | Report per-shape latency and SpeedUp separately for Torch and pinned vLLM; no aggregation may hide a key-shape regression |
| Acceptance | All functional cases pass; each key shape has baseline/FlagGems latency >=0.9; no measured claims before H800 results |
| Timeout | First GPU test/JIT 20 min; full tests 30 min; benchmark 45 min (external timeout recommended) |
| Profiler | Card: `nsys profile --stats=true ...`; `ncu --set full ...` after narrowing to the score kernel. Unknown availability until the card run |
| Autotune | `qsa_select_paged_prefill_score` through `get_tuned_config` + `libtuner`; key includes chunk rows, query length, logits width, request count, page size, layouts; candidate TILE_R/BLOCK_N/K_TILES/warps/stages in YAML |
| Fallback | No Torch compute fallback. Reuse repository Triton `persistent_topk` (TLE required); unsupported types/devices/shapes raise |

## Contract table (before implementation)

| Field | Contract |
| --- | --- |
| Device | NVIDIA CUDA SM90 (H800/H100); first acceptance target H800 |
| Q/cache | Matching BF16/FP8 E4M3FN Q `[T,H,D]` and K `[physical_pages,page_size,1,D]`, H,D>0 inferred from Q; forward only, no gradients; compilation must fit device resources |
| Layout | Unit innermost stride; positive, non-overlapping outer strides; Q/cache/page-table row padding and input slices supported; contiguous metadata and output |
| Metadata | Same device; page table int32/int64 `[B,P]`; `query_start_loc` int32/int64 `[B+1]`; visible counts int32 `[T]` |
| Metadata values | Caller supplies monotone offsets whose terminal-minus-first value is T; request lengths <= max_query_len; visible counts in `[0,min(capacity,ceil(max_seq_len/ratio))]`; active physical page IDs valid. These device-resident value preconditions avoid host synchronization |
| Output | Caller-owned int32 `[T,token_topk/ratio]`, contiguous; no input/output overlap; filled in place, returns None |
| Parameters | Positive integer compression ratio, divisible token_topk; actual block K in 512/1024/2048; positive max lengths for nonempty input |
| Score | FP32 accumulation; sum across heads of ReLU of each head's Q.K, without sqrt(D) scaling |
| Selection | Logical compressed indices, not physical pages or expanded tokens; valid count min(visible,K), remaining positions -1; order and choice among exact ties unspecified |
| Numerics | BF16/FP8 inputs with finite scores; NaN/Inf and overflow outside supported contract. Signed zero behaves as tied zero. Upstream BF16 fixtures require exact sorted index sets; FP8 uses upstream SM90 score tolerances; exact ties in synthetic inputs allow set-valued selection |
| Empty | T=0 returns without allocation/launch after shape/type checks; zero visible rows emit -1 |
| Unsupported | FP16/FP32, zero heads/dimensions, gradients, non-SM90/non-NVIDIA, non-unit inner stride, noncontiguous metadata/output, overlapping output |
| Reference | Torch compute only in tests/benchmark helper. Pinned vLLM reference only in tests/benchmark; never imported by production |

## Upstream coverage follow-up (before editing)

| Project field | Frozen choice |
| --- | --- |
| Editable | Test input helper, functional tests, benchmark, this guide; only the production logits-budget validation |
| Read-only | Pinned upstream implementation/test source, score/Top-K kernels, tuning candidates, existing synthetic benchmark tuples |
| Validation | Existing quick/full card runner; add exact upstream BF16 prefill inputs and Torch/pinned-vLLM comparisons |
| Benchmark | Separate upstream full-operator group: slack 0/1792, default/one-row chunks; same timing/aggregation as existing benchmark |
| Environment | Mac has no Torch/Triton/CUDA; local static/style checks only, H800 execution pending |

| Contract field | Choice |
| --- | --- |
| Upstream source | `tests/models/qwen4_exp/test_qsa_reference.py` at frozen commit `7a877ae34dcf4a7d46de08eb16db4699e1e0047c` |
| Prefill inputs | BF16 Q `[36,4,128]`, cache `[128,24,1,128]`, shuffled table `[2,64]`, query lengths `[3,33]`, sequence lengths `[5120,4224]`, token_topk 2048, ratio 4 |
| Bounds | max_seq_len 5120/6912; expected clipped logits widths 1280/1536, versus page-table capacity 1536 |
| Mixed batch | Preserve upstream decode query length 1/2/3/4, then pass only the 33-row prefill slice with nonzero query_start_loc base |
| Scope | BF16 H=4/D=128 only; upstream FP8 and D=8 cases remain outside the production contract |

| Path | Change | Verification |
| --- | --- | --- |
| Score/Top-K | Existing kernels/configs | Selected values, score values, ranges, uniqueness and padding against independent Torch reference |
| Workspace | Accept budget 0 to match upstream one-row chunk behavior; negative/invalid budgets still raise | Assert actual chunk offsets/counts and clipped widths for all four upstream prefill cases |
| Official comparison | Exact upstream BF16 input builders, pinned implementation | Both implementations independently checked against Torch |
| Performance | Separate full-operator upstream group | Report default and one-row chunks separately; no measured claims before H800 results |

## Implementation paths (before implementation)

| Path | Trigger | Implementation | Tuning | Validation/benchmark | Risk |
| --- | --- | --- | --- | --- | --- |
| Empty/error | Empty T or invalid shape/type | Metadata checks and return/raise | None | Empty and contract tests | Invalid device metadata values are caller preconditions |
| Select all | Upper bound on visible blocks <= K | Triton index/padding fill, no Q.K needed | Exempt: simple fixed-size index fill | Zero/short contexts | Must retain logical index and -1 convention |
| Prefill score | Visible upper bound > K | Request/tiled-query/tiled-key `tl.dot`, explicit strides and nonzero query base | YAML + libtuner | Ragged requests, shuffled pages, chunks, score reference | Tensor-core rounding and incomplete coverage |
| Top-K | Per-row score selection | Repository `persistent_topk` on `[0,visible)` | Existing operator's policy; no changes | Random/tied/short rows and K sweep | Existing TLE/non-TLE scratch and tie behavior |
| Workspace | Nontrivial selection | One per-call logits allocation reused across chunks; 1 MiB uint8 persistent Top-K scratch | Fill exempt; score tuned | Force 1 MiB budget and cross-request chunks | Only logits is bounded by configured budget; Top-K scratch is additional |
| Unsupported | Outside contract | Explicit exceptions, no conversions or compute fallback | None | Error tests | Unsupported types/shapes raise |

The public Python entry is exported through `ops.__all__` and the existing
top-level export mechanism. It has no ATen schema, so it must be called
explicitly by the vLLM plugin/model integration; registering an invented ATen
name would not activate it. Plugin changes are outside this repository change.

The score workspace is bounded by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default
512 MiB), except that at least one logits row must fit. A zero budget forces
one row per chunk, matching upstream; negative/invalid budgets raise. Top-K scratch is
additional and must be included in model memory profiling. Workspaces are
reused within a call, not cached globally. CUDA graph/model memory-profiler
integration is not claimed by this operator-only change.

## Running on the card

From the existing card repository, inspect local changes before bringing in
the branch. An additional worktree keeps the current checkout and branch
intact and also freezes the existing Top-K dependency used by this change:

```bash
git status --short
git fetch https://github.com/cheersluvs/FlagGems-vllm.git codex/qsa-select-paged-prefill-h800
qsa_card_worktree=$(mktemp -d /tmp/flaggems-qsa-h800.XXXXXX)
git worktree add --detach "$qsa_card_worktree" FETCH_HEAD
cd "$qsa_card_worktree"

# Activate the existing Torch + Triton/FlagTree H800 environment first.
# Optional: PYTHON=/absolute/path/to/venv/bin/python
bash tools/run_qsa_prefill_h800.sh torch-only
```

The first run needs no vLLM baseline. The independent accuracy tests run;
the pinned vLLM comparison may be explicitly reported as skipped. A second
SM90 device is optional: its device-guard test skips on a one-GPU environment.
Any skipped accelerator tests indicate an unsuitable environment, not a pass.

For the official comparison and performance results:

```bash
bash tools/run_qsa_prefill_h800.sh all
```

This downloads the pinned reference into a fresh report directory and checks
its SHA256. The existing vLLM build must provide the reference's imports and
the `persistent_topk` and `cooperative_topk` extension symbols. A missing or
incompatible baseline is a hard failure for this mode, never a Torch fallback.
For an offline card, pass `QSA_VLLM_REFERENCE_FILE=/path/to/the/pinned/file`.
The runner installs no packages and changes no repository files. It disables
vLLM plugins for the isolated comparisons, verifies the imported source path,
and writes environment details, commit ID, pytest text and JSON to the printed
`/tmp/qsa-prefill-h800.*` directory.

Modes `quick`, `full`, and `bench` run individual stages. `full-bench` runs full accuracy then benchmarks, stopping on failure. Benchmark timing uses
operator mode (host work and device execution), after explicitly warming up
both functions. Separate results cover full selection against Torch, full
selection against pinned vLLM, score against pinned vLLM, and Top-K against
pinned vLLM. The Torch reference includes host metadata reads; performance
acceptance must use the official comparison and report key shapes separately.
The score/Top-K stage measurements use a full preallocated logits matrix and
isolate those stages; they are not a replacement for the chunked full-operator
measurements. Full benchmark defaults to a 512 MiB logits budget.

### Shape provenance and upstream coverage

The original `CASES`, `CORE_SHAPES` and `EXTRA_SHAPES` are synthetic coverage
and performance sweeps. They are not model traces or a verbatim copy of the
upstream test matrix. BF16, H=4, D=128, ratio=4 and block K=512 agree with the
upstream fixture; K=1024/2048, page=64 and the long-context tuples are additional
implementation coverage. Context lengths in those tuples are compressed block
counts: 65536 blocks at ratio=4 represent 262144 logical tokens.

The named input builders reproduce the BF16/FP8 selection inputs from
`test_qsa_prefill_selection_correctness` and the prefill slice of
`test_qsa_split_selection_correctness` at the pinned upstream commit.
All these functional cases run in both quick and full modes:

- page=24, cache `[128,24,1,128]`, table `[2,64]`, Q `[36,4,128]`, query
  lengths `[3,33]`, logical sequence lengths `[5120,4224]`;
- max_seq_len slack 0/1792, checking clipped logits widths 1280/1536;
- default 512 MiB budget and budget 0, checking all one-row chunk offsets,
  including the transition between requests;
- decode lengths 1/2/3/4 preceding a 33-row prefill slice, retaining its
  nonzero query_start_loc base;
- independent Torch score/selection checks and separate pinned-vLLM selection
  comparisons; FP8 now runs the same 8-case prefill matrix. The upstream D=8 zero-visible fixture checks exact -1 output, with additional D=8 scored-selection and every-candidate checks.

The benchmark adds a separate upstream full-operator group with slack 0/1792
and default/one-row chunks, against both Torch and pinned vLLM. It preserves
the original synthetic tuples and reports the chunk modes under distinct
operator names. FP8 uses the same groups with its dtype recorded separately; the D=8 zero-visible fixture has a separate group. These are regression fixtures, not model traces. New coverage awaits H800 results.

## Current verification and delivery limits

FP8/D=8 follow-up local checks: Python syntax and all configured style hooks
passed; 27 checks executed the actual host functions with mocked allocations
and launchers (not GPU kernels). The static matrix has 63 quick / 67 full
cases; collection and CUDA execution still require H800. Every score candidate
is exercised for BF16/FP8 and D=8/128. The benchmark preserves the original
BF16 groups and adds eight FP8 upstream rows plus two D=8 zero-visible rows
(42 records total). Existing Top-K code/configs are unchanged.


- Mac: Python compilation, exact upstream argument-name/order comparison,
  YAML config validation, no-Torch-compute host audit, shell syntax checks,
  and all configured pre-commit hooks passed.
- Upstream coverage follow-up: changed Python sources compile and configured
  style hooks pass. Executing the actual Python host function with mocked
  allocations/launchers confirms widths 1280/1536 and 1/36 chunks for the
  four slack/budget combinations, plus negative/invalid budget rejection.
  This exercises no GPU kernels and is not a GPU correctness result.
- H800 baseline b5fae92: quick 47/full 51 passed with no skips. Official core full-operator speedups 0.389–0.425, upstream default 0.368–0.373, one-row 0.630–0.638; performance is below 0.9. This coverage-only change needs a new H800 run. Top-K optimization was reverted at the user's request.
- Autotune: six score candidates; fill kernels use fixed launches because
  they only initialize indices/row starts. Reused Top-K retains its existing
  tuning/dispatch policy; this change does not retune it.
- No external library production path; no Torch compute fallback. Autograd,
  other head configurations, other devices, model memory profiling,
  CUDA graph integration and vLLM plugin wiring are outside this first scope.
- Export/integration: `flaggems_vllm.qsa_select_paged_prefill` is callable;
  excluded from ATen registration because this API has no ATen schema.

## H800 report and container-local official baseline setup

Report commit `4784cd9b8403a27b4f8b4697f8ae1bb9eaf9b190` on
`codex/qsa-h800-results-20261013-065119` records the quick run of code commit
`e0e9cff867dcc6116b7f165ad4b4552a4dcadc82`: **38 passed, 9 skipped**, in
62.80 seconds. The nine skips are all pinned-vLLM comparisons, blocked by
missing `vllm._C_stable_libtorch` in the installed `0.24.0+flagos` distribution.
The added BF16 upstream fixtures, all six score candidates on their test
input, and noncurrent-device validation passed. Full accuracy, official
comparisons and performance acceptance remain pending.

Reported container: Ubuntu 24.04.1, NVIDIA H800, host driver 570.124.06,
Torch 2.10.0+cu128 and Triton 3.6.0. No nvcc/Docker executable was reported.
`tools/setup_qsa_vllm031_h800.sh` prepares a separate environment inside this
container using the official vLLM 0.31.0 CUDA 12.9 wheel. It extracts NVIDIA's
SHA256-checked `cuda-compat-12-9` package into a temporary directory rather
than installing system packages. Both package and wheel are SHA256-pinned.
NVIDIA's forward-compatibility matrix lists
this package as compatible with the 570 driver family on data-center GPUs.
It then checks CUDA extension symbols, BF16 matmul and Triton JIT before
running the original pinned-reference runner. Installation and preflight
succeeded in the second H800 report below. The compatibility package's SHA256, library paths
and symlinks were checked on the Mac; shell syntax and embedded Python compile.

```bash
bash tools/setup_qsa_vllm031_h800.sh all
# Or install/preflight only, then use the printed activate_qsa.sh path:
bash tools/setup_qsa_vllm031_h800.sh setup
```

The setup targets Linux x86_64, downloads several GB of dependencies, and
prints its environment directory and setup log. It preserves the existing
FlagOS Python environment and host driver. Reusing the printed activation
file selects the extracted compatibility libraries for subsequent QSA runs.
Changing the environment changes the benchmark baseline; compare FlagGems
and vLLM in the same new environment and record its version details.

Report commit `31dfdec27c637d60db5c0fc9073945760237bda3` on
`codex/qsa-h800-results-20261013-074711` records code commit
`9395ef639d27bf90473b9623667f6ad9506079b6`. The official environment installed
Torch 2.13.0+cu129, Triton 3.7.1 and vLLM 0.31.0+cu129. Extension-symbol,
BF16 matmul and simple Triton JIT preflight checks passed. Quick accuracy
reported **27 passed, 20 failed**. The first failure occurred while calling
the pinned upstream reference's `cooperative_topk`: CUDA illegal instruction.
Later cases failed during input preparation with the same CUDA error, so
they do not establish independent numerical failures. CUDA launches were
asynchronous; the reported call alone does not identify the failing kernel.
The runner stopped at quick accuracy; full accuracy and benchmarks did not
run. Preflight logged Triton 3.7.1, whereas the runner logged 3.6.0; the actual
import paths and any version changes still need diagnosis.

Reuse the printed environment activation file, then run:

```bash
"$PYTHON" tools/diagnose_qsa_prefill_h800.py
```

This diagnostic writes a new `/tmp/qsa-prefill-h800.*` report directory and
runs each check in a fresh process with `CUDA_LAUNCH_BLOCKING=1`: import
fingerprints and loaded CUDA driver libraries; direct official cooperative
and persistent Top-K calls on `[32, 8192]`, K=512; cooperative Top-K after
FlagGems import; full independent Torch comparisons; and the original first
pinned-vLLM comparison. Direct Top-K checks use distinct nonnegative FP32
scores and validate the selected index sets. Each check has a 900-second
timeout, and remaining independent checks run after a failure. The final
exit code is nonzero if any check fails. This is diagnostic evidence, not a
performance run or a replacement for full official comparisons. It does
not change upstream dispatch or any operator implementation.

The normal runner now stops at the first accuracy failure to avoid noisy
cascading CUDA errors. Its `torch-only` mode explicitly deselects official
comparisons even when vLLM imports successfully.

Report `74a94a1fb71b50a30443dc557921bd1ec0215af4` (results branch
`codex/qsa-h800-results-20261013-081043`) ran code `1cd2c01` with
`CUDA_LAUNCH_BLOCKING=1`. All six independent diagnostic jobs succeeded:
three direct Top-K checks, **41 passed / 1 skipped** in full Torch comparisons,
and **1 passed** in the isolated first vLLM comparison. The skip required a
second visible SM90 GPU; nine official comparisons were deselected in the
Torch-only run. Import fingerprints consistently showed Triton 3.7.1 from
the new venv, and the loaded libcuda was the extracted 575.57.08 library.

Report `f294c179062eef92d22b347ef61e10de7afd39a3` (results branch
`codex/qsa-h800-results-20261013-081721`) reran the normal runner on the same
code: **27 passed / 1 failed** at the original cooperative Top-K call.
Its environment log again showed Triton 3.6.0. It stopped before full accuracy
or benchmarks. These reports do not establish whether synchronization,
import-path differences or preceding tests caused the differing outcomes.
Normal asynchronous execution and performance acceptance remain unresolved.

The next diagnostic keeps the operator and upstream reference unchanged:

```bash
"$PYTHON" tools/diagnose_qsa_prefill_h800.py --mode matrix
```

`--mode async` removes `CUDA_LAUNCH_BLOCKING`; `--mode sync` sets it to 1
(the default). Matrix mode runs independent async checks and the combined
quick suite, then sync environment checks and the combined quick suite.
It records actual Torch/Triton versions and source paths inside pytest after
collection, without introducing synchronization there. Environment probes
compare file execution, the current-directory import path used by `python -`,
and the runner's appended inherited PYTHONPATH. A separate launch-environment
JSON records the inherited PYTHON/PYTHONPATH and selected interpreter. These
are diagnostic runs; no performance numbers are collected.

Report `f0a2a09b4d4ee873a172a361a78f8abbd0563b25` (results branch
`codex/qsa-h800-results-20261013-083350`) ran code `f152286` and passed all
13 matrix checks. Async full Torch accuracy was **42 passed**, including
the second-device guard. Combined quick accuracy was **47 passed** in both
async and sync modes, including all nine pinned-vLLM comparisons. All import
probes and pytest fingerprints used Triton 3.7.1 from the selected venv.
The pinned reference hash and extracted libcuda path matched the setup.

Report `4d1181491307280f4fa4d3e53c15e28c2994115c` (results branch
`codex/qsa-h800-results-20261013-083900`) ran the normal runner on `f152286`:
**27 passed / 1 failed**, again stopping before full accuracy or benchmarks.
Its new fingerprints expose the environment mismatch: PYTHONPATH contained
`/opt/flagtree`, and Triton was imported from `/opt/flagtree/triton/__init__.py`
with module version 3.6.0, while installed distribution metadata reported
3.7.1. Clearing PYTHONPATH before launching Bash did not remove this path
inside the runner; the exact reinsertion mechanism has not been inspected.

The runner now sets PYTHONPATH to only the current repository's `src` and
root **inside the script**, after shell startup, rather than appending inherited
paths. It clears PYTHONHOME and disables the user site. Official-reference
modes fail preflight if Torch or Triton imports from outside the selected
Python prefix or its module version differs from distribution metadata.
`torch-only` can still use the selected Python's installed FlagTree without
this official-baseline check. No operator, autotune candidate or pinned
reference changed. The path isolation has a local shell-startup regression
check; corrected full accuracy and performance still require a card run.

Official installation sources:
- https://github.com/vllm-project/vllm/releases/tag/v0.31.0
- https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/

After operator validation, the plugin's prefill path can call the exported
function with the existing ten arguments. Keep compressed-index expansion
and decode behavior in their existing caller. Confirm that this entry is
actually reached in a model run before reporting end-to-end acceleration.

## Running the persistent Top-K trial

First check the current Python/runtime without installing or changing packages:

```bash
"${PYTHON:-python}" tools/check_qsa_tle_h800.py
```

The check prints actual Python/Torch/Triton/TLE source paths and probes rows
1 and 33 twice with the same workspace, checking exact indices. Missing TLE,
missing CUDA, compilation/execution failure or timeout returns nonzero. It is
a readiness check; full QSA accuracy still needs to pass.

```bash
bash tools/run_qsa_prefill_h800.sh full-bench
```

If the current Python uses stock Triton without TLE, the trial cannot run.
An existing TLE module directory may be explicitly selected (for example,
if `/opt/flagtree/triton` exists); do not append inherited paths silently:

```bash
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "${PYTHON:-python}" tools/check_qsa_tle_h800.py
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  bash tools/run_qsa_prefill_h800.sh full-bench
```

The runner checks that Triton and TLE originate from that explicit directory
and Torch from the selected Python prefix. This runtime also compiles the
pinned vLLM score baseline, so old stock-Triton timings must not be used to
isolate Top-K gains. The extra six-row stage benchmark directly compares old
prefill Top-K with persistent Top-K using identical inputs/runtime. Its
SpeedUp uses the old FlagGems stage as baseline, separately from vLLM results.
The core benchmark now has 48 records across 11 tests; the full functional
suite has 68 cases (static counts, not results for the new code).

Local syntax/style checks, 27 mocked QSA host checks and 109 mocked persistent host/state-init checks passed. Those checks
confirm one uint8 workspace per call, reuse across chunks, correct compressed
width bound/K and unchanged empty/select-all paths; no GPU kernel ran on Mac.
Persistent selection kernels, score kernels and tune candidates are unchanged.
The only persistent implementation changes are an explicit TLE error and a
fixed Triton scratch zero fill replacing the old Torch slice assignment.

## TLE trial: isolate native baseline failures

The supplied log for code `2701be8` reports successful TLE readiness probes
(rows 1/33, workspace reused twice), followed by **38 passed / 1 failed** in
full accuracy. The failure is reported at native vLLM `cooperative_topk` under
explicit FlagTree 0.6.1 / Triton 3.6.0. An asynchronous score-kernel failure
can also surface at this launch. The full suite stopped before benchmarks;
this log does not establish complete QSA alignment or performance.

Use the same Python and CUDA compatibility-library environment as that run:

```bash
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_prefill_h800.py --mode tle
```

Seven fresh processes compare stock Triton with explicit TLE: pure native
cooperative Top-K, upstream QSA, native persistent Top-K under TLE, upstream
QSA after a validated FlagGems QSA call, and the preceding quick tests plus
the first native comparison. The upstream QSA probes synchronize before and
after each original `_topk` call, recording whether the score stage or native
selection first fails. No native dispatch is replaced. Diagnostic failures
are recorded and later jobs continue; the final exit code is nonzero if any
job fails. These synchronized probes must not be used as performance data.
The old diagnostic modes now also preserve an explicitly selected TLE path.

For independent full correctness followed by performance in the TLE runtime:

```bash
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  bash tools/run_qsa_prefill_h800.sh torch-bench
```

This mode explicitly selects `not vllm` for both suites and records that
selection in `selection.txt`. It retains the independent upstream fixtures,
BF16/FP8, D=8/128 and chunking tests, and only proceeds to timing if they pass.
It collects full-operator versus Torch and persistent versus old FlagGems
prefill Top-K measurements in the same runtime. These results do not provide
a speedup against vLLM or a passed native-comparison gate. `full-bench` retains
all native comparisons and still stops on failure. No production kernel,
selection algorithm, tune candidate or pinned reference changed in this fix.

Parameterized H/D implementation: heads are padded to a power of two, small
query-row tiles are padded so the dot M dimension is at least 16, and fake
rows/heads/dimensions are masked. The H4/D8/D128 path uses the same tensor
shapes and six YAML candidates. NUM_HEADS now separates autotune cache keys.
Positive H/D are admitted by metadata validation; very large tensors remain
subject to GPU compiler/resource limits, as in the parameterized upstream.
No production Torch compute, Top-K change, or new registration is introduced.

Local checks passed: 61 actual-host metadata/dispatch checks and 360 CPU
executions of the actual kernel AST with NumPy across all six tiles, padded
layouts, nonzero query base, empty requests, short rows and chunk tails. The
CPU checks poison padding and verify both values and exact write masks; they
do not exercise Triton JIT, Tensor Core instruction rounding or CUDA execution.
Style/syntax checks passed. All new H/D cases await H800 compilation and
accuracy/performance validation.

Static collection counts: full 127 cases (105 independent / 22 native vLLM),
quick 99 (77 independent / 22 native). The core benchmark adds eight rows each
to separate Torch/vLLM parameterized groups; torch-bench selects six tests and
29 measurements. Existing upstream fixtures and all their checks are unchanged.

## Run native direct alignment

Use the existing official vLLM 0.31.0 Python/CUDA compatibility environment:

```bash
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  bash tools/run_qsa_prefill_h800.sh native-align
```

`QSA_NATIVE_REFERENCE_PYTHON` optionally selects another existing official
Python; otherwise the runner uses the same interpreter with a clean import
path for each native child. It does not install packages. The worker validates
actual Torch/Triton source paths/versions, SM90, native extension location,
vLLM version and the unchanged QSA source hash, and rejects a TLE override or
recursive adapter. A native cooperative Top-K preflight runs before pytest.

The mode runs all **139 full cases**, including **34 native comparison
cases** and the 105 independently validated cases. The native cases require
36 reference executions because each empty/select-all test contains two calls.
Coverage includes the previous 22 cases, ten dtype/K/batch-routing cases, and
two empty/select-all cases. One failed child does not stop later comparisons;
any failure still yields a nonzero exit status. Native compiler/resource limits
are reported as failures, not skipped or hidden by changing upstream tiling.

Both runtime fingerprints and original native outputs are saved per case.
The adapter transfers actual tensor values/shape/stride/dtype and scalar
arguments rather than regenerating inputs from a seed. The native child uses
CUDA_LAUNCH_BLOCKING=1 for accuracy/error localization; the FlagGems process
retains its normal launch mode. Production kernels/configuration are unchanged.
Strict upstream BF16 sorted-index equality and FP8 score tolerances remain.
Additional direct score comparisons use the existing tolerance/tie contract.
The adapter explicitly rejects benchmark use, and native-align performs no
benchmark. This run establishes functional agreement only; shared-process
TLE/native compatibility and performance comparison remain separate work.

Return the whole printed Reports directory, including `native/`: per-case
`input.pt`, `output.pt`, `gems-output.pt`, native fingerprint/log, input manifest
and alignment metadata. `native-summary.json` at the top level aggregates
fingerprints, log tails, comparison results and whether sorted indices match;
it distinguishes completed native executions from passed pytest comparisons.
The summary is written on failure as well, preserving the original exit code.

Local syntax/style and orchestration checks passed: real adapter/worker code
under mocked Torch/CUDA preserves values/strides/device/budget and checks all
hashes; native errors/timeouts/mismatches fail; benchmark misuse is rejected;
runner selects every case, saves summaries on success/failure and performs no
IPC timing. Existing input builders/reference functions/tolerances and all
production operator files are unchanged. No GPU ran on Mac; H800 results for
this native mode are pending.

## Native alignment report 4265a12 (2026-10-14)

Code ae3c122 completed **138 passed / 1 failed** in 259.50 s: all 105
independent cases passed; 33 native comparison cases passed and one failed.
There were 36 native executions, 35 completed. Their input/source hashes,
stock Triton 3.7.1 source paths, SM90/device index and native extension paths
were checked against the report manifests; FlagGems used FlagTree/Triton 3.6.0.

The sole failure is original vLLM's BF16 H8/D256 scoring kernel, before Top-K:
shared memory required **335872 bytes**, H800 limit **232448 bytes**. No native
output exists for this case. The identical H/D dtype scope passed our prior
independent default/one-row tests. This establishes a native resource limit
for that pinned implementation/configuration on H800, not an output mismatch;
the failing comparison remains visible and the full native suite is not marked
fully passed. No upstream tiling change or skip was introduced.

All strict BF16 upstream fixture comparisons passed. Across 35 completed
executions, 29 had identical sorted index sets and six differed but passed the
existing direct selected-score checks: five FP8 and one synthetic BF16 H1/D32.
The results branch contains the top-level summary but omits the native/ raw
tensors, so exact ties versus tolerated numeric differences cannot yet be
confirmed for that BF16 case.

Audit the saved card tensors without executing QSA again:

```bash
"$PYTHON" tools/audit_qsa_native_outputs.py /tmp/qsa-prefill-h800.UBvV0t
```

The tool reads the six differing cases, validates input/output checksums,
recomputes the independent FP32 Torch scores with TF32 disabled, and records
both full-selected and changed-only score differences. BF16 non-exact ties or
changed scores outside the existing tolerance return nonzero. FP8 retains the
existing scaled-score tolerance. It writes native-output-audit.json into the
original report directory; return that JSON. This is a saved-output audit,
not a rerun of either implementation or a performance measurement.

## Saved-output audit 2ae99ce and compatibility run

The audit of the previous native run reports `needs_followup=false`: 29
completed executions have identical index sets, two differ only at exact
zero-score ties (one BF16, one FP8), and four FP8 cases differ within the
existing `score/sqrt(D)` tolerance (`rtol=atol=1e-3`, maximum absolute delta
0.00125432). This audits saved outputs and does not add another GPU test run.

At the user's request, BF16 H8/D256 is now removed from the native test
parameter list, rather than marked skipped or changing upstream tiling.
Independent BF16 H8/D256 default/one-row tests and native FP8 H8/D256 remain.
The full suite is expected to collect 138 cases, including 33 native
comparisons (35 native executions in isolated native-align mode).

Run shared-process compatibility using the existing vLLM 0.31.0 environment:

```bash
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_prefill_h800.py --mode compat
```

This schedules 18 fresh processes: nine normal asynchronous checks, then the
same nine with `CUDA_LAUNCH_BLOCKING=1`. Three stock-Triton controls exercise
native cooperative Top-K, native persistent Top-K (65 rows), and original QSA.
The six FlagTree checks inspect imports, exercise both native Top-K paths after
FlagGems import, alternate Gems/native QSA in both orders, and run the full
138-case accuracy suite in one process. The isolated native adapter is
rejected. Source paths, runtime versions, PIDs and launch environment appear
in the logs. No production implementation or upstream dispatch is patched.

Each alternating QSA probe uses BF16/FP8, H4/D128, rows 3/65, K
512/1024/2048 and logits budgets 512/1 MiB, twice in the selected order. Each
whole operator call is synchronized and validated, without inserting waits
between the original native score/Top-K kernels. The ordinary full pytest
process remains asynchronous in the first half of the matrix and is the
stronger check for normal execution. Both halves must pass to claim
compatibility within this test scope; sync-only success is insufficient.

Return the printed Reports directory, especially `compatibility-summary.json`,
`diagnostic-summary.json`, both `accuracy` JSONs and all job logs. A failed
job or a 900-second job timeout produces a nonzero overall exit while later
jobs continue in fresh CUDA contexts. The report distinguishes stock controls,
async FlagTree checks and sync FlagTree checks. This run collects no timings;
model integration and CUDA Graph compatibility are outside its scope.

Mac syntax/style checks and actual-runner mock checks passed: exact native
parameter removal, 18-job runtime/environment routing, both repeated QSA
orders, output validations, failure/timeout continuation, summary classification
and isolated-adapter rejection. The previous diagnostic/independent benchmark
routing checks also passed. No CUDA kernel executed on Mac; the H800
compatibility matrix result is pending.

## Compatibility evidence e61086b and score fault probes

Code 5e39c8d ran all 18 compatibility jobs: 12 passed, six failed. All stock
controls, FlagTree imports and direct native cooperative/persistent Top-K
checks passed. Both alternating QSA orders failed under both launch modes.
Each full pytest process stopped at BF16 H1/D32 with 55 passed / 1 failed;
the remaining tests were not executed in that process. The removed native
BF16 H8/D256 case was absent.

The blocking H4/D128 native-first probe reports the illegal instruction at
the original `_qsa_mqa_paged_prefill_kernel` launch, before any Gems launch or
native Top-K. The asynchronous error surfaces later at cooperative Top-K.
This identifies the failing score stage, but does not yet distinguish invalid
generated instructions, synchronization/operand encoding, a launcher fault,
or cached binary/toolchain issues.

Run the score-only diagnostic in the selected card virtualenv:

```bash
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_native_score_h800.py
```

The preparation process uses stock Triton/Torch to save exactly one input set
for each reproducer: BF16 H4/D128 (3 rows, 8192 keys) and BF16 H1/D32 (packed
3/7 queries, 1025/1537 keys, query base 19). Every probe verifies the same input
SHA256 and reconstructs original strides/device. Default score probes import no
FlagGems package; the explicit `import-gems` probe imports it before native QSA
to detect initialization effects. No probe launches a Gems or Top-K kernel. The pinned original
score kernel source is copied without edits and checked by SHA256.

There are 22 score probes plus preparation. Eight per reproducer compile the
original kernel with stock or FlagTree, or apply one explicitly labeled
control/change under FlagTree: import FlagGems only, loop `STAGES=1`, compiler
`num_stages=1`, `TILE_R=16`, target `sm80`, or stock ptxas. All other native
BF16 defaults remain TILE_R64/BLOCK_N64/K_TILES16/STAGES2/warps4. The two
stage experiments are separate. The sm80 experiment must actually have sm80
metadata and no WGMMA; it changes the architecture's entire compiler path
and alone cannot prove which specific Hopper lowering pass is defective.
The ptxas override must resolve to the recorded stock assembler; compare
PTX checksums before attributing a changed result to the assembler alone.

Two more probes per reproducer replay the original Stock CUBIN under FlagTree
and the original FlagTree CUBIN under Stock. They reconstruct saved
signature/constants/metadata and verify binary checksums without recompiling.
Unsupported cooperative/cluster/scratch ABIs are rejected explicitly.
Reconstruction or binary-load errors are recorded separately from a GPU
execution failure. A failing FlagTree CUBIN under both launchers, together
with a working Stock CUBIN under both, would localize the defect to the
produced binary/toolchain rather than requiring a FlagTree launcher.

One additional probe per reproducer replaces only the assembler on the
original FlagTree PTX: retain its bytes, target and all observed original
ptxas options, use stock ptxas, and replay the resulting CUBIN under Stock.
The tool observes the original assembler invocation during compilation and
rejects missing/ambiguous invocation data; it never guesses options. It saves
the reassembly command and output and verifies unchanged PTX bytes. This is
stronger assembler evidence than changing the compiler's ptxas path, which
can also change its selected PTX version. New SASS is captured when available.

If `compute-sanitizer` already exists, four additional fresh-process jobs run
memcheck/synccheck on the original FlagTree score probes, for fault PC and
synchronization evidence. The tool and version are recorded; nothing is
installed if unavailable. See the [NVIDIA Compute Sanitizer guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Each job uses a new cache directory and saves TTIR, TTGIR, LLVM IR, PTX, CUBIN,
original metadata and available SASS before binary loading. Actual compiler,
driver and libtriton sources/hashes and assembler versions are recorded. No
existing cache or runtime file is changed. Failure stages are distinct:
compilation, replay construction, binary load, launch, synchronization and
FP32 reference validation. Failures/timeouts do not prevent later jobs.

Return **all top-level `.txt` and `.json` files**, particularly
`score-diagnostic-summary.json` and every `*-artifacts.json`. Those compressed
JSON bundles include the full compiler artifacts even when a report uploader
only copies top-level text/JSON files. The raw CUBINs and inputs remain under
the printed card Reports directory for replay. A nonzero exit is expected if
the fault reproduces. No performance numbers are collected.

Mac syntax/style and mocked actual-worker/runner checks validate input binding,
one-variable configurations, prelaunch artifact/checksum bundles, all failure
stages, no-recompile replay and corruption rejection, identical-PTX reassembly
with original options, stock/FlagTree import
routing, unique caches, sanitizer availability and continuation. These checks
do not determine the faulting GPU instruction. The specific root cause is
pending this H800 report.

## Score report a4685e8: fractional async cache policy

Code ac2ce7d completed 22 score probes (6 passed, 16 failed), with no sanitizer
installed. Every original Stock score binary passed. Both original FlagTree
score binaries failed at score launch, with or without FlagGems import and
with fresh caches. Stock CUBIN replay under FlagTree passed; FlagTree CUBIN
replay under Stock failed. Both assemblers are CUDA 12.8.93; reassembling the
same original FlagTree PTX with Stock ptxas yielded byte-identical CUBINs and
the same error. This excludes a FlagTree-launcher-only defect or different
assembler output in these controls.

Loop `STAGES=1` passed both inputs and removed the asynchronous copies.
Setting only compiler `num_stages=1`, reducing TILE_R to 16, or targeting
sm80 did not help. The sm80 binaries contain no WGMMA and still fail, so
WGMMA is not required to reproduce the fault. The original H4 and H1 shared
memory allocations are 106496 and 20480 bytes, respectively, within H800's
limit.

The more specific static finding is in the async-copy cache-policy path:
original FlagTree PTX contains respectively 16/4 copies with
`cp.async.cg.shared.global.L2::cache_hint`, plus two
`createpolicy.fractional.L2::evict_first.b64 ..., 1.0` instructions. In both
SASS files their LDGSTS instructions consume `desc[UR1]`, while no instruction
writes UR1; the policy initialization has disappeared. Original Stock PTX
has no fractional policies/async L2 hints and its SASS has no such undefined
descriptor read.

This matches the mechanism in the [NVIDIA user report](https://forums.developer.nvidia.com/t/nvcc-13-1-ptxas-codegen-bug-createpolicy-fractional-dropped-on-sm-90/368663)
of ptxas dropping fractional-policy initialization while retaining the async
cache descriptor. That report used CUDA 13.1/H20; our observed tool is
12.8.93/H800, so it does not establish identical version boundaries.
[Triton PR 11660](https://github.com/triton-lang/triton/pull/11660) disables
fractional policies for cp.async to work around a ptxas miscompile; its source
comment identifies Blackwell/13.3.33. This supports the class of failure but
does not replace a controlled test of our binary.

Complete that test using the original card report including its raw inputs:

```bash
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_native_score_h800.py \
  --cache-hint-report /tmp/qsa-prefill-h800.kwkjy154
```

Eight fresh processes reuse the saved input/ABI/binary artifacts. For each
reproducer they replay the original FlagTree CUBIN under Stock as a failing
control; separately reassemble PTX after removing only the cp.async cache
qualifier and policy operand, or changing only the two policy fractions from
1.0 to 0.5. The no-hint binary is tested under both Stock and FlagTree.
Pipeline stages, cp.async commit/wait and WGMMA/MMA instructions remain
unchanged; all original assembler flags are retained. The original source
file and report are copied without modification. These are explicit PTX
experiments, not an upstream source patch, native baseline replacement or
performance measurement.

Return all top-level text/JSON files from the new Reports directory, including
`cache-policy-summary.json` and every `*-artifacts.json`. Original binary
controls are expected to fail, so the overall exit can remain nonzero when
the interventions pass. Source/input/artifact hashes and the exact PTX
interventions are recorded. Local actual-H800-PTX transformation checks and
mocked worker/reassembly/eight-job orchestration passed; the hardware A/B
result is pending.
