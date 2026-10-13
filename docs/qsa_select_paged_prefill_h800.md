# QSA paged prefill: H800 implementation and validation

Status: H800 quick accuracy report records 38 passed and 9 unavailable-vLLM
skips. Full accuracy, official comparisons and performance remain pending.
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
  follows the available workflow. Deep optimization awaits measured evidence.

## Project truth table (before implementation)

| Field | Frozen choice |
| --- | --- |
| Current op | vLLM QSA paged prefill; public `flaggems_vllm.qsa_select_paged_prefill` |
| Editable files | New op, NVIDIA tune YAML, exports, new tests and benchmark, reference helper, this guide, card runner |
| Read-only | Existing QSA/Top-K operators, upstream reference, input/reference helpers after first validation; existing dirty FP8 gate files in the primary checkout |
| Build | Python-only change. Card: `python -m pip install --no-build-isolation --no-deps -e .`; development/validation uses `PYTHONPATH=$PWD/src` |
| Validation | `python -m pytest -q tests/test_qsa_select_paged_prefill.py --quick`, then full file; value-based Top-K and score comparison |
| Benchmark | `python -m pytest -s -q benchmark/test_qsa_select_paged_prefill_perf.py --level core --mode operator --warmup 100 --iter 500 --record json`; median latency in ms, smaller is better |
| Active set | BF16, H=4, D=128, page=16/64, requests=1/4, total query rows=1/64/512/2048, compressed width=2048/4096/16384/65536, block K=512/1024/2048; exact tuples in benchmark |
| Aggregation | Report per-shape latency and SpeedUp separately for Torch and pinned vLLM; no aggregation may hide a key-shape regression |
| Acceptance | All functional cases pass; each key shape has baseline/FlagGems latency >=0.9; no measured claims before H800 results |
| Timeout | First GPU test/JIT 20 min; full tests 30 min; benchmark 45 min (external timeout recommended) |
| Profiler | Card: `nsys profile --stats=true ...`; `ncu --set full ...` after narrowing to the score kernel. Unknown availability until the card run |
| Autotune | `qsa_select_paged_prefill_score` through `get_tuned_config` + `libtuner`; key includes chunk rows, query length, logits width, request count, page size, layouts; candidate TILE_R/BLOCK_N/K_TILES/warps/stages in YAML |
| Fallback | No Torch compute fallback. Reuse existing Triton `top_k_per_row_prefill`; unsupported types/devices/shapes raise |

## Contract table (before implementation)

| Field | Contract |
| --- | --- |
| Device | NVIDIA CUDA SM90 (H800/H100); first acceptance target H800 |
| Q/cache | BF16 Q `[T,4,128]`; BF16 compressed K `[physical_pages,page_size,1,128]`; forward only, no gradients |
| Layout | Unit innermost stride; positive, non-overlapping outer strides; Q/cache/page-table row padding and input slices supported; contiguous metadata and output |
| Metadata | Same device; page table int32/int64 `[B,P]`; `query_start_loc` int32/int64 `[B+1]`; visible counts int32 `[T]` |
| Metadata values | Caller supplies monotone offsets whose terminal-minus-first value is T; request lengths <= max_query_len; visible counts in `[0,min(capacity,ceil(max_seq_len/ratio))]`; active physical page IDs valid. These device-resident value preconditions avoid host synchronization |
| Output | Caller-owned int32 `[T,token_topk/ratio]`, contiguous; no input/output overlap; filled in place, returns None |
| Parameters | Positive integer compression ratio, divisible token_topk; actual block K in 512/1024/2048; positive max lengths for nonempty input |
| Score | FP32 accumulation; sum across heads of ReLU of each head's Q.K, without sqrt(D) scaling |
| Selection | Logical compressed indices, not physical pages or expanded tokens; valid count min(visible,K), remaining positions -1; order and choice among exact ties unspecified |
| Numerics | BF16 inputs with finite FP32 scores; NaN/Inf and overflow outside supported contract. Signed zero behaves as tied zero. Score comparison uses FP32 tolerance; exact index equality is not required for near ties |
| Empty | T=0 returns without allocation/launch after shape/type checks; zero visible rows emit -1 |
| Unsupported | FP8/FP16/FP32, H/D other than 4/128, gradients, non-SM90/non-NVIDIA, non-unit inner stride, noncontiguous metadata/output, overlapping output |
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
| Top-K | Per-row score selection | Existing `top_k_per_row_prefill` on `[0,visible)` | Existing operator's policy; no changes | Random/tied/short rows and K sweep | Existing TLE/non-TLE scratch and tie behavior |
| Workspace | Nontrivial selection | One per-call logits allocation reused across chunks; Triton zero row starts | Fill exempt; score tuned | Force 1 MiB budget and cross-request chunks | Only logits is bounded by configured budget; Top-K scratch is additional |
| Unsupported | Outside contract | Explicit exceptions, no conversions or compute fallback | None | Error tests | FP8 is deferred |

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

Modes `quick`, `full`, and `bench` run individual stages. Benchmark timing uses
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

The new named input builders reproduce the BF16 selection inputs from
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
  comparisons; FP8 and upstream D=8 inputs explicitly test unsupported errors.

The benchmark adds a separate upstream full-operator group with slack 0/1792
and default/one-row chunks, against both Torch and pinned vLLM. It preserves
the original synthetic tuples and reports the chunk modes under distinct
operator names. These are upstream regression fixtures, not evidence of
production workload frequency. No H800 results are available yet.

## Current verification and delivery limits

- Mac: Python compilation, exact upstream argument-name/order comparison,
  YAML config validation, no-Torch-compute host audit, shell syntax checks,
  and all configured pre-commit hooks passed.
- Upstream coverage follow-up: changed Python sources compile and configured
  style hooks pass. Executing the actual Python host function with mocked
  allocations/launchers confirms widths 1280/1536 and 1/36 chunks for the
  four slack/budget combinations, plus negative/invalid budget rejection.
  This exercises no GPU kernels and is not a GPU correctness result.
- H800: the user's quick report records supported-case accuracy passes and
  explicit unsupported/error checks; official comparisons were skipped.
  Full accuracy and performance acceptance are pending. No speedup claim is
  made; deep optimization awaits performance evidence.
- Autotune: six score candidates; fill kernels use fixed launches because
  they only initialize indices/row starts. Reused Top-K retains its existing
  tuning/dispatch policy; this change does not retune it.
- No external library production path; no Torch compute fallback. FP8,
  autograd, other head configurations, other devices, model memory profiling,
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
running the original pinned-reference runner. This setup has only been
checked statically on the Mac; actual container installation and execution
are pending. The downloaded compatibility package's SHA256, library paths
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

Official installation sources:
- https://github.com/vllm-project/vllm/releases/tag/v0.31.0
- https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/

After operator validation, the plugin's prefill path can call the exported
function with the existing ten arguments. Keep compressed-index expansion
and decode behavior in their existing caller. Confirm that this entry is
actually reached in a model run before reporting end-to-end acceleration.
