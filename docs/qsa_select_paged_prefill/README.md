# QSA paged prefill selection for NVIDIA SM90

Writes logical compressed-group indices; it does not expand them to original
token positions or execute sparse attention. The API follows vLLM commit
`7a877ae34dcf4a7d46de08eb16db4699e1e0047c`,
`vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`.

## Contract

| Item | Contract |
| --- | --- |
| Backend | NVIDIA SM90; BF16 or FP8 E4M3FN, matching Q/K dtype |
| Q / cache | `[tokens,H,D]` / `[pages,page_size,1,D]`; H,D positive |
| Metadata | Packed request boundaries, supplied per-row visible compressed-group counts |
| Output | Caller-allocated contiguous INT32 `[tokens,token_topk//compress_ratio]` |
| Selection | Unscaled sum of per-head ReLU dot products; FP32 accumulation/logits |
| Ordering | Unspecified output order and tie-breaking; short rows padded with -1 |
| TopK | Existing persistent_topk, k=512/1024/2048; FlagTree TLE required |
| Preconditions | Finite Q/K and valid device-resident metadata values; inference only |
| Unsupported | Other backends/architectures/dtypes raise; no Torch compute fallback |

## Implementation and exemptions

Rectangular tiles reuse paged K across query rows. Compact request-local tiles
handle eligible ragged batches. BF16 H4/D128 can tune head grouping. Aligned
FP8 logits use vector bulk stores and scalar tails; eligible R32 split-query
variants issue paired dots before the epilogue. Every scoring path uses YAML
configs and libtuner. Fixed select-all and byte-zero-fill kernels have no tuning
search: they only enumerate indices or initialize radix state.

Production Torch use is limited to metadata/device queries, uninitialized
allocation and no-copy views. Persistent TopK radix-state initialization uses a
Triton zero-fill instead of Tensor assignment. No TopK algorithm is changed
by this PR. The latest upstream sampled TopK implementation is retained.

## Test and benchmark scope

The PR retains 24 production test functions, including upstream BF16/FP8
fixtures, generic head/dimension cases, ties, chunk reuse, empty/select-all,
invalid contracts, current-device handling, score-cache keys and FP8 store
canaries. Retained function ASTs match the development source. Historical
candidate A/B tests remain on the research branch; see preparation.json.

The benchmark retains exactly ten FP8 cases (eight recorded envelopes and two
dtype variants). Tensor values, visibility and page mappings are synthetic;
these are not complete inference tensor traces. BF16 remains supported and
tested but is excluded from the active performance matrix by user decision.

Run on H800/H100 in a compatible FlagTree/TLE + vLLM environment:

```bash
PYTHONPATH="$PWD/src:$PWD" python -m pytest -q tests/test_qsa_select_paged_prefill.py --quick
PYTHONPATH="$PWD/src:$PWD" python -m pytest -s benchmark/test_qsa_select_paged_prefill.py --mode kernel --iter 2 --warmup 1 -k 'full_vs_vllm or scores_vs_vllm or topk_vs_vllm'
```

A SHA256-pinned native vLLM baseline is required. Set QSA_VLLM_REFERENCE_FILE
to the matching qsa_indexer.py and QSA_REQUIRE_VLLM_REFERENCE=1; absent baseline
must not be counted as validation. The isolated stock-runtime native adapter
is accuracy-only and cannot be used for performance comparisons.

## Historical H800 evidence (not PR-base acceptance)

Source commit: `16bc82f5bbc9e1033d71b5136243b8e80ad6e0c1`.
Report commit: `5ba1e500e392a672a7f914667965d340312123da`.
The iteration correctness profile passed 49 tests; this was not the full suite.
Both score implementations used the same FlagTree/PTXAS 13.4 toolchain;
native TopK was the vLLM CUDA extension and production TopK was TLE.
All ratios below are native/production; pipeline was measured directly.

| FP8 case | Score | TopK | Pipeline |
| --- | --- | --- | --- |
| first_chunk | 1.1458x | 1.0179x | 1.0500x |
| budget_exact | 1.1231x | 0.9765x | 1.0309x |
| dominant_long | 1.2333x | 0.9606x | 1.0511x |
| tail112 | 1.3081x | 0.9635x | 1.0748x |
| tail113 | 1.3117x | 0.9629x | 1.0764x |
| two_balanced | 1.3360x | 0.9222x | 1.0622x |
| two_skewed | 1.3196x | 0.9638x | 1.0785x |
| short_tail | 1.3248x | 0.8958x | 1.0748x |
| compact | 0.9960x | 1.1217x | 1.0731x |
| long100k | 1.2458x | 0.7842x | 0.9838x |
| Arithmetic mean | 1.2344x | 0.9569x | 1.0556x |

NV performance acceptance is score-only kernel speedup, excluding TopK.
TopK and measured pipeline data remain mandatory reporting fields.
The PR preserves the exact score-operator source SHA256 but rebases onto a
newer upstream persistent_topk. Consequently these historical measurements
must not be relabeled as validation of the prepared PR branch.

## Current PR validation (2026-10-28)

Report commit: `5a9978ddd6d8774ebff511c90c449092002171b9`.
Validated PR production/test/config version: `a4f8ab9`.
Runner commit: `b69df69`. Independent audit verified all 27 pinned PR/framework
files and all 73 source snapshots. The later documentation-only update does
not change tested production, test, benchmark or tuning bytes.

The entire retained QSA test file passed: **140 passed, 1 skipped**. The sole
skip is the optional second-SM90 device-guard case with one visible GPU. This
is the targeted QSA suite, not the entire repository test suite.

All ten FP8 stage benchmarks passed. Equal-weight arithmetic means:
**score 1.2325x; TopK 0.9572x; directly measured pipeline 1.0578x**.
Nine score cases exceed 1x; compact is 0.9945x. Long100k's pipeline is 0.9824x,
with score 1.2591x and TopK 0.7856x. TopK optimization remains outside scope.
See h800-pr-validation.json for individual times/ratios.

Both score paths used the same FlagTree/PTXAS13.4/CUDA compatibility profile.
All 90 GPU guards passed (30-37C); all captured production kernels had zero
spills. Native TopK remained CUDA and production TopK remained TLE.

Latest upstream package imports require FlagGems, absent from the original
vLLM virtualenv. The runner extracted digest-pinned FlagGems 5.0.2 into an
isolated report directory and loaded native vLLM before FlagGems-vllm. No
compiler, source schema or production kernel was patched. This import order
passed; it does not identify the exact conflicting schema from the earlier
aborted import. Runtime/source identities were checked after imports.

## Preparation status

- Production source and retained test bodies extracted without algorithm changes.
- Latest upstream exports, backend registrations and TopK additions preserved.
- New-base targeted H800 correctness and ten-case performance validation passed.
- Local Mac validation was static/CPU-only; GPU evidence came from the user run.
- The personal FlagGems submission skill remains incompatible with this repository:
  its checker expects src/flag_gems, conf/operators.yaml and ATen dispatch.
  That submission gate has not been marked passed or bypassed.
- No PR has been created. An updated description is in pr-description.md.

Benchmark naming follows `benchmark/test_<operator>.py`. The former `_perf`
suffix was removed after validation; contents and workload definitions are
byte-identical, so the recorded GPU measurements remain applicable. Historical
source snapshots retain their original filenames.
