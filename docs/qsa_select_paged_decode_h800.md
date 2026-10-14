# Native QSA decode cache-policy validation on H800

Run `tools/diagnose_qsa_native_decode_h800.py` in the existing official vLLM
0.31.0 environment. It checks whether native decode reproduces the prefill
illegal-instruction fault in FlagTree's asynchronous L2 cache-policy path.
It does not implement a FlagGems decode operator, run Top-K, collect timings,
or establish whole-operator compatibility.

The original source is pinned to vLLM commit
`7a877ae34dcf4a7d46de08eb16db4699e1e0047c`, file
`vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`, SHA256
`4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b`.
The script invokes its original `_qsa_mqa_paged_uniform_kernel`, retaining
the host function's BLOCK_N=64, STAGES=2 and BF16/FP8 num_warps=2/1 settings.

## Diagnostic inputs

Default coverage is BF16 and FP8 E4M3FN, H=4, D=128,
decode_query_len=3, page_size=16, and 8192 compressed cache columns.
The shapes below are designed diagnostic inputs, rather than copied upstream
test fixtures. Request counts select the four configurations through the
original `_decode_tiles_per_program` function; this is checked at preparation
and before every launch.

| Requests | Requests × ceil(columns / 64) | TILES_PER_PROG |
| --- | --- | --- |
| 3 | 384 | 1 |
| 128 | 16384 | 2 |
| 256 | 32768 | 4 |
| 1024 | 131072 | 8 |

Queries are contiguous and request-major. Shuffled page tables share valid
physical pages, modeling prefix-cache reuse without large serialized caches.
The first request sees all 8192 columns, ensuring that each program actually
executes the configured number of tiles. Other queries have different visible
lengths, including zero visibility within a request and a fully empty request.
The FP32 Torch reference gathers through the actual page tables, applies ReLU
per head and sums the heads, with TF32 disabled. BF16 uses rtol=2e-4,
atol=2e-3 on unscaled scores. FP8 uses rtol=atol=1e-3 on scores divided by
sqrt(D), consistent with the existing QSA FP8 value-check tolerance.
Invisible logits must retain their initial -inf sentinel.

## Checks per input

1. Original score under stock Triton.
2. Original score under FlagTree.
3. FlagTree with only loop STAGES changed to 1, explicitly labeled as an
   ablation rather than the original baseline.
4. Replay the original FlagTree CUBIN under stock Triton without recompiling.
5. Reassemble identical FlagTree PTX with stock ptxas and the original observed
   assembler options, then replay under stock Triton.
6. Remove only the async cp.async L2 cache-hint qualifier and policy operand
   from original FlagTree PTX; reassemble using the **original producer's
   ptxas and options**, then replay under FlagTree.
7. Replay that same patched CUBIN under stock Triton without recompiling.

Each check has a fresh subprocess and cache. A CUDA fault, compile failure
or 900-second subprocess timeout is recorded and does not stop later checks.
Binary-dependent checks are marked unavailable if the producer never exported
a binary. If the original PTX has no asynchronous L2 cache hints, the hint
intervention is marked not applicable instead of inventing a patch.

Before binary loading, the script exports TTIR, TTGIR, LLVM IR, PTX, CUBIN,
original metadata and replay ABI, plus SASS when disassembly is available.
Checksummed gzip/base64 bundles make these artifacts portable in top-level
`*-artifacts.json` files. Runtime records identify the imported compiler,
Torch, vLLM, ptxas and GPU; source and serialized input SHA256 values are checked
before execution. Original source, production code and compiler installations
are not modified.

`decode-diagnostic-summary.json` distinguishes successful original runs,
controlled cache-hint fault reproduction, and inconclusive results. A causal
reproduction requires the stock original to pass, the original FlagTree binary
to fail with an illegal instruction under both runtimes, and the same patched
binary to pass under both. Static uniform-register lines are preserved for
inspection; a regex does not claim to prove an uninitialized descriptor.
Passing selected shapes does not prove that every decode specialization works.

## Card commands

After fetching the code into a clean checkout containing both diagnostic
scripts and `tests/qsa_select_prefill_utils.py`:

```bash
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_native_decode_h800.py
```

This schedules up to 56 score probes plus preparation. To also check
single-query decode, pass `--query-lengths 1 3`. Heads, dimension, dtypes and
tile configurations can be selected, for example:

```bash
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_native_decode_h800.py \
  --heads 1 --head-dim 32 --dtypes bf16 --query-lengths 1 3
```

The script prints `Reports: /tmp/qsa-decode-h800.<suffix>`. Return all top-level
`*.txt` and `*.json`, including the summary and every `*-artifacts.json`.
Keep raw inputs and artifact directories on the card for further replay.
A nonzero exit is expected if a fault reproduces, even when its explicitly
labeled intervention passes; read the summary before interpreting the exit.

Mac validation checks syntax/style, input/reference construction, original
decode launch binding, one-variable interventions, process/environment
routing, failure continuation and portable summaries. These checks cannot
establish the GPU result; H800 execution is still required.
