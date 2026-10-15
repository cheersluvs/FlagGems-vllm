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

## Report 4ef5279: original decode fault reproduced

Code `28bcd718b025820a610ed1b1628223c10e28746a` completed the default eight
H4/D128/query-length-3 inputs on H800. All 186 uploaded files and all 56
compiler-artifact bundles were checksum-verified.

| Check | BF16, tiles 1/2/4/8 | FP8, tiles 1/2/4/8 |
| --- | --- | --- |
| Original, stock Triton 3.7.1 | 4 score checks passed | 4 score checks passed |
| Original, FlagTree 0.6.1 / Triton 3.6.0 | 4 illegal-instruction failures | 4 illegal-instruction failures |
| FlagTree, loop STAGES=1 | 4 score checks passed | 4 score checks passed |
| Four binary replay checks per input | 16 launcher IndexErrors | 16 launcher IndexErrors |

The 32 replay failures are a diagnostic-script defect rather than GPU kernel
results. Decode has a two-dimensional JIT grid. `JITFunction` pads its grid,
but the raw `CompiledKernel` runner reads all three grid dimensions directly;
our replay path omitted the third dimension and raised `IndexError` before
launching. The corrected path retains the original grid for ABI comparison
and pads only the raw replay launch to `(grid_x, grid_y, 1)`. The original
JIT executions and saved binaries are unaffected.

All eight original FlagTree PTX files contain 32 async L2 cache-hint copies
and two full-fraction evict_first policy definitions. Their SASS consumes
UR0/UR1 in LDGSTS operands without any initializing writes. Stock PTX
retains async copies but has no fractional async cache-policy definitions;
Stock and no-hint SASS have no UR0/UR1 references. There is no WGMMA in these
decode kernels. Shared memory is 21504 bytes for BF16 and 13312 for FP8.
This matches the prefill fault's static mechanism, but the decode-specific
no-hint GPU A/B has not executed yet. Every identical-PTX stock reassembly
produced the original FlagTree CUBIN byte for byte; the saved no-hint CUBIN
was also identical in both replay-runtime records.

Source: [original decode report](https://github.com/cheersluvs/FlagGems-vllm/blob/4ef5279f59d390bb072c5a1bc75d131f1d7e1af2/results/qsa_select_paged_decode/h800/20261014-093323/decode-diagnostic-summary.json)
and its compiler-artifact bundles. Original failure reproduction is confirmed
for these eight inputs; a controlled decode cache-hint conclusion remains
pending the corrected replay run.

### Complete the saved binary controls

With the corrected scripts in the checkout, reuse the raw card report:

```bash
QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  "$PYTHON" tools/diagnose_qsa_native_decode_h800.py \
  --replay-report /tmp/qsa-decode-h800.jeoxayk0
```

This schedules exactly 32 fresh-process binary replays, using the saved
inputs, original FlagTree binaries, identical-PTX stock reassemblies and
producer-assembled no-hint binaries. It generates no new inputs, JIT kernels
or assembled binaries. Original source/input/artifact hashes are checked;
the saved no-hint PTX must equal the exact one-variable transformation of
original PTX, with unchanged metadata, signature, constants and grid.
The original report is copied without modification. Earlier original and
STAGES=1 results are retained and labeled `reused_from`, rather than counted
as new GPU runs. New reports include all saved producer bundles and
`previous-summary.json`, so an uploader copying top-level text/JSON preserves
the evidence. A full diagnostic run also uses the corrected grid padding.

## Report 441aba3: decode cache-policy trigger confirmed

Code `76a810fffa8ef72c4700252a38e0d4f15791c10d` completed all 32 corrected
binary replays. All 174 uploaded files and 56 artifact bundles were
checksum-verified. The input manifest and copied previous summary are
byte-identical to report 4ef5279. Earlier original/loop-STAGES=1 results are
explicitly reused; they are not new runs in this report.

For each of the eight BF16/FP8, H=4, D=128, decode_query_len=3,
TILES_PER_PROG=1/2/4/8 inputs:

| New binary replay | Runtime | Result |
| --- | --- | --- |
| Original FlagTree CUBIN | Stock Triton | Illegal instruction, 8/8 |
| Identical original FlagTree PTX reassembled by stock ptxas | Stock Triton | Illegal instruction, 8/8 |
| Remove only async-copy L2 cache-hint qualifier and policy operand | FlagTree | Score validation passed, 8/8 |
| Same no-hint CUBIN | Stock Triton | Score validation passed, 8/8 |

All raw replay grids have the expected third dimension of 1, all binaries
load, and no new replay log contains the earlier IndexError. The 16 failures
are intentional original-binary controls that reach the GPU; the other 16
runs reach and pass score validation, including the invisible-logit sentinel
checks. The overall nonzero exit correctly retains the reproduced faults.

Verified saved artifacts establish that each no-hint PTX differs only at
the 32 async-copy qualifier/operand sites. Original metadata, signature,
constants and grid remain identical, including loop STAGES=2. Both runtimes
execute the same patched CUBIN bytes. Identical-PTX stock reassembly produces
the original failing FlagTree CUBIN byte for byte. Original SASS has 32
LDGSTS uses of UR0/UR1 without initializing writes; patched SASS has no such
UR0/UR1 references. Async copies and MMA remain present.

This completes the decode-specific causal A/B for these inputs: the
FlagTree 0.6.1 / Triton 3.6.0 lowering emits fractional L2 policies on async
copies, and ptxas 12.8.93 drops policy initialization while retaining
descriptor-dependent LDGSTS instructions. Removing only the affected async
cache-policy use resolves the illegal instruction under both runtimes,
matching the prefill failure mechanism. This is a runtime binary fault,
not a compile-time rejection or a Top-K failure.

Source: [corrected decode replay summary](https://github.com/cheersluvs/FlagGems-vllm/blob/441aba3acec450f5a2f27d5e0edcd1bcaa0f4f2e/results/qsa_select_paged_decode/h800/20261015-002706/decode-diagnostic-summary.json)
and the saved producer/consumer artifact bundles in the same report.
The PTX intervention remains diagnostic only. No compiler installation,
production operator, original vLLM source or Top-K implementation has been
modified, and no timings were collected. Full native-decode compatibility
and same-compiler performance acceptance still require a compiler fix and
the corresponding full-operator checks.
