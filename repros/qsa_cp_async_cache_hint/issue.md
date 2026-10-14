# [NVIDIA SM90] cp.async fractional L2 cache policy triggers illegal instructions with ptxas 12.8.93

Draft only; not submitted. Both the standalone reduced source and captured-PTX
replay were confirmed on H800 by the user on 2026-10-14, using reproducer commit
`763904a758e01696933d35ca15651e5bf1d070c8`. See [validation.md](validation.md)
for the transcribed results and their provenance.

## Problem

On NVIDIA H800, a BF16 native vLLM QSA prefill scoring kernel compiled by our
FlagTree runtime launches successfully but fails at CUDA synchronization with
`CUDA error: an illegal instruction was encountered`. The same source compiled
with stock Triton passes. This is a runtime failure, not a compilation error.

We isolated the failure to the generated `cp.async` fractional L2 cache-policy
path. With exactly the same FlagTree-generated PTX, removing only
`.L2::cache_hint` and the final policy operand from four `cp.async` instructions
makes the kernel pass. Paging inputs, launch parameters, software pipeline,
`cp.async.commit_group` / `wait_group`, and WGMMA instructions are unchanged.

The generated SASS references a cache descriptor in `UR1` without any
initialization of `UR0` / `UR1` / `UR2` in the function. This strongly indicates
a ptxas miscompilation exposed by FlagTree's emitted PTX. We are requesting a
FlagTree backend workaround/regression test rather than claiming a host
launcher or TopK bug. We have not captured the actual faulting PC with
compute-sanitizer, so the precise trapping instruction remains unconfirmed.

## Environment

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA H800 80 GB, compute capability 9.0 |
| OS | Ubuntu 24.04.1 LTS container |
| Host NVIDIA driver | 570.124.06 (`nvidia-smi` reports CUDA 12.8) |
| Runtime setup | CUDA 12.9 compatibility libraries in `LD_LIBRARY_PATH` |
| PyTorch | 2.13.0+cu129 |
| FlagTree distribution label | 0.6.1 |
| Actual imported FlagTree Triton | 3.6.0, `/opt/flagtree/triton` |
| Stock imported Triton | 3.7.1 |
| ptxas, both runtime bundles | CUDA 12.8, V12.8.93 |
| Generated target | PTX 8.7, `sm_90a` |
| Original integration | vLLM 0.31.0+cu129; standalone scripts do not import it |

The FlagTree tree is a local runtime overlay; a clean upstream Git revision
was not recorded, and its driver contains a local scratch-zeroing patch.
The installed Triton distribution metadata says 3.7.1 even while the actual
import resolves to FlagTree's 3.6.0. Please use imported paths and these hashes
for provenance, rather than assuming an unmodified FlagTree 0.6.1 installation:

```text
triton/backends/nvidia/compiler.py
7357b379581baf59b088e939853c57e75bffc6b158d3059eb217be972d59ee68
triton/backends/nvidia/driver.py
45868d30c38de18f154cffe807374fcd0fcac6c149c2ff9ace6dc9a62b59091d
triton/knobs.py
233f2698dcaf166056b8794df1caf55a0a438025f086eeb6ec1c064038b86793
triton/_C/libtriton.so
0435ea3da94763e73816bd48d9bf3565e258ba709eb0167a382b067ec862357e
```

## Reproduction artifacts

The self-contained directory is on branch
[`codex/qsa-cp-async-issue-repro-h800`](https://github.com/cheersluvs/FlagGems-vllm/tree/codex/qsa-cp-async-issue-repro-h800/repros/qsa_cp_async_cache_hint).
It contains `minimal.py`, `replay.py`, `captured.ptx`, `captured.json`, and a
six-job runner. Only PyTorch, the selected Triton runtime, and its ptxas are
required. No vLLM/FlagGems imports or TopK execution are involved.

```bash
git clone --single-branch --branch codex/qsa-cp-async-issue-repro-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git qsa-cp-async-repro
cd qsa-cp-async-repro
# Activate your compatible PyTorch/FlagTree environment first.
# /opt/flagtree must contain the intended triton package.
# Primary minimal reproduction (expected: illegal instruction at synchronize):
PYTHONPATH=/opt/flagtree python repros/qsa_cp_async_cache_hint/minimal.py
# Run all six reproduction/control jobs:
PYTHONPATH=/opt/flagtree python repros/qsa_cp_async_cache_hint/run.py
```

Each job gets a new process/CUDA context and private compilation cache. The
runner records raw return codes, compiler artifacts, and failure phase;
expected failing cases cause a nonzero overall exit code. Please inspect
`summary.json` and each job's `run.log` / `result.json`.

The standalone H800 run confirmed: `original` and `fraction-half` fail at
synchronization with an illegal instruction; `no-hint` exits successfully.
The original replay reproduces the captured CUBIN SHA256 exactly.
Replaying captured PTX bypasses source compilation, even under stock Triton.

### Confirmed minimal source reproduction

`minimal.py` provides allocation, numerical reference, and artifact collection
around this reduced kernel:

```python
@triton.jit
def async_dot(Q, K, Out, STAGES: tl.constexpr, POLICY: tl.constexpr):
    n = tl.arange(0, 64)
    d = tl.arange(0, 32)
    query = tl.load(Q + n[None, :] * 32 + d[:, None])
    for tile in tl.range(0, 16, num_stages=STAGES):
        rows = tile * 64 + tl.arange(0, 64)
        keys = tl.load(K + rows[:, None] * 32 + d[None, :], eviction_policy=POLICY)
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        tl.store(Out + rows[:, None] * 64 + n[None, :], scores)
```

Q is BF16 `[64,32]`, K is BF16 `[1024,32]`, Out is FP32 `[1024,64]`;
grid `(1,)`, `num_warps=4`, `STAGES=2`, `POLICY="evict_first"`.
The H800 run confirms that this source compiles and launches, emits four
`cp.async` cache-hint sites, and fails at `torch.cuda.synchronize()` with
`CUDA error: an illegal instruction was encountered`. Both controls
(`STAGES=1` or `POLICY=""`) exit successfully, after numerical validation.
The original native source was tested under stock Triton; this newly reduced
source has not yet been tested under stock Triton. Its own SASS/faulting PC
has not been inspected, so descriptor-register evidence below refers to the
captured native kernel, not this reduced kernel.

## Verified controlled experiments

Every row below was run in a fresh process, for both H=1/D=32 and H=4/D=128.
H1/D32 has 10 packed query rows (3+7), context lengths 1025/1537, page size 16,
and query-start offset 19. Original loop constants were TILE_R=64, BLOCK_N=64,
K_TILES=16, STAGES=2, `num_warps=4` (backend default `num_stages=3`).

| Experiment | Result |
| --- | --- |
| Stock source compilation / native score | Pass |
| FlagTree source compilation / native score, no FlagGems import | Illegal instruction |
| Stock-generated CUBIN loaded through FlagTree | Pass |
| FlagTree-generated CUBIN loaded through stock Triton | Illegal instruction |
| Identical FlagTree PTX assembled with stock-bundled ptxas | Byte-identical failing CUBIN |
| FlagTree source, loop `STAGES=1` | Pass; async copy disappears, WGMMA remains |
| Only backend `num_stages=1`, loop still STAGES=2 | Illegal instruction |
| Captured FlagTree PTX, remove only cp.async L2 hint + policy operands, stock loader | Pass |
| Same no-hint PTX, FlagTree loader | Pass |
| Captured FlagTree PTX, only fraction 1.0 → 0.5 | Illegal instruction |

The last four experiments per shape form eight targeted jobs: four passes and
four expected failures. The no-hint variant preserves the async pipeline and
WGMMA, avoiding the confounding change caused by disabling pipelining.
No production source, TopK implementation, or compiler installation was changed.
A full integration suite after a repaired compiler has not been run.

Evidence is preserved in report commits
[`a4685e8eaae5bc8b3e100ca4c3d5977e8b3a792e`](https://github.com/cheersluvs/FlagGems-vllm/commit/a4685e8eaae5bc8b3e100ca4c3d5977e8b3a792e)
and
[`957a8f437e99f29816044953d19dc2a338f41e99`](https://github.com/cheersluvs/FlagGems-vllm/commit/957a8f437e99f29816044953d19dc2a338f41e99).
Original native source was pinned to vLLM commit
[`7a877ae34dcf4a7d46de08eb16db4699e1e0047c`](https://github.com/vllm-project/vllm/blob/7a877ae34dcf4a7d46de08eb16db4699e1e0047c/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py).

## PTX / SASS evidence

Captured H1/D32 PTX contains two policy definitions and four hinted async copies,
including:

```ptx
createpolicy.fractional.L2::evict_first.b64 %rd54, 1.0;
cp.async.cg.shared.global.L2::cache_hint [ %r87 + 0 ], [ %rd53 + 0 ], 0x10, %r88, %rd54;
```

Its disassembly includes:

```text
LDGSTS.E.BYPASS.128 [R12+UR0], desc[UR1][R26.64], P0;
LDGSTS.E.BYPASS.128 [R12+UR0+0x800], desc[UR1][R24.64], P1;
```

No instruction in that function initializes UR0/UR1/UR2. The fraction-0.5
variant retains the same problematic descriptor use. The passing no-hint
variant has no `desc[UR1]` read. Captured artifact hashes:

```text
PTX SHA256
b865d6c3114eb7f8d9dcd1b2e8253c7fcbb1e0c835ca3c18952d4877fce1f819
Original CUBIN SHA256
dd097a244194ef3a4e13a7e6bf0a271be4421fd3ecd1d2a9188fd4c2ff5aa9b7
```

## Related work / requested action

Upstream Triton [PR #11660](https://github.com/triton-lang/triton/pull/11660)
removed fractional cache policy from `cp.async` lowering as a ptxas workaround.
[PR #11848](https://github.com/triton-lang/triton/pull/11848) later gated its
restoration on newer PTX. Those reports concern different assembler/GPU
versions; they do not establish a fixed ptxas version for this H800 case.

A related [NVIDIA compiler report](https://forums.developer.nvidia.com/t/nvcc-13-1-ptxas-codegen-bug-createpolicy-fractional-dropped-on-sm-90/368663)
describes dropped fractional-policy initialization on H20 with CUDA 13.1.
Our observed H800 / ptxas 12.8.93 case is a separate tested configuration.

Could FlagTree adopt/validate an equivalent workaround for affected NVIDIA
toolchains and add a regression that actually launches and validates the
generated kernel? Removing the policy hint while retaining async copies and
pipeline scheduling is the diagnostic change verified here. A newer ptxas may
also help, but we have not verified a safe version boundary on this hardware.
