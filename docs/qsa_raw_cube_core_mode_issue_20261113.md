# Cube custom-op launch mode mismatch in the tested 910B environment

## Result

Both the INT8 QK control and complete two-Cube chain compile to native binaries
through the normal TLE custom-op entry when using a scalar-address ABI. The
tested public `mix_mode="aic"` option does not correct their launch metadata.
No target kernel was executed: the probe stopped at its consistency gate.
There is no numerical, timing, performance or full-QSA acceptance result.

This finding is limited to this installed environment and these expressions.
It does not prove the hardware dataflow unsupported or rule out every possible
operator expression. It does identify a concrete propagation/registration
inconsistency in the tested normal entry.

## Frozen evidence

- Report branch: `codex/qsa-ascend910b-results-20261113-020331-768673`.
- Report commit: `428ef49099c3138fc271968664cd0b2efa9e9533`.
- Probe code commit: `1423026711a2e7c302be693b6a7bf7a084346659`.
- Device: Ascend910B4-1; CANN 9.1.0; FlagTree-provided Triton 3.5.1;
  PyTorch 2.10.0+cpu with torch_npu.
- The report contains installed compiler.py, driver.py and npu_utils.cpp snapshots
  with their hashes, each compiler input/command/error stream, and all binaries.

Report files below are relative to
`results/qsa_select_paged_prefill/ascend910b/raw-cube-chain-20261113-020331/`
in the immutable report commit.

| Requested configuration | Entry | Native compile | Final mix_mode | Callback bs_task_type | Executed |
| --- | --- | --- | --- | --- | --- |
| Default | INT8 QK | Passed | aiv | 20 | No |
| Default | QK -> Fixpipe/ReLU/half L1 -> weighted Cube | Passed | aiv | 20 | No |
| mix_mode=aic | INT8 QK | Passed | aiv | 20 | No |
| mix_mode=aic | Complete chain | Passed | aiv | 20 | No |

All binary pairs are byte-identical across the two option settings:

| Entry | ELF bytes | .text bytes | SHA256 (both options) |
| --- | --- | --- | --- |
| QK | 3448 | 1400 | 578a481454b6e680929ed2545c2bf02018f1a3451916d06b16c9572d8ed26095 |
| Chain | 3672 | 1620 | ccd6752a9b40361bd6551f6593ae9e491373f62ce02f9b9b55b2c79ad6448ad0 |

ELF machine is 0x1029 and flags are 0x940000. These header observations do not
stand in for instruction disassembly or device execution.

## Minimal expression

The existing probe ships an operator-owned AscendC file, compiles ordinary AIC
bitcode with ccec, and binds it using:

```python
@dialect(name="cann", format="bitcode", file=bitcode,
         extern_func_name="qsa_raw_qk_address",
         pipeline={"core": "cube", "pipe": "PIPE_M"})
def qk_op(*args, **kwargs):
    ...

@triton.jit
def custom(Q, K, W, Out, SCALE_BITS: tl.constexpr):
    with al.scope(core_mode="cube"):
        q = tl.cast(Q, tl.uint64)
        k = tl.cast(K, tl.uint64)
        w = tl.cast(W, tl.uint64)
        out = tl.cast(Out, tl.uint64)
        bits = tl.full((), SCALE_BITS, tl.int64)
        raw.call(qk_op, [q, k, w, bits, out], output_indices=[])
```

The C interface takes four uint64_t addresses and an int64_t scale in that
order and writes the supplied destination. It owns its isolated TPipe buffers
and events. There is no dummy Dot, external Triton local allocation, modified
compiler source, altered IR or manual launch-metadata override.

## Source-level evidence from the installed files

1. `installed-compiler.py.txt:329`, `_parse_linalg_metadata`, unconditionally
   assigns `metadata["mix_mode"]` from the intermediate IR. The captured input
   contains an explicit CUBE scope/custom op but outer `mix_mode="aiv"`.
2. The final BiSheng callback supplies `bs_task_type=20`. The actual installed
   `installed-driver.py.txt:538` helper divides this by ten and maps task kind 2
   to `MSPROF_GE_TASK_TYPE_AI_CORE`, with ratio 0.
3. `installed-driver.py.txt:74` passes mix_mode to native binary loading.
   `installed-npu_utils.cpp.txt:50` selects RT_DEV_BINARY_MAGIC_ELF_AIVEC when
   that string is aiv; otherwise it selects RT_DEV_BINARY_MAGIC_ELF. The Python
   launcher also uses mix_mode to select physical core count at line 605.

The parser overwrite was reproduced on the Mac by extracting the pure Python
function from the actual installed snapshot, seeding metadata with aic, and
passing `core-explicit-aic-qk-compiler-1-input.mlir.txt`. It returned aiv. The
actual installed task-classification helper returned AI_CORE/0 for (20,aiv).
These checks required no NPU execution and did not modify the source snapshots.

## Boundary and recommended compiler work

The scalar-address representation got past the earlier custom memref memory
planning failures. Native compilation is established for both controls, but
normal loading/execution is still unverified because the metadata is incoherent.
Do not remove the guard and present a device or performance result.

A compiler-side resolution should keep final inferred core/task type and
loader mode consistent, while preserving valid explicit configurations. Possible
repair locations are initial global-mode inference for Cube custom ops and/or
final metadata propagation after BiSheng's task-type callback. The exact repair
must cover AIC, AIV, MIX_AIC and MIX_AIV, including core ratios, workspace and
runtime registration; replacing a string in the probe would not establish that.

After a supported resolution, rerun the same normal entry for numerical checks
and paired diagnostic timing in one invocation, then assess full-QSA integration
and the repository kernel-mode harness separately. This candidate is not in the
production operator. No further unrelated GPU option sweeps are proposed here.
