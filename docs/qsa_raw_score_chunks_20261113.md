# Raw two-Cube scoring on representative INT8 chunks

## Frozen project scope

| Field | Contract |
| --- | --- |
| Editable | Isolated raw probe, its new score/audit helper, CPU checks and this note |
| Read-only | Production and integrated INT8 kernels, harness, CANN/OPP, original FlagTree |
| Compiler | Previously verified independent final-task-mode candidate; same copy for both arms |
| Build | Reuse unchanged ordinary AIC bitcode and address ABI; no private buffers in the enclosing raw wrapper |
| Active set | First complete chunk from compact (128 rows/W1088), budget_exact (128/W8192), long100k (112/W29696); existing seeded tensors, shuffled pages, page192, H4/D128 |
| Benchmark | Two interleaved rounds, 3 warmups/10 active samples, strict CPU/NPU trace task counts; device-task sums, no host gaps |
| Acceptance | Exploratory chunk score comparison only; not the formal harness mean, no CANN or TopK speedup claim |
| Autotune | Exempt API feasibility candidate; unchanged production parameters |
| Timeout | Whole worker bounded by 1800 seconds; all diagnostics uploaded together |

## Numerical contract

| Field | Contract |
| --- | --- |
| Inputs | Existing quantized INT8 Q/K, positive FP16 query/key scales, FP16 per-head weights, shared row table within chosen request chunk |
| Current | Existing PreparedSharedMapInt8Selection.score_into with integrated defaults, key_group4/single_page64 enabled; log actual route and source hashes |
| Candidate | Pack four queries (16 heads); first Cube INT8 -> INT32; Fixpipe half(ReLU(dot/1024)); second Cube with block-diagonal half(1024*qs*weights); apply per-key ks and visibility afterward |
| Precision difference | Half intermediate and half coefficient differ from current FP32 arithmetic; compare all scores/masks to current at rtol=atol=1e-3, independently check sampled CPU dot/FP32/half-chain oracles; reject on mismatch |
| Layout | Candidate Q/K NZ and W packing are real device kernels; charge their full device time every chunk, even though K could potentially be reused |
| Storage | Raw output retains 16x64 per four-query tile; extra scratch and output finalization charged; no unsupported claim that this is an optimal final implementation |
| Metadata | CPU slice/shared-table/visibility proof outside device timing, host preparation time reported; no model-forward or dynamic-request validation claim |
| Unsupported | Other H/D/page sizes, cross-request chunks, nonpositive scales, nonfinite/overflowing coefficients; no fallback or relaxed tolerance |

## Implementation paths

| Arm | Work and interpretation |
| --- | --- |
| current | One actual integrated score launch, same original device tensors |
| pack_qw / pack_k | Independent device packing stages |
| raw_cube / finish | External native tile loop and per-key scale/visibility/dense output stages |
| prepared | raw_cube + finish; excludes explicitly measured packing |
| complete | pack_qw + pack_k + raw_cube + finish; primary candidate comparison |
| audit | Export final binaries/ELF identity and try installed CCE/LLVM disassemblers; absence/unsupported ISA is pending evidence, not a claimed decoded Fixpipe |

Compile, audit, correctness and timing run in one invocation. A failed chunk is
reported and later cases still run; invalid candidates receive no timing.
The original micro checks remain a prerequisite. Native instruction export does
not itself prove traffic volume or Cube/Vector overlap. Production is unchanged.

The current INT8 entry here is the existing integrated experimental adapter,
not the separately registered BF16 operator. Its unchanged source hashes and
actual route are recorded. Device buffers are allocated before timing, as for
the current arm; packing is executed again in every complete candidate window.
Fixture generation, metadata proof, allocation and transfer time are labelled
separately and do not constitute an end-to-end model latency measurement.

Mac validation: six CPU checks cover NZ address ordering, page transitions,
four-query head isolation/tails, scratch bounds, ELF/decode evidence, and both
generated raw entries. Python formatting/lint/syntax and Bash syntax are checked.
There is no Mac NPU result; card compilation and numerical gates remain pending.

Prerequisite evidence: report 07937ddc2b5bb8a8d39d98f487b2b4fef4201c50 passed
15 micro checks and normal Vector/Dot-Vector smokes on Ascend910B4-1. QK/chain
binaries were byte-identical to the preceding failed-metadata report; the
isolated compiler metadata correction enabled normal AIC execution. That result
does not establish numerical equivalence with the FP32 scoring contract below.

## First chunk report and correction

Report a44aa380c2dc238c6c265090f501f89c4bd48c23 compiled and ran all three
chunks, but rejected every candidate before timing. All masks and Q/K packing
checks passed; current matched the sampled FP32 oracle, while raw differed
even from its own half oracle. budget_exact contained 4470 nonfinite scores.
These are correctness failures, not evidence of a performance regression or
ordinary FP16 rounding. No speedup was recorded.

The newly added weight packer had omitted the transpose required by the native
LoadData2D call. The passing micro probe already packed W.T; the chunk packer
and its CPU packing check had incorrectly agreed on W. Both now use W.T.
New CPU tests exercise the actual generated index expressions against the
native transpose contract and fail against the previous source.

The returned BCMLIR also placed dynamic offsets in memref.reinterpret_cast,
then extracted only the underlying aligned pointer before the raw call. Per the
[MLIR operation contract](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_aligned_pointer_as_index-memrefextractalignedpointerasindexop),
this is not a proof that the effective address includes the view offset.
The wrapper now casts each unsliced base once, then adds explicit uint64 byte
offsets: Q2048, K8192, W512 and output4096. This is an operator-side expression
change; no additional compiler patch is introduced. A pre-execution IR gate
rejects aligned-pointer extraction from reinterpreted views. Device numerical
checks remain the final gate; the exact effect of the old native address code
has not been directly instrumented.

Raw output is initialized to NaN outside timing and must be entirely finite.
Score failures include coordinate/value samples. The retry retains all micro
correctness checks but omits their already-verified micro timings; the three
chunk comparisons still compile, validate and time in a single invocation.
The installed llvm-objdump returned only <not available> instruction lines;
final Fixpipe/ReLU decoding therefore remains unproven.
