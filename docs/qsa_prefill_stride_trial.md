# QSA prefill: contiguous stride validation trial

Trial 2 starts from retained overlap optimization commit
`f9d754887a55e068d1275738b9f25ec7822e68e5` (operator source unchanged from
`7739abad`). The prior H800 profile records three stride-check calls per
validation, each sorting all stride/size pairs. The core_1 instrumented
cumulative cost is 6.68 us per public entry; this is a profiler observation,
not an uninstrumented saving estimate. Shared libentry dispatch is a larger
remaining cost, but changing it would affect unrelated operators and needs
its own runtime contract and validation scope.

The following tables freeze this trial before production edits.

| Project field | Choice |
| --- | --- |
| Current op | qsa_select_paged_prefill, same upstream QSA contract as trial 1; no direct Torch API |
| Editable | One stride helper in production; differential tests; pinned benchmark-only old stride helper and validation baseline selection; this trial document |
| Read-only | Public entry, all other validator code, overlap helper, GPU kernels, score launcher, Top-K implementation, workspace, configs, numerical references and tolerances |
| Hypothesis | Nonempty contiguous tensors are already internally nonoverlapping; bypassing Python sorting preserves behavior and reduces host validation cost |
| Build | Python-only, syntax/lint; Mac has no Torch/GPU |
| Validation | Randomized metadata differential tests; real Torch stride boundary cases; full independent H800 suite before timings, no unexpected skips |
| Benchmark | Existing schema 2 same-process full-entry validation A/B; baseline is retained trial 1, not the pre-overlap original |
| Active set / aggregation | Existing 22 comprehensive cases; full wall us per case, lower is better; old/new ratios and paired-round ratios, no aggregate masking regressions |
| Timeout / profiler | Existing 900 s per worker; unprofiled host validation probe and untimed cProfile after measurements; no GPU profiler needed for a metadata-only change |
| Autotune | No new kernel/performance parameter; original six score candidates and keys unchanged, shared launchers/cache in A/B |
| Fallback | No Torch compute/copy/cast; metadata reads only |
| Acceptance | Keep only after correctness and measurable full-entry benefit without important regression; otherwise revert production helper change |
| Missing docs | workflow.md and AGENTS.md read; optimization.md/deep_opt.md absent, as recorded in trial 1 |

| Contract field | Preserved behavior |
| --- | --- |
| Device / dtype | NVIDIA SM90 BF16/FP8; existing checks and error order unchanged |
| Shape / layout | Q/cache unit inner stride; existing conservative nonoverlapping-stride rule for strided inputs; singleton dimensions and transposes retain their old behavior |
| Empty tensors | Always use original stride algorithm: Torch considers empty tensors contiguous even when old conservative checker rejects their strides |
| Output / alias | Same int32 output, contiguous; retained conservative byte-hull alias validation including padding holes |
| Numerics | Unchanged score/ReLU/head sum/Top-K/tie/padding semantics and finite-input assumption |
| Error / autograd | Same type/message/order, inference-only; no expanded support promise |
| Workspace / streams | Same per-call allocations and ownership; no cross-call metadata caching |
| Reference use | Torch only constructs test fixtures and validates results; production uses metadata reads and existing empty allocation only |

| Path | Change / verification |
| --- | --- |
| Nonempty contiguous Q/cache/table | Return immediately from the stride helper, before sorted/zip; no GPU work |
| Empty tensor | Run original checker even if is_contiguous is true; test overlapping/zero-stride empty shapes |
| Strided / overlapping / singleton | Original checker, same acceptance/error; randomized differential and actual Torch layouts |
| Empty public entry / select-all / scoring | All dispatch and kernels remain unchanged; full correctness and 22-case regression coverage |
| Baseline | Clone current validator/public function with only stride helper replaced by exact frozen old helper; verify pinned helper, validator, overlap helper and public source hashes; share actual GPU launchers, never mutate production globals |

Status: candidate implemented with local checks; not yet measured on H800. CPU microbench alone cannot
justify keep; large device-bound cases may show no end-to-end benefit. The
prior long-case instability remains a limitation, not an assumed speedup.

Local verification: 22 CPU test methods passed across the metadata differential,
benchmark timing/failure/worker-propagation and run/upload suites. The stride
comparison covers 20,000 generated layouts (including empty, singleton and
negative-stride metadata); the prior 20,000 byte-hull comparisons still pass.
AST comparison proves the stride helper is the only changed production
function, and the benchmark's frozen helper/source pins match `f9d7548`.
Top-K, libentry and tuning files match that commit byte for byte. Production
Torch calls remain metadata queries, the existing device context and empty
allocations; the trial adds only is_contiguous/numel metadata reads.

H800 validation adds eight actual Torch layout fixtures, including both an
accepted and a rejected empty-contiguous layout. The expected full gate is
133 passes and one second-device skip with one visible GPU (33 native-vLLM
cases excluded); unexpected skips/failures prevent timing. Actual card results
and a keep/revert decision are pending.

The runner below runs correctness, all 22 paired benchmark cases and automatic
report upload. `--validation-baseline overlap-retained` is required for this
trial: the default `original` is the historical baseline before both host
optimizations and would measure their combined difference. The selected
baseline kind, commit, operator hash, shared public/validator bytecode, overlap
helper and GPU callable identities are recorded in each case JSON. Full wall
remains the primary metric; Graph replay is a device-work regression check.

```bash
if bash <<'QSA_STRIDE_AB'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-stride.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
CUDA_VISIBLE_DEVICES=0 QSA_TRITON_RUNTIME_PATH=/opt/flagtree \
  bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh" \
  --validation-baseline overlap-retained
QSA_STRIDE_AB
then
  echo "运行和报告回传均完成。"
else
  echo "运行或回传失败，当前会话已保留，请检查上方输出。" >&2
fi
```

The baseline comparison keeps trial 1's overlap optimization on both sides.
No native-vLLM performance comparison or cross-compiler speedup is collected.
