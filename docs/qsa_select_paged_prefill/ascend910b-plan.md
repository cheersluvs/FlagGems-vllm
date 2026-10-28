# QSA prefill: Ascend 910B bring-up

The NV PR preparation is frozen on `codex/qsa-prefill-nv-pr-20261028`.
This branch adds environment collection and a compile probe only. It does not
change production dispatch or claim that the operator supports Ascend yet.

## Project truth

| Field | Frozen decision |
| --- | --- |
| API | Keep qsa_select_paged_prefill compressed-group output; no token-index expansion |
| Backend | Ascend 910B, initially BF16; driver/CANN/Python/FlagTree unknown |
| Editable | Two diagnostic scripts and this bring-up plan |
| Read-only | NV operator, NV autotune YAML, production tests/benchmark/reference |
| Build | No install/build in environment collection; optional JIT tile probe |
| Validation | Import/device report, then one BF16 scoring tile vs FP32 CPU reference |
| Benchmark | None at this stage; no performance acceptance claims |
| Active set | Probe Q [4,128], K [64,128], output FP32 [64] |
| Timeout | 15s npu-smi; default 60s native worker; configurable for first JIT |
| Profiler | Unknown until device/toolchain report |
| Autotune | Probe fixed one tile, not production; later backend configs separate from NV |
| Fallback | No Torch compute in future production; Torch reference only in probe |

## Contract and backend choices

| Item | Decision |
| --- | --- |
| Inputs | First establish BF16 Q/K and INT32 visibility/index metadata |
| Output | Compressed-group indices, same meaning as NV; expansion is a separate op |
| Score | sum_h ReLU(Q_h dot K_j), unscaled; FP32 accumulation/logits |
| TopK | Examine existing Ascend persistent_topk before adding any new algorithm |
| Hardware | Do not assume the A3-only E3 op or LightningIndexer supports 910B |
| FP8 | Not part of initial bring-up; BF16 conversion must not become a hidden production fallback |
| Autograd | Inference only |
| Ties | Set-valued TopK; output ordering and tie-breaking unspecified |

## Paths to investigate after environment report

| Path | Action | Gate |
| --- | --- | --- |
| Environment | Detect torch_npu, CANN, Triton/FlagTree paths and actual NPU name | Must run on user's 910B |
| Dot compile | One BF16 dot/ReLU/head-sum tile | --probe; no paging or TopK |
| Existing score | Inspect _ascend/ops/qsa_mqa.py qwen4_qsa_mqa_paged_dot | BF16 H4/D128; currently scales by 1/sqrt(D), derives visibility from positions |
| Existing TopK | Inspect _ascend/ops/persistent_topk.py | radix route plus optional TLE segment-sort routes; verify on 910B |
| Production integration | Add vendor override with original API | Only after correctness of metadata, visibility, padding and TopK established |
| Performance | Profile score, probe tiling, integrate proven source-level changes | No compiler patches; preserve NV backend |

The existing Ascend score uses per-head FP32 multiply/reduce rather than the
NV rectangular dot layout. Reusing it requires reconciling metadata and scale;
it cannot simply replace our API. A different positive scale preserves exact
mathematical ordering but does not prove identical floating-point tie behavior.

## Environment report

Run from the checkout using whichever Python is currently installed:

```bash
python3 tools/check_qsa_prefill_ascend910b.py --report /tmp/qsa-910b-env-NEW
```

Exit 2 means prerequisites are missing; summary.json and worker logs are still
written. Do not enable interactive `set -e`. The script installs nothing and
never stops another user's process. A driver-installed host/container may need
its existing CANN set_env.sh sourced first; the collector records the active
environment rather than guessing an installation version.

After reviewing imports/device status, run a new report directory:

```bash
python3 tools/check_qsa_prefill_ascend910b.py --probe --device 0 --timeout 180 --report /tmp/qsa-910b-probe-NEW
```

The optional probe allocates only small tensors and reports actual values/error.
A passing result establishes a basic BF16 JIT route, not production correctness
or LightningIndexer/E3 compatibility. Review free memory before selecting a
logical NPU on a shared machine.

## Fetch, inspect and return the first report

On the 910B host/container, use the existing system Python for the initial
inventory. This command uses a fresh checkout and uploads reports even when
prerequisites are missing. GitHub authentication must already work; upload
failure retains the report and result worktree for retry without rerunning.

```bash
if bash <<'QSA_RUN'
set -euo pipefail
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-910b-env.XXXXXXXX)
git clone --depth 1 --single-branch --branch codex/qsa-prefill-ascend910b-env-20261028 https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" status --short
bash "$qsa_repo/tools/run_qsa_prefill_ascend910b_env.sh"
QSA_RUN
then
    echo '环境检测和报告回传完成。'
else
    qsa_status=$?
    echo "退出码：$qsa_status；2表示环境未就绪。报告路径在上方，当前容器会话保留。"
fi
```

Use `QSA_UPLOAD_REPORT=0` for a local-only report. Do not add `--probe` until
the inventory has established which logical device and toolchain to use.
