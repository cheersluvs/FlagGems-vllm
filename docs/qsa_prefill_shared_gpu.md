# H800 共卡正确性与探索性三阶段 benchmark

在已有 FlagTree CUDA13.4 入口上显式加入 `--shared-gpu-exploratory`。
正确性仍运行完整 `tests/test_qsa_select_paged_prefill.py`，包括原生 vLLM 直接比较；
通过后测上游12组 shape 的评分、TopK和顺序组合。默认生产 `persistent_topk` 保持不变。
两侧仍使用同一个 FlagTree、新 ptxas 和 CUDA13.4 兼容库，计时器、参数、shape及正确性容差不变。

## 共卡条件与数据含义

- 允许已有计算进程和超过256MiB的驻留显存；这两项豁免仅对显式共卡模式生效。
- 默认启动前要求至少5120MiB空闲显存，可用 `--shared-min-free-mib` 指定。
  这是准入下限，不保证全部测试能够装下；真实显存不足仍报OOM并回传，未调整shape或logits budget。
- 仍检查起始温度不高于75°C、利用率不高于5%，保留thermal slowdown及遥测有效性检查。
  温度/滞后利用率的冷却等待沿用现有参数；实际计时中发生热降频仍拒绝该组。
- 自动选卡时，从低温到高温寻找满足共卡准入条件的H800；显式完整UUID仍不会被替换。
  全套测试固定同一张卡。脚本不结束、暂停、重配已有vLLM进程。
- 每次计时遥测记录所有共卡PID及其显存占用。允许共卡不表示这些进程没有执行GPU任务；
  其他任务可能使延迟、比值、时钟和温度变化，采样也不能证明两次采样之间没有干扰。

该模式中，汇总和每组均带有 `measurement_context=shared_gpu_exploratory`、
`acceptance_eligible=false`；每阶段即使完成正确性和计时检查，其 `accepted` 仍为false，
另以 `measurement_validated=true` 表示数据已完成检查。
TSV状态为 `exploratory`、列名为 `exploratory_speedup`，不得当作独占环境性能验收结果。
正确性比较通过与否照常记录。没有显式共卡参数时，原独占检查保持原样。

## 容器运行命令

使用现有vLLM0.31.0虚拟环境和 `/opt/flagtree`，工具链仍自动隔离下载、校验和解包。
整个命令块在子shell中执行，失败保留交互会话；成功及失败均自动尝试回传已有报告。

```bash
if bash <<'QSA_SHARED_AB'
set -euo pipefail
trap 'qsa_rc=$?; printf "失败：退出码=%s，行=%s，命令=%s\n" "$qsa_rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_rc"' ERR
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
export QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python
export CUDA_VISIBLE_DEVICES=auto
test -x "$QSA_STOCK_PYTHON"
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-shared.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" rev-parse HEAD
bash "$qsa_repo/tools/run_qsa_prefill_flagtree_ab_h800.sh" \
  --flagtree /opt/flagtree --shared-gpu-exploratory \
  --shared-min-free-mib 5120 \
  --mode kernel --suite upstream-prefill \
  --rounds 7 --kernel-warmup-ms 1000 --kernel-rep-ms 100 \
  --timeout 1800 --accuracy-timeout 7200 \
  --cooldown-timeout 900 --cooldown-interval 10
QSA_SHARED_AB
then
  echo "共卡正确性、探索性测试和报告回传均完成。"
else
  echo "测试或回传未完成；当前会话已保留，请检查上方诊断。" >&2
fi
```

报告仍包含完整正确性JSON/TXT、各case JSON/TXT、编译器/实际兼容库/原生扩展身份、源代码快照及TSV。
结果分支沿用 `codex/qsa-h800-results-<UTC时间>`；共卡报告单独放在
`results/qsa_select_paged_prefill/h800/flagtree134-shared-<UTC时间>/`。
退出0仅表示功能检查和所选数据收集完成；1表示功能/编译/OOM/计时检查等失败，2表示准入环境不满足。
