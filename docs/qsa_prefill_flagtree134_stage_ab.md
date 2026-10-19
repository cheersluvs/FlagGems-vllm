# H800：同一 FlagTree / CUDA 13.4 工具链的原生 QSA 对照

入口：`tools/run_qsa_prefill_flagtree_ab_h800.sh`。先运行现有完整 prefill 功能测试，
包括原生 vLLM 直接比较，通过后测量上游12组 shape 的评分、TopK、顺序组合三个阶段。
本轮保留默认生产 `persistent_topk`，不修改生产算子、TopK、调优配置和 benchmark shape。
此前上游 Triton 加 `top_k_per_row_prefill` 变体仍使用原入口
`tools/run_qsa_prefill_triton_ab_h800.sh`，两轮结果分别保存。

## 已验证的工具链前提

[报告 c963992](https://github.com/cheersluvs/FlagGems-vllm/blob/c963992beabe137a2133b3ff932211b7de06cd16/results/qsa_select_paged_prefill/h800/ptxas-20261019-090923/ptxas-summary.json)
完成2组 prefill、8组 decode 的原生评分诊断：FlagTree 加旧 ptxas 全部复现非法指令；
原 PTX 字节不变，加 ptxas 13.4.59 重组装后全部通过；新 CUBIN 跨运行时回放、FlagTree
全新 JIT 也全部通过，且保留 fractional policy / async L2 hint。此结论仅覆盖评分诊断。

新入口复用该验证的下载、SHA256校验和隔离解包逻辑，自动准备：

- 用户指定的已有 FlagTree 目录，默认 `/opt/flagtree`；TLE 必须可用。
- NVIDIA ptxas **13.4.59**，通过 `TRITON_PTXAS_PATH` 选择，二进制SHA256固定。
- `cuda-compat-13-4` **615.71.09**，解包到临时目录后，在新 Python 进程启动前设置库路径。

不执行包安装或修改已有虚拟环境、FlagTree、系统工具、宿主机内核驱动。
Python 使用现有 vLLM 0.31.0 虚拟环境；两侧评分、我方 TopK 使用同一 FlagTree 模块。
原生 vLLM TopK 沿用其 CUDA 扩展，不由 FlagTree 编译。

## 正确性与计时

1. 两组预检覆盖原生 cooperative / persistent TopK，以及我方默认生产链路。
2. 在独立进程内运行整个 `tests/test_qsa_select_paged_prefill.py`，不排除原生 vLLM 比较，
   强制 `QSA_REQUIRE_VLLM_REFERENCE=1`，禁止转用另一个 Python 运行原生参考。
   检查完整测试结果和必需原生比较项；失败、意外跳过、身份变化都阻止 benchmark。
   仅允许单卡下既有的“需要第二张SM90卡”测试跳过。
   保持此前已移除的原生 BF16 H8/D256 共享内存超限用例不在直接比较参数中。
3. 运行上游12组、每组3阶段，共36条计时结果。各阶段计时前后继续做原有采样正确性检查。

| 阶段 | vLLM baseline | 我方生产算子 |
| --- | --- | --- |
| score | 原生 `_prefill_logits` | 生产 `_prefill_logits` |
| topk | 原生 CUDA `_topk` | 仓库 `persistent_topk`，TLE路径 |
| pipeline | 各chunk连续评分→原生TopK | 各chunk连续生产评分→生产TopK |

三阶段复用 `Benchmark.get_latency` 的 `--mode kernel` 计时器，默认1000ms warmup、
100ms repetition、7轮，交替两侧测量顺序。加速比为 `vllm_us / gems_production_us`。
**NV评分优化验收仍只看 score 阶段加速比，不含TopK**；其余两阶段用于观察总链路影响。
pipeline 是独立实测顺序调用，不是相加两个独立计时，也不包含public入口的外层分配与验证。
TopK单独计时的两侧读取相同冻结 logits，计时外核对全部输入哈希；组合计时各用实时评分结果。

预检、完整功能测试和所有case都验证同一编译器源文件、libtriton、TLE、ptxas及原生扩展身份。
各进程检查 `/proc/self/maps` 中实际加载的 `libcuda`，其路径和哈希必须匹配已校验的兼容库，
且 Driver API 至少为13040。清理旧的架构、PTX版本、kernel override等设置，使用独立编译缓存。
不会因为环境里存在另一份 stock Triton 的发行元数据就误判编译器；以实际模块路径及哈希为准。

默认自动选择空闲H800；也可设置单个完整GPU UUID。自动选择不构成独占预留，沿用共享机器
计时前后检查；其他进程干扰时不发布可接受加速比，不停止其他任务。

每个独立进程启动前保留原有空闲门槛：温度不高于75°C、利用率不高于5%、显存不高于256MiB、
无计算进程且没有 thermal slowdown。仅温度或滞后的利用率超限时，每10秒复查一次，最多等待
900秒（`--cooldown-interval` / `--cooldown-timeout`）。冷却等待位于计时区间之外；其他进程、
内存占用、实际热降频或无法读取遥测立即拒绝；等待超时仍退出2。不会放宽门槛或切换GPU混合结果。

[报告8681362](https://github.com/cheersluvs/FlagGems-vllm/blob/868136298de1717dbb705b8fe69d8731b0f772d4/results/qsa_select_paged_prefill/h800/flagtree134-ab-20261019-093620/stage-summary.json)
已完成172项功能测试，只有需第二张SM90卡的1项跳过；预检、完整正确性和首个benchmark的
编译器/实际libcuda/原生扩展身份一致。首组完成三阶段计时，第二组前空闲温度80°C触发原门槛。
因此加入上述冷却等待；尚未完成12组数据，不据此报告整体加速比。

## 运行与报告回传

在容器中执行整个命令块。外层 `if bash` 保留交互会话；成功和失败均自动尝试回传已有报告。

```bash
if bash <<'QSA_FLAGTREE134_AB'
set -euo pipefail
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
export QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python
export CUDA_VISIBLE_DEVICES=auto
test -x "$QSA_STOCK_PYTHON"
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-flagtree134-ab.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" rev-parse HEAD
bash "$qsa_repo/tools/run_qsa_prefill_flagtree_ab_h800.sh" \
  --flagtree /opt/flagtree --mode kernel --suite upstream-prefill \
  --rounds 7 --kernel-warmup-ms 1000 --kernel-rep-ms 100 \
  --timeout 1800 --accuracy-timeout 7200 \
  --cooldown-timeout 900 --cooldown-interval 10
QSA_FLAGTREE134_AB
then
  echo "正确性、12组三阶段测试和报告回传均完成。"
else
  echo "测试或回传未完成；当前会话已保留，请检查上方诊断。" >&2
fi
```

`stage-summary.json` 汇总正确性状态和计时；`accuracy-full.json/txt` 为完整功能结果；
`accuracy-runtime.json` / `preflight.json` / 各case JSON记录环境身份。
`triton-stage-ab.tsv` 的FlagTree模式使用 `gems_production_us` 列。
`compat134.json`、`flagtree134-tools.json` 记录下载工具身份，`sources.json` 保存源代码快照。
回传目录为 `results/qsa_select_paged_prefill/h800/flagtree134-ab-<UTC时间>/`，
结果分支仍为 `codex/qsa-h800-results-<UTC时间>`。缓存、下载工具保留在卡上，不上传。
