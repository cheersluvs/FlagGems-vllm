# 摩尔线程：同 token 连续 head 分块 A/B/C 实验

本目录是独立实验，不修改正式算子、注册、测试 gate 或 dispatch。
准备日期：2026-10-12；冻结基线：`ebe39aa1c129142490b88301a2e641c2ef786e68`。
Mac 无摩尔线程卡，远端正确性和性能尚未验证；不能据此宣称优化有效。

## 编码前冻结：项目真相

| 字段 | 本实验约定 |
| --- | --- |
| Current op | fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert；无单一 torch API，以仓库测试 oracle 为语义参考 |
| Editable / read-only | 只新增本目录；src、tests、benchmark 不改；A 是该提交中摩尔线程源文件的逐字冻结副本 |
| Build | 无需安装包；FlagTree JIT；`python -m compileall -q bench-scripts/mthreads-fused-abc` 做静态检查 |
| Validation | `python bench-scripts/mthreads-fused-abc/run.py --validate-only`；任一检查失败立即停止，不计时 |
| Benchmark | 同一命令去掉 validate-only；eager MUSA events、ms（可能含 host 发射间隙）；SpeedUp=A/候选，越大越好 |
| Active set | 默认 tokens=512,1024,2048,8192,32768、heads=64、block_size=64、BF16；显存不足显式跳过，报告不完整 |
| Aggregation | 每轮重复取中位数，再对轮中位数取中位数；逐 shape 报告；最小/最大/算术平均加速比不混用 |
| Timeout | 远端用外部 timeout（若可用）：validate 20 分钟、完整 benchmark 60 分钟；编译失败立即停止 |
| Profiler | 当前未知；记录编译 metadata / n_regs / n_spills / shared（若可获得），不把它们当实测占用率 |
| Autotune | 实验豁免 NVIDIA libtuner/注册；显式 sweep HPP=4（可选 1,2,4,8），warps=4、stages=2；不挑最佳值掩盖回归 |
| Fallback | 不使用 torch 计算 fallback；torch 只在输入构造、CPU oracle、检查和计时区外的重置中使用 |

## 编码前冻结：语义契约

| 字段 | 本实验约定 |
| --- | --- |
| Reference | 仓库 tests/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py 中 RMSNorm/GPT-J RoPE/UE8M0 FP8 规则；本目录实现独立 CPU oracle |
| Device | torch.musa（先 import torch_musa）；记录 torch/FlagTree 路径、版本、llc 路径和哈希，不凭版本字符串断言编译器可用 |
| Input | q=[N,H,512]、kv=[N,512] 连续 BF16；slots=[I] int64, 0<=I<=N；pos=[N] int64；cos_sin=[P,64] FP32；同设备 |
| Output | q 原地写；kv 只读；cache 连续 uint8 [blocks,stride]；每 token 576 数据字节，独立 scale 区每 token 8 字节 |
| Acc/dtype | RMSNorm/旋转/量化 FP32；q 和 KV RoPE BF16；非 RoPE KV FP8 E4M3FN + UE8M0 scale |
| Semantics | 有限 BF16、eps>0；负 slot 不写，短 slots 不插入尾部 token；合法 slots 必须唯一且不越界；其他 padding 保持哨兵 |
| Comparison | Q 与 CPU oracle rtol=atol=1e-2；cache 量化/scale/padding 与 oracle 精确，RoPE 与 oracle 1e-2；B/B1/C 的完整 cache 与 A 逐字节精确 |
| Autograd/unsupported | forward-only；不支持 NaN/Inf、非连续、重复/越界 slots、其他 dtype、backward；实验构造器拒绝非法参数，无 silent fallback |
| Torch use | 仅驱动/测试；候选 JIT 和 launcher 无 torch compute |

## 编码前冻结：路径设计

| 路径 | Q 映射 | KV 映射 | 配置 / 验证 / 风险 |
| --- | --- | --- | --- |
| A | 每 program 4 个 token，同一 head | 独立 program，4 token | 冻结已提交的 _tiled_kernel，warps=4/stages=2；小形状/128 heads 强制实验路径，不等同当前 public dispatch |
| B | 每 program 1 token 的 HPP 个连续 head | 独立 program，4 token | 相同 Q 元素数（HPP=4）和 KV 粒度；验证 tail mask、cos/sin 广播；布局变更可能改变资源使用 |
| B1（辅助） | 同 B | 独立 program，1 token | B→B1 分离 KV 粒度的影响；Q/KV 独立，但仍一次 kernel launch |
| C | 同 B | 最后一组 Q 完成后，处理该 token 的 KV | B1→C 才是合并的净对照；所有 Q head 都算，包括最后一个，绝不以 KV 取代最后一个 Q |
| Empty/error | 参数检查，无计时 | 同左 | N<=0/非法参数拒绝；空 slots 允许，且验证 cache 不被修改 |
| Production/general | 不变 | 不变 | 不接入正式 fast/general 路径，无外部库实现，无生产收益验收 |

A/B 的 Q 每 program 元素数在 HPP=4 时相同；HPP sweep 是单独实验。
B/C 直接比值同时包含 KV 粒度变化与合并，不应全部归因于内存连续性。
程序还计时 A2（同 A、独立一次测量），给出 A/A 噪声参考。

## 运行

在已激活摩尔线程环境、远端仓库根目录执行（不使用 T-Head SDK wrapper）：

```bash
python bench-scripts/mthreads-fused-abc/run.py --validate-only --output /tmp/mthreads-abc-check.json
python bench-scripts/mthreads-fused-abc/run.py --output /tmp/mthreads-abc.json
# 可选第二阶段参数扫描，保留全部配置结果
python bench-scripts/mthreads-fused-abc/run.py --head-tiles 1,2,4,8 --output /tmp/mthreads-abc-sweep.json
```

`--quick` 使用 tokens=512,1024、rounds=3、repeat=10；每个实际计时 shape 都先对拍。
固定边界集合覆盖 token tail、head tail、block_size=16/64、空/短/负/逆序 slots、零和量化边界值。
默认先跑全部固定边界测试，再逐形状 validate→benchmark。
计时每次先恢复原始 q/cache，重置和同步不在 events 区间；每个 arm 一次目标 kernel launch。
各 arm 共用输出 buffer，轮换顺序，计时前清空等待队列；属于重置后的 warm-cache 单 kernel 延迟，非端到端吞吐。
每轮有原始 samples；没有用反复归一化的 q 输入计时。不能从此直接推断 HBM/占用率已打满。
输出 JSON 及同名 CSV；异常也写 JSON（含阶段/已完成结果），并非零退出。
已有同名 JSON/CSV 时拒绝覆盖，请换一个 `--output`。

计时口径明确是 eager event latency，不是纯硬件执行周期；特别小的 shape 可能受 CPU 发射间隙影响。
没有采用 graph replay：历史同机器记录显示 graph replay 存在显著固定开销，会稀释微秒级加速比。
若 A/A 噪声与候选收益接近，不宣称稳定改善；完整实验建议再独立运行一遍并保留两个报告。
MUSA Event API 参考：[torch_musa 官方说明](https://github.com/MooreThreads/torch_musa#usage)。

## 远端拉取：不要切换整个仓库分支

Mac 的 managed-worktree 工具未识别当前任务关联的仓库，所以本实验使用 workspace 内的独立 git worktree。
远端 origin 历史上不是 cheersluvs 的 fork，始终使用显式 URL。
以下命令在远端 FlagGems-vllm 仓库根目录、已激活摩尔线程 Python 环境下执行。
若目标路径有修改/未跟踪文件，则停下，不覆盖它们；不需要安装 vLLM 或导入插件。

```bash
(
  set -euo pipefail
  git rev-parse --show-toplevel
  git status --short -- bench-scripts/mthreads-fused-abc
  if [ -n "$(git status --porcelain -- bench-scripts/mthreads-fused-abc)" ]; then
    echo "实验目录已有改动，请先保留这些改动；停止拉取。"
    exit 1
  fi
  git fetch https://github.com/cheersluvs/FlagGems-vllm.git codex/mthreads-fused-abc-20261012
  git log -1 --oneline FETCH_HEAD
  git checkout FETCH_HEAD -- bench-scripts/mthreads-fused-abc
  python bench-scripts/mthreads-fused-abc/test_plan.py
  ABC_RUN_DIR=$(mktemp -d /tmp/mthreads-abc.XXXXXX)
  echo "结果目录：$ABC_RUN_DIR"
  python bench-scripts/mthreads-fused-abc/run.py --validate-only --quick \
    --output "$ABC_RUN_DIR/check.json" 2>&1 | tee "$ABC_RUN_DIR/check.log"
  python bench-scripts/mthreads-fused-abc/run.py \
    --output "$ABC_RUN_DIR/result.json" 2>&1 | tee "$ABC_RUN_DIR/result.log"
)
```

将结果目录中的 `result.json`、`result.csv`、`result.log` 发回即可分析。
脚本直接从同目录 import 冻结 A 和候选 kernel，记录绝对路径和哈希，不受旧的包安装路径影响。
本地 CPU 检查：`test_plan.py` 4 项通过；Black/isort/flake8、语法检查通过。
`optimization.md` 和 `deep_opt.md` 在本提交中不存在；未据缺失文档杜撰指导。
GPU correctness / compile / latency gate 均等待摩尔线程执行；准备实验不是生产优化验收。
