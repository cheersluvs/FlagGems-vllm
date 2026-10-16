# QSA prefill：上游 Triton 三阶段 A/B

## 本轮实验

入口：`tools/run_qsa_prefill_triton_ab_h800.sh`，默认运行现有22组 `inference-prefill` 矩阵。
本轮只增加对照脚本，不进行评分优化，不修改生产源码、Top-K源码、调优配置和已有benchmark矩阵。

| 阶段 | vLLM baseline | FlagGems 测试变体 |
| --- | --- | --- |
| score | 原生 `_prefill_logits`，上游 Triton 3.6.0 | 当前生产 `_prefill_logits`，同一上游 Triton |
| topk | 原生 `_topk`，保留 CUDA 扩展 dispatch | 仓库 `top_k_per_row_prefill`，Triton 非TLE分支 |
| pipeline | 每个chunk连续执行原生评分→原生Top-K | 每个chunk连续执行生产评分→`top_k_per_row_prefill` |

vLLM 原生 Top-K 是 CUDA 扩展，并不由 Triton 编译；其实现保持原样。当前22组的chunk行数均大于64，
原生Top-K使用persistent dispatch；预检另外验证8行cooperative和128行persistent两种原生路径。
“同编译器”指双方评分kernel及我方Triton Top-K都使用同一份上游Triton，不涉及FlagTree切换。

仓库实际函数名为 `top_k_per_row_prefill`。脚本建立生产函数的私有Python模块副本，保留相同的
public函数字节码与评分kernel对象，只在副本里将 `persistent_topk` 绑定到调用上述函数的适配器。
默认生产模块中的 `persistent_topk` 不会被替换。生产源SHA256固定为
`d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73`，变更后须重新审计实验。

原生vLLM文件仍使用既有参考提交 `7a877ae34dcf4a7d46de08eb16db4699e1e0047c`，SHA256为
`4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b`。
该QSA文件已核实与v0.31.0标签的对应模块相同。脚本使用原文件，不更改cache hint或kernel配置。

## 计时边界

三个阶段都通过仓库共享 `Benchmark.get_latency` 的 `--mode kernel` 分支测量，沿用
`triton.testing.do_bench` 默认cache策略，默认1000ms warmup、100ms repetition、7轮。
每阶段每轮交替baseline/candidate顺序，保留全部原始样本、两侧中位数、加速比和配对比值。

- **评分**：执行本次public调用实际产生的全部评分chunk，不调用Top-K。
- **Top-K**：两侧读取同一份原生评分生成的冻结logits、相同的visible与chunk划分，输出缓冲区独立。
  所有chunk的logits都在计时外准备；前后校验完整输入内容哈希未变化。
- **合计**：实际测量每个chunk的“评分→Top-K”序列。不是两个独立阶段时间相加，也不是新的融合kernel。

每个阶段的加速比均为 `vllm_us / gems_prefill_topk_us`；pipeline采用其独立实测时间计算。
外层工作区分配、public参数验证、输入构造、观察/拷贝logits、FP32校验、Top-K适配器所需的全零
row_starts准备均在计时之外。两侧沿用各自public实际分块与原始workspace布局，运行前检查其
chunk offset/行数/宽度一致，且符合22组矩阵定义。

`top_k_per_row_prefill` 内部原有scratch分配保留在调用中，未重写或移出该API。
因此结果体现仓库现有非TLE实现及共享kernel计时器定义的表现；不能当作排除所有内部管理成本的
纯指令耗时，也不能当作默认persistent_topk生产版本的完整算子延迟。

## 环境和正确性检查

使用现有vLLM 0.31.0虚拟环境中的上游Triton 3.6.0和原生CUDA扩展。脚本不安装、卸载或替换包。
Python通过 `-I` 启动，显式加入本checkout的src/root，清除 `/opt/flagtree` 的环境路径覆盖。
检查Torch/Triton来自所选虚拟环境、Triton发行元数据和wheel RECORD文件哈希、没有TLE包，
并确认vLLM、生产评分和我方Top-K引用的是同一Triton/language模块。

报告记录实际编译器、libtriton、ptxas路径与哈希，以及已加载的vLLM原生扩展路径与哈希。
各case使用独立进程和报告内独立编译/调优缓存。preflight与所有case的编译器、GPU、原生扩展
身份必须一致，GPU闲置/计时环境检查沿用已有共享机器检查逻辑。

本实验的正确性证据是：

1. 8行和128行预检，覆盖原生两种Top-K dispatch及我方非TLE实现。
2. 每组评分阶段计时前后，按请求首尾、评分chunk边界和均匀采样行做独立FP32评分检查。
3. Top-K阶段计时前后，检查有效数量、范围、唯一性、-1填充、选中分数集合，直接比较原生与我方结果。
4. pipeline阶段计时前后，用独立FP32分数检查两侧完整输出的采样行，并直接比较选中分数集合。

这是阶段benchmark的采样检查；已有默认生产实现的完整功能测试报告单独保留。
编译、正确性、身份或环境检查失败的case不会在汇总TSV发布可接受的加速比，原始失败记录仍回传。

## H800运行与自动回传

在原H800容器中执行整段。激活脚本用于准备现有Torch/CUDA兼容环境；新runner随后清除
FlagTree路径，并以 `-I` 选择虚拟环境内的上游Triton。如果虚拟环境已被FlagTree覆盖，预检会拒绝，
不会自动修改该环境。外层if保证命令失败时保留当前容器会话。

```bash
if bash <<'QSA_STOCK_TRITON_AB'
set -euo pipefail
trap 'qsa_rc=$?; printf "失败：退出码=%s，行=%s，命令=%s\n" "$qsa_rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_rc"' ERR
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
export QSA_STOCK_PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-2d17872f-4327-2beb-8049-604ecf6a3d66
test -x "$QSA_STOCK_PYTHON"
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-stock-triton.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" status --short
git -C "$qsa_repo" rev-parse HEAD
bash "$qsa_repo/tools/run_qsa_prefill_triton_ab_h800.sh" \
  --mode kernel --suite inference-prefill \
  --rounds 7 --kernel-warmup-ms 1000 --kernel-rep-ms 100 --timeout 1800
QSA_STOCK_TRITON_AB
then
  echo "三阶段benchmark和报告回传均完成。"
else
  echo "测试或回传未完成；当前会话已保留，请检查上方诊断。" >&2
fi
```

默认自动下载并校验固定的原生QSA文件。已有该原始文件时可传
`--reference-file /absolute/path/qsa_indexer_vllm.py`，仍须通过同一SHA256检查。
`--suite upstream-prefill` 可只跑原有12组；`--cases <name>`用于显式子集诊断，汇总记录实际active set。

输出：`stage-summary.json`、`triton-stage-ab.tsv`、`preflight.json/txt`、各case的JSON/TXT、
`sources.json`、`manifest.json`、`runner.txt`及`run-status.txt`。完整22组预期有66条阶段数据。
回传位置为 `results/qsa_select_paged_prefill/h800/triton-ab-<UTC时间>/`，结果分支沿用
`codex/qsa-h800-results-<UTC时间>`。测试失败时也会尝试回传已有报告，并保留测试退出码。
