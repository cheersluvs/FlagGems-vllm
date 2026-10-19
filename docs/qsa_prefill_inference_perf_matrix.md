# QSA prefill：历史22组诊断矩阵

> 2026-10-19 更新：当前正式 benchmark 仅保留上游 vLLM 的12组，见
> [当前矩阵](qsa_prefill_upstream_perf_matrix.md)。本文保留历史实验及显式诊断用例；下述默认值和命令不代表当前正式配置。

## 范围与来源

默认 `inference-prefill` suite 为 **12 个上游公开布局 + 8 个推理长度候选 + 2 个布局变体**。
`benchmark/test_qsa_select_paged_prefill_perf.py` 的 core/comprehensive 和 H800 runner 共用
`tools/qsa_prefill_inference_workloads.py` 定义。显式 `--suite upstream-prefill` 仍只跑原有 12 组。

固定 BF16、H=4、D=128、KV heads=1、压缩率=4、token_topk=2048、block Top-K=512、
page=196、logits 预算 512 MiB。性能验收保持 `--mode kernel`，计时不含 Top-K。
生产算子、Top-K 和调优候选未改动；两侧仍使用同一实际编译器。

新增 10 组是按已核实调用链构造的候选，**不是完整模型的实测入口 trace**，也没有业务频次权重。
当前没有部署缓存 trace；两组布局变体显式标为 synthetic。现有 12 组的历史 H800 报告不作为新增
10 组已通过的证据。新增输入必须在卡上跑完本轮校验后再验收。

来源：

- [上游公开性能表 PR #54513](https://github.com/vllm-project/vllm/pull/54513)。
- [模型配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)。
- [prefill 调用：切片、绝对 starts、公共 max_seq_len](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/models/qwen4_exp/nvidia/indexer_qsa.py#L428)。
- [cache planner 的 block/layer stride 规则](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/kv_cache_interface.py#L1423)。

## 原有 12 组

每种 query 布局与 Prefix 的三种取值作笛卡尔积，合计 12 组；原有 case 名称、参数和输入生成保持不变。
Prefix 是当前 query 之前的原始 token 数，K=1024 token。

| 每请求 query 布局 | Prefix |
| --- | --- |
| 1×8192 | 8192 / 51200 / 102400 |
| 1×16384 | 8192 / 51200 / 102400 |
| 2×8192 | 8192 / 51200 / 102400 |
| 4×4096 | 8192 / 51200 / 102400 |

完整 tensor shape 和历史报告见[原矩阵](qsa_prefill_upstream_perf_matrix.md)与
[形状研究](qsa_prefill_benchmark_inference_shapes.md)。

## 新增 8 组长度候选

表中 case 名省略共同前缀 `infer_` 和后缀 `_bfloat16`。Q 为 `[T,4,128]`，输出 `[T,512]`。
Prefix 与 query 按请求逐一对应；逐行 visible 为 `floor((prefix_i+j+1)/4)`，每个请求重新起算。
普通场景的缓存为 `[Σceil(ceil((prefix_i+q_i)/4)/196),196,1,128]`，物理页随机打乱。

| case | query_lengths | Prefix | T | logits W | 评分 chunk 行数 |
| --- | --- | --- | ---: | ---: | --- |
| short128 | [128] | [8192] | 128 | 2112 | 128 |
| short512_long | [512] | [102400] | 512 | 25728 | 512 |
| scheduler_tail | [2048] | [49152] | 2048 | 12800 | 2048 |
| first_chunk | [8192] | [0] | 8192 | 2048 | 8192 |
| unequal | [4096,2048,1024,1024] | [8192,51200,102400,0] | 8192 | 25856 | 5190/3002 |
| many_requests | [256]×32 | [8192]×32 | 8192 | 2112 | 8192 |
| skewed | [7168]+[64]×16 | [51200]+[8192]×16 | 8192 | 14592 | 8192 |
| mixed_decode | [4096] | [4096] | 4096 | 51200 | 2621/1475 |

`mixed_decode` 模拟前面一个 decode token，其 sequence 长度为 204800；prefill 自身 sequence
长度是 8192。Q/visible/output 是去掉 decode 首行后的 view，starts 为 `[1,4097]`，page table
为 `[1,335]`，保留公共 `max_seq_len=204800`。335 是本例根据 262144 模型长度和 page196
选取的容量 `ceil(262144/(4*196))`，不是声称所有部署的 runner 都分配335列。

这一组覆盖“短 prefill、长 decode 上界”的评分分块。只有 prefill 切片进入本算子，
decode 自身不计入评分计时。未引用真实请求的内容或调用概率。

## 两组布局变体

各在 `short128` 和 `mixed_decode` 上新增一个 `_global_strided_bfloat16` 变体，总数为22。

| 参数 | 构造值 | 目的 |
| --- | --- | --- |
| cache shape | `[4096,196,1,128]` | 使用大于本例活跃页数的全局池 |
| page table | `[1,335]` | 容量与活跃页数分开，尾部列填 -1 |
| backing shape | `[4096,2,196,1,128]` | 模拟两层页面交错的 block-outermost 布局 |
| 本层 cache view | `backing[:,1]` | 保留跨页 stride 和非零层偏移 |
| cache stride，单位元素 | `[50176,128,128,1]` | 页步长为紧凑布局的两倍 |
| cache storage_offset，单位元素 | 25088 | 第二层页面起点 |
| 物理页映射 | 从4096页中随机选出活跃页并重映射 | 保持逻辑 key 内容相同，改变物理分布 |

4096页、两层和335列是可复现的布局探针参数，不是从用户部署采集的值。
变体组合了容量、页分布和stride因素，不能用一次差异分别归因于其中某一项。
拿到入口 trace 后可用真实参数替换，但应保留独立来源标签与历史结果。
所有未使用页和层间存储初始化为零，避免输入哈希依赖未初始化数据。

## 校验与计时

输入生成、重映射、layout 构造、校验均在计时之外。功能检查保留原有独立完整测试，
每个 benchmark case 另做独立 FP32 检查。采样覆盖每个请求的首尾行和每个评分 chunk 的边界，
防止长 query 掩盖短请求错误。计时仍通过共享 `Benchmark.get_latency` 的 kernel 分支执行，
不改变 do_bench 策略，也不引入跨编译器的原生 vLLM 性能比较。

报告增加 `inference-prefill-shapes.tsv`；manifest 和每个 case JSON 保存来源、预期/实际
shape、stride、storage_offset、绝对 starts、max_seq_len 和评分 chunk。worker 在计时前
检查这些布局与预期一致，重映射前后逻辑 key 内容在 CPU 构造测试中检查。

## H800 运行与自动回传

在原容器中执行；GPU UUID 沿用此前通过的物理 GPU。需要换卡时替换该变量。
这是新一轮22组测试，不复用旧12组或38组报告。脚本即使测试失败，也会在已有汇总报告时
尝试回传诊断。外层 if 保留当前容器会话。

```bash
if bash <<'QSA_INFERENCE_RUN'
set -euo pipefail
trap 'qsa_rc=$?; printf "失败：退出码=%s，行=%s，命令=%s\n" "$qsa_rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$qsa_rc"' ERR
source /tmp/qsa-vllm031-h800.PQnVpV/activate_qsa.sh
export PYTHON=/tmp/qsa-vllm031-h800.PQnVpV/venv/bin/python
export QSA_TRITON_RUNTIME_PATH=/opt/flagtree
export CUDA_VISIBLE_DEVICES=GPU-2d17872f-4327-2beb-8049-604ecf6a3d66
export QSA_PREFILL_AB_KIND=single-row
test -x "$PYTHON"
test -d "$QSA_TRITON_RUNTIME_PATH/triton"
qsa_repo=$(mktemp -d /tmp/flaggems-qsa-inference.XXXXXX)
git clone --depth 1 --single-branch \
  --branch codex/qsa-select-paged-prefill-h800 \
  https://github.com/cheersluvs/FlagGems-vllm.git "$qsa_repo"
git -C "$qsa_repo" status --short
git -C "$qsa_repo" rev-parse HEAD
bash "$qsa_repo/tools/run_qsa_prefill_validation_ab_h800.sh" \
  --suite inference-prefill --mode kernel --require-idle-gpu \
  --rounds 7 --kernel-warmup-ms 1000 --kernel-rep-ms 100 --timeout 1800
QSA_INFERENCE_RUN
then
  echo "22组测试和报告回传均完成。"
else
  echo "测试或回传未完成；当前会话已保留，请检查上方诊断。" >&2
fi
```

A/B 的单行 vector 试验入口沿用现有 runner 接口；这22组预期评分 chunk 均大于1行，
因此两侧都执行当前生产 matmul 路径，测得的是生产评分基线与对照波动。
