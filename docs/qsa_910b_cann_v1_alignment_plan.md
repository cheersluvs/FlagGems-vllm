# 910B INT8 QSA：CANN V1 结构对齐与固定推进顺序

更新日期：2026-11-11。本文完成第一步的源码结构盘点，不宣称性能提升或新的设备验证。

## 固定顺序与阶段门槛

| 顺序 | 工作 | 状态 | 完成门槛 |
|---|---|---|---|
| 1 | 建立阶段、dtype、缓冲区与依赖对应表 | 本文完成；列出的未知项保留为显式验证任务 | 每项有源码定位，区分已知事实、推断和未知项 |
| 2 | 以正确 N64 为起点，建立同结构串行版 | 待开始 | 先固定块划分、评分公式、精度、排序合并及输出契约；串行版对照通过；生成 IR 与设计一致 |
| 3 | 在同一实现上加入双缓冲 | 待开始，依赖第 2 步 | 同一算法、dtype、分块、TopK 状态与更新粒度；仅改变排程/缓冲槽；交错配对测量并证明实际重叠 |
| 4 | 考察第二次 Cube 归约及内部 tile | 待开始，依赖第 3 步 | 一次改变一项；所有中间值和最终选择通过；有收益才保留 |
| 5 | 生产入口集成与正式验收 | 待开始，依赖前述候选筛选 | 完整冻结 benchmark 表、正式入口正确性、harness `--mode kernel` 算术平均加速比；同时列评分、TopK 和整段数据 |

执行约束：第二步允许在 N64 上采用 CANN 的外层结构，但先用现有 FP32 评分作明确命名的变体；这不等于 CANN 数值路径已完全对齐。Fixpipe/FP16 与第二次 Cube 的实际数值路径在第四步单独加入验证。任何阶段回到前一步时记录原因，不以 tile 参数相同作为算法对齐的证明。

N128 不作为任何阶段的前置条件。当前 N128 混合候选未通过正确性，暂停其拼接/屏障试错，不进入生产。若日后第四步重启，必须先解决独立 Cube 正确、混合路径错误这一差异。

## 证据范围与实现身份

CANN 来源为本机保存的 `cann-qli-arch22-20261104` 四个头文件。下表记录快照摘要；它们不证明远端镜像当前二进制必然由这份源码构建，也不代表在线 master 的最新版本。最终行为仍需以固定镜像/环境中的 native 调用及设备对照为准。

| 源文件 | SHA-256 |
|---|---|
| `quant_lightning_indexer_kernel_arch22.h` | `3a87b22f93de7615e609600e51e89f5bf02fd600278731aa5da1b6922fe156d7` |
| `quant_lightning_indexer_service_cube_arch22.h` | `a8766ecb9a925425483cbc01fce1ba04fae64bb49346838626461069b209f1a2` |
| `quant_lightning_indexer_service_vector_arch22.h` | `c54265977f5ff0487eec7f7525de03776f5ea7038039e30735604d7d3cfe82ad` |
| `quant_lightning_indexer_vector.h` | `9b094a4d755dfe5568b06ea1f3cc584fe35599bd46c88141fff85a1cad7a796a` |

本机快照位置：`/Users/wuyuqing/workspace/qsa-ascend-baseline-20261029/cann-qli-arch22-20261104`。

| 本文名称 | 实际定位 | 范围 |
|---|---|---|
| INT8 验收整合入口 | `benchmark/test_qsa_select_paged_prefill_int8_perf.py::test_qsa_select_paged_prefill_int8_kernel` → `tools/qsa_int8_shared_map_selection_910b_trial.py::PreparedSharedMapInt8Selection` | 当前讨论的 INT8 控制/整合路径；分派到已验证评分路线及 TopK |
| 独立 TopK | `src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py`，由 `load_topk` 加载 | `persistent_topk` 名称下可有实际分派；必须记录策略、实际子 kernel，不能只看函数名 |
| N64 混合候选 | `tools/qsa_int8_online_topk_910b_trial.py::PreparedOnlineTopKSelection` → `tools/probe_qsa_int8_mixed_pipeline_910b.py::_qsa_online_scoped` | 手动 Cube/Vector scope、GM ring、64-key 局部排序与持续 Top512 |
| 通用 Ascend 算子入口 | `src/flaggems_vllm/runtime/backend/_ascend/ops/qsa_select_paged_prefill.py::_score` | 当前源码检查要求 BF16 Q/K；不可将它描述成已集成的通用 INT8 生产 API |

所有相对路径均相对仓库根目录。本文基于代码基线 `763347727c8d1d0de9c514cd86d366140da72faf`；后续文档提交不改变 kernel。

## 输入输出与数学契约

当前冻结 INT8 对照：Q 为 `[R,4,128]` INT8；cache 为 `[pages,192,1,128]` INT8；query scale/weight 按 query/head，key scale 按物理页/token；page table 和逐行 `visible` 为 INT32。CANN 源码相关 scale/weight GM tensor 使用 `half`。具体 stride 与连续性由适配层验证，不能从模型配置推定。

我们 CPU 参考计算：

```
d[r,h,k] = sum_D(Q_int8[r,h,D] * K_int8[physical(k),D])  # INT32
s[r,k]   = sum_h(weight[r,h] * max(float(d[r,h,k]) * qs[r,h] * ks[k], 0))
```

参考路径使用 FP32 评分，`k >= visible[r]` 为 `-inf`。输出为压缩序列中的逻辑组索引 INT32 `[R,512]`，不足位置为 `-1`；不能误用物理页地址或展开后的原始 token 索引。

当前 native harness 使用 `layout_query="TND"`、`layout_key="PA_BSND"`、`sparse_count=512`、逐行长度和逐行页表，得到 `[R,1,512]`，仅 squeeze head 维后对照。这是对压缩序列的选择测试，不包含模型层索引展开或 attention。

不得要求 tied-score 下的索引身份/排序完全一致来替代选择正确性；应验证范围、有效数量、唯一性、padding，以及所选分数集合与参考 TopK 的容差。候选内部数值另做逐元素检查，不能只用最终索引掩盖中间错误。

## 阶段对应表

| 阶段 | CANN V1 源码事实与定位 | INT8 验收整合入口 | N64 混合候选 | 对齐任务/约束 |
|---|---|---|---|---|
| 请求与任务划分 | `kernel_arch22.h`: `CalcGS1LoopParams`、`CalcS2LoopParams`、`ProcessMain`；S1_BASE_SIZE=4、S2_BASE_SIZE=2048，两个 runInfo 槽 | 按 request/chunk；共享页表证明控制 query 组、N64/其他评分路线 | 2 query/program，跨同一 request 的 query 共享页表 | 对齐工作量与 request 边界；不要把我们 32 行诊断 chunk 误当成 CANN S1=4 |
| 可见范围与空输入 | 同文件 `CalcS2LoopParams`、`DealActSeqLenIsZero`，按实际范围计算任务 | `visible`、有界路线与掩码；不同分派可能跳过 tile | 扫描指定宽度，评分后对不可见位置置 -inf | 先统一实际处理范围，再比较跳过策略；全不可见输出必须保留 |
| query/weight 预处理 | `service_vector_arch22.h::ProcessVec0`：half weight×half query scale，广播后存 weightWorkspace GM | 评分 kernel 内将 scale/weight 升 FP32；部分历史合并候选不等于已上线 | 每个 tile 加载 scale/weight，升 FP32 | 第二步明确采用 FP32 参考变体；预合并/缓存的准备成本计入整算子 |
| Q/K 装载与复用 | `service_cube_arch22.h::QueryNd2Nz`、`KeyNd2NzForPA`、`LoadKeyToL0b`：分页 GM→L1→L0；query/weight 跨 key 内循环复用 | 由评分分派及共享页表路线决定 | Q 在 Cube 循环外加载；每轮装载 N64 K | 复用是否落实需检查编译产物；不复制物理地址管理假设 |
| QK | `ComputeQk`：INT8×INT8，L0C INT32，内部 S2_BASIC_BLOCK_L0=128 | `tl.dot` INT32；按评分路线采用不同 tile | M16×K128×N64，2 query×4 heads 只有 8 个有效 M 行 | 第一阶段使用正确 N64，不强求 N128 |
| QK 后激活/转换 | `FixpSToL1`：DEQF16、reluPre=1、SetFixpipePreQuantFlag(0x3a800000)，结果进 sL1 | INT32→FP32，乘尺度、ReLU | Vector INT32→FP32，乘尺度、ReLU | 数值顺序、half 舍入和 Fixpipe 参数是显式差异，不能声称逐值等价 |
| head 加权归约 | `ProcessWs`、`ComputeWs`：half weight 与 half S 做第二次 Cube Mmad，结果 FP32 | Vector 上 FP32 加权求和 | Vector 上 FP32 加权求和 | 第四步单独验证第二次 Cube，含布局转换及全成本 |
| score 交接 | `FixpResToGm`：归约后的 FP32 `[query,key]` 分数进双槽 mm1ResGm | 评分写 FP32 chunk logits，独立 kernel 的 TopK 后续读取 | Cube 先将逐 head INT32 点积写 ring，Vector 再归约 | **交接对象不同**：CANN 交接已归约分数；我们的 ring 是逐 head 点积，字节量和消费者工作不同 |
| key scale | `ProcessVec1`/`GetKeyScale`：half key scale→FP32，在归约后乘分数 | 参考评分在归约前乘 key scale | 同参考评分 | 对正 key scale，在实数算术下可交换；有限精度、负 scale 的契约不能省略检查 |
| 局部排序 | `ProcessVec1`：对有效外层块补齐后 `SortAll` | 实际 TopK 路由可为 radix/分段排序合并等 | sort_1d_pack，每 64-key 更新 | 第二步先冻结统一排序/合并算法，不能把当前两条不同 TopK 路径只称串行/双缓冲 |
| 持续 TopK | `BASE_TOPK=2048`，`globalTopkUb_`；`ProcessVec1` 调用 `MergeSort` | 独立 TopK 读取 chunk logits；32 行子批，scratch 复用 | Top512 状态，UB snapshot、merge_exhaust_sort4、尾部保留，每 64-key 更新 | 2048 内部状态与输出512是不同概念；对齐变体应注明状态容量和截断时机，不能静默改动 |
| 多段结束与输出 | Vector `ProcessVec1`、`ProcessLD` 及 `ExtractIndex`/MergeSort；存在 workspace 中间结果/最终输出条件 | 各请求 chunk 的输出拷入 `[R,512]` | 最终 unpack_sort 后按 visible 填 -1 | 必须覆盖多请求、尾 query、短 key、ties；模型层展开不属于此处 |
| 排程 | `ProcessBaseBlock`：AIV ProcessVec0 准备当前权重，AIC ComputeMm1 当前块，AIV ProcessVec1 消费上一块；首尾填充/排空 | 每 chunk 先评分、再独立 TopK | Cube/Vector 两槽，当前 tile 产生，Vector sort/merge 后归还槽 | 第三步对照只能改变排程，不能同时改变 TopK 算法或交接对象 |

## 缓冲区与依赖

| 缓冲区 | CANN V1 | 当前实现与需要区分之处 |
|---|---|---|
| Q/K、weight 局部存储 | L1、L0A/B；显式 Nd2Nz/LoadData 与事件管理 | Triton tensor 的物理空间和实际复制由编译产物确认，不从变量名推定 |
| QK 结果/激活结果 | L0C INT32 → Fixpipe → L1 half S | N64 混合候选是 INT32 GM ring；并未复现 CANN 本地 S 流转 |
| 归约分数 | L0C FP32 → mm1ResGm 双槽 | 验收整合入口为 FP32 chunk scratch；混合候选当前没有同样的已归约分数双槽 |
| TopK 状态 | UB globalTopkUb_，另有 GM 中间状态及结束合并 | 混合候选的 Top512 与 snapshot 是另一种实现；其显式 UB 拷贝/屏障必须计入 |

CANN 依赖图（阶段标签，不代表我们已复现的耗时重叠）：

```text
Vec0: (weight × query_scale)_half → weightWorkspace GM
                                      ↓ syncV0C1
Cube: QK_INT32 → Fixpipe(ReLU/half) → second Cube weighted reduction
                                      ↓ FP32 mm1ResGm / syncC1V1
Vec1: key_scale → SortAll → MergeSort → persistent UB state → final/workspace output
                                      ↓ syncV1C1: 读完并可回收当前 GM 槽
```

Q/weight 的回收另受 `syncC1V0` 等事件约束。GM 槽可回收、L1 输入可覆盖、L0C 可复用是三类依赖；一个笼统 barrier 不能证明三者都正确，也不能证明有效流水重叠。

## 已验证事实与未决项

- 最近 N128 报告 `335fe11be542cbd52a9dc39a6a9a6ed124c4dcfa`：独立分页 Cube 的 Q/K/点积通过，混合交接仍失败；离散加载、三维拼接与单纯加 barrier 均未解决。尚未证明具体编译器根因。
- 最近可用 chunk 诊断 `e7d7df23836927ba855475c68501429c7ad5a86d`：compact 的正确 N64 混合版仍慢于验收整合入口；双缓冲的生产收益没有得到证明。这不是完整冻结 shape 表的验收值。
- `FixpSToL1` 的预量化标志常量、有效缩放、half 溢出/饱和及潜在统一正比例因子，需要结合设备行为确认；当前文件集不足以保证逐值还原 native 内部分数。
- `MergeSort` 的 exhausted 行为不能按名称一概而论；不同合并调用有不同参数，需要在第二步冻结具体调用与尾部保留算法。
- 具体 CANN binary 与此源码快照的构建对应关系、硬件指令级时序尚未证明。源码结构相似不等于二进制排程相同。

## 第二步的交付定义（尚未实施）

先交付设计冻结清单与串行 probe，记录：request/query 分组、外层 key 块、内部 N64、FP32 评分变体的公式、交接对象、局部排序块、持续状态容量、结束合并、临时空间和输出契约。对每种结构选择明确“沿用参考”或“对齐 CANN”，禁止同时引入未命名的其他优化。

必须建立同输入的逐层对照：INT32 点积精确一致；FP32 参考变体评分 rtol/atol=1e-3；局部排序排列/索引/分数；持续合并每轮状态；最终选择与 padding。后续 half/Cube 归约变体使用单独的数值误差记录，不能放宽现有参考检查来隐藏差异。

串行版通过后冻结为第三步唯一算法控制。第三步用相同输入与算法比较串行和双缓冲，同时报告评分、TopK、整体耗时；正式验收仍是第五步的完整 shape 表整算子 harness kernel 算术平均加速比。

## 第二步首轮串行 probe（已准备，待设备验证）

入口：`QSA_HARNESS_CANN_SERIAL=1`，仍由 chunk harness 的两组 32 行输入驱动。
实现：`tools/qsa_cann_serial_910b_trial.py`；验证：`tools/probe_qsa_cann_serial_910b.py`。

本轮冻结：内部 N64，外层 2048-key，FP32 参考评分公式；每块完整 sort2048，持续 Top2048，最终 Top512。第一个块直接初始化状态；后续两个等长排序列表做 exhaustion merge，检查已消费总数至少 2048，再截取前 2048。两条输入都长 2048，首次耗尽不可能早于产出 2048，因此本控制不需要 CHUNK64/Top512 候选中的尾部 gather 补全。

每个外层块严格按“评分→局部排序→状态合并”依赖执行。两个 GM 状态缓冲只是避免读写别名，并没有并发执行或流水重叠。score 与 sort/merge 的 kernel 边界提供可验证的依赖；当前不做 CANN 原始事件级调度。

本轮刻意沿用既有单行 N64 评分模板，不声称已复现 CANN 的 query4 数据复用、UB 驻留状态、Fixpipe half 转换或第二次 Cube。它是第二步的首个可核对算法控制；阶段二尚未完成，待设备结果决定后续补齐项。

完整校验包括：所有行的每块 INT32 点积、FP32 评分、局部排列/分数/排序、每次合并的 Top2048 状态、最终 Top512 与 padding。临时 RawDot 只在校验调用中写入，计时调用不写；额外导出编译 IR。评分、TopK-only、串行整段及既有整合控制都通过相同 harness `get_latency`、kernel 模式测量，两轮交错顺序，列出波动；均是诊断 chunk 数据，不给正式验收平均值。

若编译资源超限或任一中间状态校验失败，停止计时并回传失败；不会静默将内部状态改成 Top512 来让测试通过。


### 首轮设备结果：2026-11-11

报告提交 `5176365df2903bf728566562160dcd21202a70a3`，代码 `72e0b55308a6c7ca8739a5980f0f2cad8943f3ff`：budget_exact 四个块及 compact 一个块的点积、评分、局部排序、Top2048 状态、最终输出检查均通过；N64 点积和 merge2048 的 IR 已核对。第二步首个串行算法控制的正确性成立，尚未完成整个第二步。

性能采集中出现 SQLite TASK 表缺失、无 profiling 数据；部分样本为 Infinity，派生量为 NaN。原 probe 未拒绝这些值而返回成功，这是报告校验缺陷。该报告只能记录“正确性通过、计时不完整”，不能据此形成完整性能结论或作为第三步已具备可靠串行耗时的证明。修正后同一 harness 的无效计时最多重试一次，仍无效则失败；不更换计时方法，保留每次尝试记录，JSON 禁止非有限数值，摘要独立复核两轮耗时。


### 第二步边界补测（待设备验证）

在两组原输入之外，只增加三个不计时的小型正确性 fixture：17 行、请求长度 3/5/9、独立滚动页表映射，visible 覆盖 0/1、63/64、127/128、191/192、511/512/513、2047/2048/2049、4095/4096/4097；3 行的全零评分 ties，扫描宽度2049；1 行全不可见、扫描宽度1。沿用逐块点积、评分、排序和状态校验，不要求 ties 的索引身份一致。这些是下层逐行页表接口的合成 fixture，不等同于真实 scheduler/model forward 测试，也不新增正式 benchmark shape。

所有边界验证安排在计时前，后续 profiler 失败也会保留边界结果。基准两组仍使用同一 harness kernel 计时，只有无效采样最多重试一次。
