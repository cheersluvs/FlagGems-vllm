# 910B INT8 QSA：CANN V1 结构对齐与固定推进顺序

更新日期：2026-11-11。本文完成第一步的源码结构盘点，不宣称性能提升或新的设备验证。

## 固定顺序与阶段门槛

| 顺序 | 工作 | 状态 | 完成门槛 |
|---|---|---|---|
| 1 | 建立阶段、dtype、缓冲区与依赖对应表 | 本文完成；列出的未知项保留为显式验证任务 | 每项有源码定位，区分已知事实、推断和未知项 |
| 2 | 以正确 N64 为起点，建立同结构串行版 | 本轮诊断范围内完成：query16 策略的 FP32/N64 串行控制已冻结 | 先固定块划分、评分公式、精度、排序合并及输出契约；串行版对照通过；生成 IR 与设计一致 |
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


### 边界与有效计时结果：2026-11-11

报告 `be8e13d719a38ddb34fde8f091656b8d0d1dfc44`，代码 `2cb0e097fd1be8b2f5288294fa0ee0b720c8f8a3`。两组原始 chunk 和三个边界 fixture 全部通过；共33条逐块检查记录，mismatches 全部为0，最终输出检查通过。全部16次 kernel 采样第一次即有效，未重试；最大两轮 range/median 约0.465%。

单位为微秒，数值为两轮中位数：

| case | 串行控制评分 | 串行控制TopK-only | 串行整段 | 现有整合入口 | 整合入口/串行控制 |
|---|---:|---:|---:|---:|---:|
| budget_exact | 670.95 | 61.44 | 729.43 | 57.83 | 0.0793x |
| compact | 136.12 | 8.64 | 146.87 | 23.79 | 0.1620x |

冻结此提交及其分块、排序与状态算法，作为已通过的正确性参考。它不是完整CANN数值/内存结构对齐，也不是可投产优化。第二步的主要剩余问题是评分控制过慢；单行模板、query复用与发射组织和已验证快速分派不同，不能仅由此表断言是哪一个因素导致。

在评分与TopK耗时不变、忽略启动/排空与资源竞争的理想重叠模型中，(S+T)/max(S,T) 仅约1.092和1.063；它不是硬件下限或真实可实现收益。即使隐藏TopK，也无法让当前评分耗时接近现有整合整段耗时。因此下一步留在第二步，先将评分控制与已验证快速N64路径做同输入、同公式、同外层块边界的对照，保留sort2048/Top2048算法，再重新冻结可供第三步比较的控制。不得将“优于当前慢串行版”当作生产收益证明。


### 第二步评分替换候选（待设备验证）

`QSA_SERIAL_FAST_SCORE=1` 只替换串行控制的评分：从已验证的 `_qsa_page64_query16_score` 模板派生 N64 共享 K 路线，增加全局 KEY_START/SCAN_WIDTH 与局部2048输出索引；组内复用 Q/scale/weight 和 K。多于4行使用 query16，短请求使用 query4。准备阶段要求显式请求所有者，并在 CPU 上证明同一分段所有行的页表完全一致；遇到所有者或映射变化就分段，不跨请求推定共享。合成边界 fixture 传入自己的请求 ID；两个32行基准均取同一请求前32行。

此版采用的是已验证共享 K 模板，不声称完全复制当前整合入口中所有 direct-page/page-reuse 特化。score 的公式、FP32、N64、2048-key 边界不变；sort2048、Top2048、输出与缓冲区不变。query16 与 CANN S1=4 的参数差异显式保留，后续串行/双缓冲对照必须使用相同分组。

候选沿用所有逐块点积/评分/排序/状态与边界校验；另在同输入上计时原单行评分串行版 `reference_serial`。两者的比值用于隔离评分实现替换，不是双缓冲收益，也不是对 CANN V1 的验收加速比。


### 共享 K 评分替换结果：2026-11-11

报告 `e7fa0701df51356b5635b42dbf1bcd05ad033206`，代码 `f8d689ae6334aa092efc73c3847f5dfe2479077b`：两组基准和三个边界 fixture 全部通过，33条逐块检查均为0错误；20次采样首次有效，无重试，最大两轮波动约1.683%。score IR 的点积为 M64×K128×N64，符合 query16×4heads 的 INT32 计算。

单位微秒，两轮中位数：

| case | 共享K评分 | TopK | 新串行 | 同轮原单行评分串行 | 当前整合入口 | 原串行/新串行 |
|---|---:|---:|---:|---:|---:|---:|
| budget_exact | 112.58 | 59.10 | 176.34 | 732.80 | 58.09 | 4.156x |
| compact | 26.65 | 8.64 | 35.12 | 146.74 | 23.68 | 4.178x |

这是评分实现替换收益，不能称为双缓冲收益。相对当前整合入口，新串行整段只有0.329x/0.674x，尚不适合进入生产；不是对CANN V1的正式验收值。

第二步下一个受控对照：固定其余算法和公式，仅将 query16 改成 CANN S1_BASE_SIZE=4 的 query4 分组。R=32、外层2048、N64、KEY_TILES=8 时，query16 的 launch grid 为 min(20,2×4)=8；query4 为 min(20,8×4)=20。这里仅证明发射规模变化，不等同于已测得的硬件占用率或性能提升。该对照兼具 CANN 分组对齐与较小工作块并行度验证目的，完成后再冻结第三步的串行控制。

### 第二步 query4/query16 配对（待设备验证）

`QSA_SERIAL_QUERY_GROUP=4`、`QSA_SERIAL_COMPARE_GROUPS=1`，并启用已有 `QSA_SERIAL_FAST_SCORE=1`。不修改任何评分、排序、合并 kernel；仅选择既有共享评分 kernel 的 QUERY_GROUP 参数。query16 策略在请求不足4行时仍使用 query4，这是已有尾组策略；32行同请求基准的实际分组分别为4/16。

两个控制均验证所有中间状态、最终值和索引；三个边界 fixture 各跑两种策略。计时项目为 query4 score/serial、query16 peer_score/peer_serial、共用 TopK-only 和当前整合入口，共两轮交错顺序；不再重复测已经建立的单行评分慢控制。摘要独立验证所有六项配对计时有限且为正，输出两种分组的 IR。


### 第二步冻结决定：2026-11-11

报告 `95d73a016caa8d4fd91134d23e1e3dfc53669b57`，代码 `888d8505e6a8d71357ab23e9de8f5b9ce1ae6ff0`。query4/query16 两组基准和各自三组边界 fixture 全部通过，66条逐块记录均为0错误；24次计时采样有效且无重试，最大两轮波动约2.917%。IR 分别为 M16×K128×N64 与 M64×K128×N64，均为INT32累加。

单位微秒，两轮中位数：

| case | query4 score | query16 score | TopK-only（共用） | query4 serial | query16 serial | 当前整合入口 |
|---|---:|---:|---:|---:|---:|---:|
| budget_exact | 193.95 | 108.72 | 56.49 | 252.79 | 169.44 | 59.00 |
| compact | 46.77 | 25.92 | 8.67 | 60.60 | 34.80 | 23.78 |

query4 的更多 program 未带来收益；不能由 launch grid 直接推断硬件有效利用率。保留 query4 为已验证的结构对齐对照，当前不选择它作为性能控制。

冻结版本：上述代码提交，`QSA_SERIAL_FAST_SCORE=1`、`QSA_SERIAL_QUERY_GROUP=16`、`QSA_SERIAL_COMPARE_GROUPS=0`；N64、FP32、2048-key 外层块、sort2048、持续Top2048、输出Top512，短请求回落query4，按请求所有者和页表一致性分段。冻结的是当前诊断范围内的串行算法控制，不是完整CANN数值路径、全部shape验收或可投产优化。其余CANN差异仍按第一步表保留，第二次Cube与内部tile在第四步考察。

第三步的首个交付应是同一算法的混合kernel串行版，先与冻结的分离kernel串行控制核对全部中间值和最终结果；然后仅改变缓冲槽/排程，做同kernel串行与双缓冲配对。融合/减少发射的收益单独记录，不归入双缓冲收益。两种版本的query分组、评分精度、sort块、内部状态、截断和结束处理必须一致；若因资源约束调整这些参数，则两侧同时调整并重新验证，不单独改变一侧。最终仍须与当前整合入口和CANN V1做完整验收。

## 第三步首轮：同 kernel 的外层块单槽/双槽对照（待设备验证）

`QSA_SERIAL_MIXED_CONTROL=1`，同时选择 query4 并保留 query16 peer 参考。实现为 `tools/qsa_cann_mixed_910b_trial.py::_mixed_outer2048`。当前 query16 的每个 Vector 核若驻留8份 Top2048，状态需要128 KiB，加现有合并输入64 KiB及输出32 KiB已超过192 KiB，尚不含评分与排序缓冲。因此首版两种混合版本都采用 query4，每个 Vector 核驻留两份 Top2048；同轮保留匹配的 query4 分离kernel控制和最佳 query16 参考。query4 实际资源占用仍须由编译结果验证。

固定：N64、INT32 QK、FP32反量化/ReLU/加权归约、2048-key 外层块、sort2048、内部Top2048和输出Top512；首块直接初始化，后续两条等长列表合并。N64 QK逐小块产生，GM ring 存完整的 M16×2048 INT32 外层结果；每个Vector核归约两行，评分和两行Top2048状态驻留UB。交接对象仍与CANN归约后FP32分数不同，第二次Cube归约留到第四步，不宣称已完全复刻CANN。

单槽：Cube产生整个外层块→通知Vector→Vector完成评分/排序/状态合并→归还槽→Cube进入下一外层块。双槽只把SLOTS从1改成2，允许Cube产生下一外层块时Vector处理上一块；槽归还、首尾排空及尾块都计入。GM ring 本身有槽，不代表实际发生重叠；性能收益和后续设备时间线需另外证实。compact只有一个外层块，不具备跨外层块重叠机会。

gate：先验证单槽带诊断写回版本的每块点积、评分、排序与状态，再验证无诊断写回的最终索引/数值；单槽通过才进入双槽同样检查。三个边界fixture在query4策略下也运行两种混合版本。任一资源/编译/正确性失败则停止计时并保存现有IR/检查记录，不降低TopK容量来绕过问题。

计时使用同一harness kernel方法，两轮交错。`fusion_speedup=query4分离kernel串行/mixed_serial`；`double_buffer_speedup=mixed_serial/mixed_double`。评分与TopK独立数据仍来自分离kernel控制，不伪称混合kernel内部阶段独立耗时。正式生产验收仍在第五步。

### 第三步首轮编译阻塞及固定源码诊断

报告 `2abaf1969937c1223a5201931f370a3b552a2c10`，代码 `a55a89b6c0af51bbdef2c48352087e08d16239dc`：分离kernel控制的query4/query16校验通过；混合单槽的TRACE=True版本在BiSheng `ConvertLinalgIRToBinary` 阶段以 `InjectSync Traverse IR Failed` 中止。未执行该混合kernel，尚无UB超限、设备正确性、混合性能或双缓冲收益的证据。

下一轮使用 `QSA_MIXED_COMPILE_DIAG=1`，固定 `_mixed_outer2048` 源码与全部原编译选项，对 slots1/2 × TRACE False/True 做4项 warmup 编译。拦截仅保存原始编译器输入IR、原命令、stdout/stderr和错误信息，不添加优化/同步选项，不修改编译器。失败项不阻止后续编译项记录，4项均完成后统一判定。四项全编译成功才继续原来的正确性和计时流程；否则停止执行混合kernel并回传全部编译证据。独立TRITON_CACHE_DIR用于确保本轮能捕获编译输入，不改变计时器。

### 四项编译结果与首块剥离对照（待设备验证）

报告 `5476c72d866f22136e975120ca949f5c7777f30d`：slots1/2 × TRACE False/True 全部在相同BiSheng同步插入阶段失败，故不是仅诊断写回或仅双槽触发。固定编译选项下，尚无资源超限/设备正确性/性能结论。

捕获的普通单槽IR中，Vector外层循环内有返回 tensor 的 `scf.if`，分支包含首次状态初始化和后续自定义merge；此为待隔离的结构，不能直接认定是根因。下一版将首个外层块显式放到循环外，通过FIRST编译期常量选择初始化；剩余循环只做合并。保持每块评分/排序、合并次数(BLOCKS-1)、缓冲区、槽协议、参数和编译选项不变，不通过全展开增加所有块的代码体。继续使用相同四项编译矩阵；全编译成功才执行正确性和计时。需在新IR中核对该自定义操作条件分支是否消失，再评估错误是否变化。

### 首块剥离结果与编译前缀定位

报告 `2b130a912252dc8bb946ea57cc7df87cd6f983c1`：首块剥离后，捕获IR中目标 `scf.if -> tensor` 已消失，但四项依旧相同InjectSync错误。因此该条件分支并非已证实的根因，停止沿此假设修改排程。

下一轮 `QSA_MIXED_COMPILE_PREFIX=1` 固定slots1、TRACE=True，仅做4项warmup编译：stage=-1为原始INT32交接；stage=0加入FP32评分及UB分块写入；stage=1加入sort2048和状态写入（不做持续合并）；stage=2为完整算法。每项保存编译器输入IR/错误日志及成功产物。删减版本不执行、不计时、不作为正确候选；普通run始终使用默认stage=2。完整stage的计算和排程保持不变；若所有编译均通过，再按原gate只执行完整版本并验证后计时。


### 2026-11-12：原始 INT32 交接也编译失败，隔离两个访存差异

报告 `40b4b6df3e5c56a92e7f8fd3d36edbedeba436fb`，代码 `f713224e5f557460173aeb4f7127968d4daf1a35`：stage=-1/0/1/2 均报 `InjectSync Traverse IR Failed`。stage=-1 的输入IR只有359行，已无评分、UB分块写回、sort/merge自定义调用；因此这些操作不是本轮最早失败所必需的条件。错误仍可能来自Cube生产、ring访存、控制流或跨核交接，不能只凭错误名认定具体原因。无设备正确性或性能数据。

本轮沿用前述项目真相、输入契约和验收口径，仅增加算子源码结构诊断；不修改编译器、编译选项、生产src或基线。

| 项目真相补充 | 本轮冻结值 |
| --- | --- |
| 可编辑范围 | mixed trial、编译捕获、serial probe集成、本文；均为诊断工具 |
| 编译/正确性/计时 | 先原始交接2×2编译，再完整算法slots1/2×TRACE0/1编译；通过才执行原逐块/最终/边界验证和两轮交错harness kernel计时 |
| Active set | budget_exact/compact前32行和原三个边界fixture；不是完整10shape验收 |
| 汇总 | 保留分离评分、TopK、query4/query16串行、生产control、mixed单槽/双槽；fusion和double-buffer两个比值分开 |
| Autotune | 固定Ascend诊断对照，不适用NVIDIA生产autotune；本轮不修改正式调优配置 |
| 环境/风险 | pinned FlagTree/CANN/OPP不变；Mac只能静态验证，设备编译及执行待卡上验证；仍可能无完整候选编译通过 |

| 输入与数值契约补充 | 处理 |
| --- | --- |
| 页表列数 | 非空WIDTH，COLUMNS至少ceil(WIDTH/192)，页表物理页有效；索引始终clamp至COLUMNS-1 |
| 不带mask的标量页号加载 | 只改变已clamp且有效的标量读取；K/scale的WIDTH mask保留，尾块无效位置仍为0/-inf |
| ring tile-major | 同样32768个INT32/槽，将[16,2048]地址改为[32,16,64]；Cube和Vector同时改变地址公式，无数值计算变化 |
| 精度和TopK | QK INT32，评分FP32，sort2048/Top2048/输出512不变；1e-3和原索引验证不变 |

| 路径 | 行为 |
| --- | --- |
| 原始交接编译矩阵 | slots1、TRACE=True、stage=-1；原布局/仅无mask页号/仅tile-major ring/两项同时，各保存原IR和stderr |
| 完整候选筛选 | 按上述固定顺序，对原始交接编译通过的候选编译完整四项；失败继续下一个，不按性能挑选 |
| 同轮设备验证 | 首个完整四项编译通过的配置写入selection，冻结用于两种缓冲、边界与compact；所有正确性通过才计时，数值失败立即停止，不回落到其他候选 |
| 无可用配置 | 所有证据回传，执行失败，禁止运行删减版或产生性能结论 |

`QSA_MIXED_STRUCTURE_DIAG=1` 启用此流程，优先于旧编译前缀开关。普通默认参数仍为原mask页号+原ring布局。此轮两个假设来自此前已运行的混合kernel使用无mask的有效标量页号读取和连续16×64 tile交接，尚未证明它们与错误有因果关系。若两项都无效，不继续归因于TopK，保留完整编译证据再检查控制流/同步。


### 2026-11-12：标量页号 mask 隔离结果与 UB workspace 版

报告 `c4a270feb433d83d9b4c5ea394ac4bed079329a2`，代码 `968ea12875fe052497268091363266379e75d031`。

| stage=-1 结构 | 原始ring | tile-major ring |
| --- | --- | --- |
| masked scalar table load | InjectSync失败 | InjectSync失败 |
| clamped unmasked scalar table load | 编译成功 | 编译成功 |

在本轮固定结构/编译选项下，去掉已clamp的标量页号mask足以使原始交接通过编译；改ring布局不能单独解决。此证据只定位了一个可绕过的编译触发条件，尚未解释编译器内部失败原因，也没有执行原始交接删减版本。

两种ring布局的完整计时版均报UB超限：需要2318336 bits = 289792 bytes = 283 KiB，可用1572864 bits = 192 KiB，超出91 KiB。TRACE=True的完整版本均在编译阶段报 `Failed to obtain op buffer shape size which should be static`；尚未得到正确性或性能数据。仅按显式状态/输入/输出估算UB不足以预测编译器的临时缓冲和生命周期。

下一版固定 TABLE_PLAIN=True、RING_TILE_MAJOR=False，继续第三步。只改诊断实现，不改生产src/编译器/编译选项。固定资源契约：QK INT32、评分FP32、两query/Vector、sort2048、内部Top2048、输出512不变。

| 改写 | 目的与约束 |
| --- | --- |
| 两query的小循环改为static_range(2) | 将评分/状态subview的query偏移静态化；32个内部tile和外层块循环仍不展开。是否解决TRACE错误由编译结果确认 |
| 合并输入workspace显式分配一次 | 每Vector核8192个FP32 word=32 KiB，两段各4096 word存old/local两条2048项列表；两段全部覆盖，随后barrier和raw消费 |
| 不活跃列表的起点改为0 | 两个长度仍为0，active列表数量2，输入范围均有效；输出容量仍8192 word，仍提取前4096 word，状态容量不缩水 |
| scratch跨query和外层块复用 | raw结果复制回独立状态后barrier，再处理下一query；workspace不与状态/score发生别名。真实峰值以新编译器资源检查为准 |

`QSA_MIXED_WORKSPACE_DIAG=1` 固定此配置先编译完整slots1/2×TRACE0/1四项。任一失败则在同一次运行补编译评分/局部排序两个前缀并保存IR/日志，禁止删减版本执行和计时；四项全通过才继续已有中间值、最终索引及三组边界验证，两轮交错计时和报告回传。分离kernel的评分与TopK时间保留，融合收益与双缓冲收益分开。没有完整10shape结果前仍不是生产验收。


### 合并输入表示隔离：2026-11-12 下一轮

上一轮报告 `276fcb5942b2f86e736dd0d9fe78af3c05356373`（代码3374c817）：完整四项均在CommonIRToHIVM触发BufferLikeType cast断言，尚未进入完整版本的UB资源检查；评分TRACE前缀需要1671168 bits=204 KiB，排序TRACE前缀仍报静态形状错误。不能把评分前缀204 KiB与更早完整版本283 KiB直接相减。

本轮仍是第三步诊断，生产src、基线、计时器、精度和TopK容量均冻结。Autotune豁免沿用固定Ascend编译实验，不适用NVIDIA生产配置。

| 隔离项 | 固定设计与gate |
| --- | --- |
| 最小合并probe | 同一merge helper，2048+2048等长有序输入，FP32值/INT32索引交错，输出前2048项，consumed计数。MERGE_KIND=0为8192-word tensor、1为8192-word显式UB、2为原16384-word tensor |
| 表示对照 | 三种表示依次独立编译，编译失败记录后继续；编译成功才验证随机/并列/513有效值/全不可见四种fixture。检查输出值、索引范围/唯一性及输入配对一致性、consumed范围与总数。设备错误或数值失败立即终止，不吞掉后继续执行其他kernel |
| 完整候选 | 只使用MERGE_KIND=0，不自动切回较大workspace；显式UB分配仅在kind1时保留，kind0走tensor insert_slice；query循环恢复range(2) |
| 评分/排序前缀 | 单槽stage0/1各TRACE=False/True，编译采集但不执行。分开资源/诊断写回证据，不用前缀时间推算整算子收益 |
| 完整编译与执行 | 固定TABLE_PLAIN=True、原ring。slots1/2×TRACE0/1完整编译均通过且kind0最小probe正确，才执行原中间值/最终索引/边界验证，并进行原两轮交错harness kernel计时 |
| 结果解释 | miniprobe可识别某种表示本身是否触发类型断言，但通过不代表混合kernel的别名/生命周期/UB问题已解决。完整编译仍可能超限，保留资源与IR证据；本轮不降低内部Top2048容量 |

`QSA_MIXED_MERGE_DIAG=1`启用，优先于旧workspace/structure开关。一条runner命令完成全部gate及自动回传，无需用户在编译后另启计时。timing仍报告分离评分/TopK、query4/query16串行、当前生产control、mixed单槽/双槽，融合与双缓冲比值分开。该诊断不是10shape正式验收。


### 2026-11-12：完整计时版本已编译，验证路径改为无诊断写回

报告0a606fa4837261a74e782a96637c4192ce0d1a12：tensor8192和tensor16384最小合并probe的随机、并列、尾填充和全不可见四类fixture均通过；buffer8192在CommonIRToHIVM触发同类BufferLikeType断言。评分/排序及完整单槽、双槽的TRACE=False版本全部编译成功；TRACE=True评分版本UB仍需204 KiB，排序/完整版本仍报形状错误。尚未执行完整混合候选，不能宣布正确性或加速收益。

本轮只新增诊断gate及验证工具，kernel、编译选项、生产入口、基线与计时方法不再变动。QSA_MIXED_FINAL_VALIDATE=1优先于其他诊断开关。仍先执行独立合并probe和两种完整TRACE=False编译；两者通过后直接验证用于计时的同一非插桩kernel。不得将TRACE=True失败当作通过，也不得把以下验证称为内部Top2048快照验证。

| 检查范围 | 内容 |
| --- | --- |
| 输出前缀 | key扫描范围依次为min(WIDTH, 1/191/192/193/2047/2048/2049/WIDTH)，visible同步截断；与原CPU参考的相同前缀比较Top512索引、配对分数和padding |
| 首尾与交接 | 覆盖页前/页整齐/页尾、单外层块/外层块尾、完整扫描；单槽与双槽采用同样输入及数值标准 |
| 原边界fixture | 保留非均匀请求、部分query group、visible=0、分数并列及WIDTH=1/2049/4097；边界fixture仅运行自身WIDTH，不再重复8个前缀 |
| 计时 | 全部输出/边界通过后沿用两轮交错harness kernel。分离评分/TopK与query4/query16参考时间保留，单槽/双槽融合和重叠比值分开 |
| 证明限制 | 可验证输出Top512以及独立合并primitive，无法直接验证内部Top2048的每块快照或证明发生硬件重叠；不是完整10shape生产验收 |

无诊断写回路径验证失败或出现设备错误时立即停止计时并回传，不能选择其他输入表示绕过失败。Mac仅做语法及gate控制流检查，设备正确性和性能待用户运行。


### 独立复测与profiling：budget_exact 单场景

已通过报告a9125c5522f2ad44fad300e8533e4e2271486082：budget_exact混合单槽215.868 us，双槽189.733 us，单槽/双槽1.13775x；compact单槽61.377 us，双槽61.268 us，无明显收益。页与2048-key前缀、原三个边界fixture输出通过，但内部Top2048快照尚未验证。当前生产分块对照budget_exact57.215 us，候选仍不适合生产整合。

本轮kernel保持不变，QSA_MIXED_RECHECK_PROFILE=1仅选择budget_exact并明确传给summarizer。沿用完整独立合并、输出前缀和边界检查，两轮交错harness kernel计时，记录两轮各自的单槽/双槽比值。评分、TopK及query4/query16控制数据保留。

计时结束后，使用同一已验证adapter分别采集单槽、双槽CPU/NPU任务时间线（各3次调用），再分别采集PipeUtilization计数（各3次调用），禁止把profile耗时当benchmark耗时。两个arm分开profile，避免同名kernel聚合混在一起；保存已加载kernel metadata、可取得的二进制内容SHA256、压缩原始trace及硬件表。原runner自动回传顶层json/txt/json.gz文件。

任务级trace及利用率不单独证明混合kernel内部引擎重叠，报告overlap_status固定为unproven，待原始证据进一步分析；硬件采集失败/无CSV在mixed_profiling中标partial_hardware，不能把计时通过理解成全部硬件指标可用。timeline导出失败仍导致脚本失败，但保留已完成的benchmark数据。精度、TopK容量、编译器/编译选项和生产src均不变。本轮是第三步的复测/证据采集，不是全shape验收；不做跨shape算术均值。


### 第四步首轮：独立第二次Cube归约probe

独立复测0f793a41b852c233a65e5be15595334beb99724d中，budget_exact单槽221.664us、双槽195.284us，配对1.1344/1.1358x。PipeUtilization：MAC均6.128us、Vector99.864→101.810us、Scalar49.124→47.630us、MTE2 75.563→77.939us；各时间不能相加。原始trace仅MIX_AIC任务，无内部引擎区间，重叠仍未直接证明。冻结此版为第三步对照，不整合生产。

CANN快照vector::Preprocess中half weights与half qScale相乘、Cube::FixpSToL1中DEQF16+reluPre及SetFixpipePreQuantFlag(0x3a800000)，第二次Mmad的操作数为half、累加输出float。现有快照不是已证明匹配当前运行二进制的源码，也不能直接把其舍入结果当作当前FP32语义。

QSA_SECOND_CUBE_PROBE=1仅对budget_exact前32行，在原正确性/分阶段计时之后执行独立后处理实验。公共INT32 QK由CPU参考预备，GM布局[rows,4,width]，要求共享实际页表且qs/ks为正且有限。三臂：原FP32 Vector；FP32合并qs*weight、key scale后移；FP16 Cube（relu(dot)/1024转half、qs*weight构造块对角half权重、FP32矩阵乘累加后乘1024*key_scale）。4query打包、M16/K16/N64，保留尾部/visible掩码和-inf。不是复刻CANN Fixpipe指令，也不是完整评分性能对照。

转换、块对角权重构造、额外零填充计算和输出均在候选kernel内，计入harness kernel计时；公共QK的CPU计算/搬运和GPU resolved-key-scale准备在计时前完成，不计入，故不得用此速度比声称整算子收益。三臂先与原CPU评分参考按rtol/atol=1e-3逐元素比较，记录mismatch/max_abs_error/非有限输出；精度不通过则记录rejected_precision且不计时，不放宽阈值。编译或设备失败保存错误并停止。通过的臂两轮交错计时，原评分/TopK/生产控制及mixed单槽/双槽数据仍保留。生产kernel、TopK及原基线不变；无需NVIDIA autotune更新。Mac仅语法、地址分组和gate检查，卡上编译与精度待验证。


### 第四步第二轮：单kernel内QK和第二次Cube

独立后处理报告bee99add59cabd7bb37c7fb2ca7c6d579b67dad8：FP32 Vector145.794us，late-key145.822us，FP16 Cube197.869us（0.737x），均满足本fixture的1e-3阈值，Cube最大绝对误差0.024513。该结果不是完整评分或TopK精度验证。当前独立形式不整合生产。

QSA_FUSED_SECOND_CUBE_PROBE=1的下一轮固定budget_exact前32行，独立微实验替换为同一评分kernel的编译期分支。两侧同query4、N64、grid、Q/K页表及原输入尺度，公共INT8 QK在kernel内计算；Vector侧按原顺序FP32 dequant/ReLU/weight/sum，Cube侧沿用上轮受检的FP16操作数缩放及FP32累加。转换、块对角权重构造、零填充及全部输出开销计入。生产src、基线、TopK和计时器不变，无NVIDIA autotune改动。

源码仅显式写最终score，没有用户分配的GM点积workspace；不据此声称编译器消除了内部GM搬运。捕获原始编译器输入、命令、错误及完整asm文本/可用二进制SHA256，记录每种IR的matmul/dot数量供人工检查。compiler_gm_intermediate_status保持unverified，直到卡上产物证据得到分析。

两侧先验证full 32-row原WIDTH，再验证3-query、WIDTH=min(193,WIDTH)尾部fixture（第一行visible=0，第二行visible最多191）。同原CPU参考逐元素rtol/atol=1e-3，含mask/-inf及非有限输出检查；Cube不通过则rejected_precision且不计时，不放宽阈值。FP32控制不通过或编译/设备异常直接停止并保存错误。通过的臂两轮反序配对harness kernel计时。原分离评分/TopK和混合单槽/双槽控制继续保留。报告仅单次调用评分微实验，无TopK候选和全shape验收结论。Mac验证语法及query/head/尾部索引模型；卡上编译、数值和计时待用户执行。


### FP32控制的UB对齐故障隔离

报告56cafd3d273d46e94f44b50d70990fddfd0234e7（88178da5）：第一个qk_vector_fp32 full调用编译完成，但output.cpu同步时报507015，设备日志给出VEC指令访问UB地址未对齐。qk_second_cube尚未编译/执行，无两种评分kernel的正确性/计时结果，不能归因于第二次Cube。此前混合kernel计时不代表新评分probe通过。

下一轮仅将Vector分支reshape(4,4,64)+按中间轴sum改为逐query的二维4×64 extract_slice+head sum，算术顺序与FP32不变；两侧query4/N64/grid/QK输入不变。此为待验证的布局隔离，尚未通过故障PC反汇编定位根因。Cube分支不变；先控制后候选，设备异常仍立即终止。编译改用warmup，执行前保存asm、二进制hash、原始编译输入和命令，分别记录compilation/device_execution阶段，避免故障导致产物未导出。仍按原精度gate、full与tail fixture及两轮kernel计时，不改生产、TopK或编译器。


### 原始INT32 QK的grid隔离

报告bcfaf6ab72dbb90891aaf8c2fb56fd1028d0782a（29a42314）：二维Vector没有再次报UB对齐异常，但full评分90564处不匹配、max_abs95.29，tail193三query通过；secondCube full91578处错误、tail384处错误。编译输入分别有1和2次linalg.matmul，不能据此推断内部GM与交接正确。FP32控制出错，所以尚不能用此对照评价第二次Cube精度或速度。

本轮只诊断工具，生产src、TopK、编译器及其选项不变。QSA_FUSED_QK_ISOLATION=1要求同时启用FUSED_SECOND_CUBE_PROBE。_qk_reduce新增编译期DOT_ONLY（默认False）与TILE_BASE（默认0），共用原Q/K加载、页表、INT8点积及2D grid，DOT_ONLY把原始INT32点积写到[rows,4,WIDTH]，不执行qs/ks/ReLU/归约，不计时。源码有常量分支，不能把raw-only通过当成score路径内部交接也已证明正确。

固定原WIDTH和输出stride，依次group0/tile0、group0/最后tile、最后group/tile0、group0/alltiles、allgroups/tile0、full、full_repeat。行视图改变时Q/qs/weights/visible一同偏移；共享实际页表需先验证。CPU使用同一INT8 Q/K做INT32矩阵乘参考，逐元素精确比较；valid dot绝对值<=2097152，所以1515870810 sentinel不可能是合法结果。检查已选区域未写入及非目标区域被写入；记录row/head、tile分布和前16个错误值。所有产物在设备执行前导出。逻辑数值错误继续记录全部7组，设备异常立即停止；任一raw组不通过即拒绝后续归约和候选计时。

所有raw通过才在同次运行测试原FP32控制；评分错误额外保存行/tile分布及样例，FP32不通过立即拒绝第二次Cube执行。通过完整/尾部精度后才按原两轮harness kernel计时，无需用户另启步骤。原分阶段评分/TopK、当前生产分块及混合单槽/双槽对照仍保留，但不代表新候选通过。无全shape验收/生产集成结论；NVIDIA autotune不适用此Ascend诊断。Mac只做语法和索引模型验证，实际编译/正确性待卡上运行。
