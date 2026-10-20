# H800 QSA 正式生产入口整合验证结果

[原始报告](https://github.com/cheersluvs/FlagGems-vllm/blob/708f30eba6278b18cdc7b7020cfabe59e1682965/results/qsa_select_paged_prefill/h800/flagtree134-ab-20261020-094731/stage-summary.json)；测试代码 `cd7195cfcd2aedac70b6879f8e81ecd31e240829`，报告提交 `708f30eba6278b18cdc7b7020cfabe59e1682965`。

结论：compact 调度、key-major 评分和八配置调优已通过本轮正式生产入口功能验证，保留这三项整合，并以本轮结果作为后续生产优化基线。
209 passed、1 skipped（第二张SM90卡设备守卫测试）；22组benchmark全部完成。NV验收口径为不含TopK的score/kernel；实际性能边界见下表，并非所有场景都快于原生vLLM。

## 分组几何平均

| 场景组 | 评分/vLLM（NV指标） | TopK/vLLM | pipeline/vLLM |
|---|---:|---:|---:|
| 上游12组 | 1.05553× | 0.89554× | 0.96716× |
| 新增8评分/TopK、10整体 | 2.00593× | 0.89676× | 1.93485× |

加速比=本轮原生vLLM中位延迟/正式生产中位延迟，>1表示我方更快。
上游评分10/12组>1，最小0.98206×；新增评分7/8组>1，最小0.98664×。上游两组较慢的是n1/q8192/p51200（0.98206×）和n1/q16384/p102400（0.99212×）；新增chunk_tail64为0.98664×。
TopK保持仓库persistent_topk/TLE，与原生CUDA扩展比较；本轮未修改或优化TopK。上游TopK几何平均0.89554×，因此评分平均更快仍不代表整体更快；上游pipeline为0.96716×。
本轮没有旧生产或私有候选的性能配对阶段，不能用历史报告的加速比相除来估算整合增量或证明零回退。正确性中包含12个正式生产与保留八配置候选的全入口比较。

## 正式入口与数据完整性审计

58份报告文件通过Git blob SHA-1/长度校验，48份源码快照的SHA256与manifest和实际测试代码一致；测试仓库干净，HEAD为cd7195c。
预检、正确性和全部22个worker的runtime.operator_binding一致：candidate_is_canonical_module=true、exported_entry_is_candidate=true，函数模块为flaggems_vllm.ops.qsa_select_paged_prefill。
生产源文件为/tmp/flaggems-qsa-production.q8f9EL/src/flaggems_vllm/ops/qsa_select_paged_prefill.py，SHA256=6db2918b5b7dc1798244548972458a8c59745f967341a10663e2d704cf35687a。性能被测侧直接使用正式导出入口，不由私有候选factory构造，也未替换TopK；功能测试另构造保留候选作为参考。
所有进程使用同一GPU UUID、实际FlagTree模块/编译器库、ptxas13.4.59、CUDA13.4兼容libcuda和原生TopK扩展。62个计时阶段、372个原始计时样本、248次计时遥测及所有汇总统计均复核通过。
两侧TopK读取相同冻结的原生logits，前后内容哈希一致；pipeline独立计时评分→各自TopK，未将分阶段中位数相加。两组选全只计pipeline，score/TopK为N/A。

## 功能覆盖与调度证据

| 新增正式入口检查 | 参数化通过数 |
|---|---:|
| test_integrated_public_entry_identity | 1 |
| test_integrated_observed_against_native | 10 |
| test_integrated_rectangular_every_config | 8 |
| test_integrated_compact_every_config | 6 |
| test_integrated_full_entry_and_retained_candidate | 12 |

原有正式测试的dtype/layout/alias/unsupported/原生直接对齐断言继续保留。每配置数值检查覆盖BF16/FP8、H/D参数化、int64/strided元数据、非零base、空请求、零visible和跨chunk边界，使用独立FP32参考与原有容差。
19个评分场景走正式矩形key-major，8个上游case实际选中了新增的TR64/BN64/warps8配置。ragged54的offset0走compact，其余场景不启用该路径；两组选全无评分launch。
所有正式评分PTX的fractional createpolicy/evict_first/cp.async L2 hint计数为0，未把被否决的hint加回生产。ncu/nsys不可用，资源数据是静态元信息，不用于宣称动态带宽、L2命中率或occupancy。

## 评分完整明细（μs）

| Case | vLLM | 正式生产 | 评分加速比 | 三轮配对范围 |
|---|---:|---:|---:|---:|
| perf_n1_q8192_p8192_bfloat16 | 144.064 | 126.208 | 1.14148× | 1.13921–1.17368 |
| perf_n1_q8192_p51200_bfloat16 | 516.800 | 526.240 | 0.98206× | 0.98146–0.98912 |
| perf_n1_q8192_p102400_bfloat16 | 1047.584 | 948.640 | 1.10430× | 1.07395–1.10430 |
| perf_n1_q16384_p8192_bfloat16 | 340.656 | 322.000 | 1.05794× | 1.03737–1.06511 |
| perf_n1_q16384_p51200_bfloat16 | 1183.712 | 1079.680 | 1.09635× | 1.07282–1.10385 |
| perf_n1_q16384_p102400_bfloat16 | 2025.328 | 2041.408 | 0.99212× | 0.98908–1.02559 |
| perf_n2_q8192_p8192_bfloat16 | 258.720 | 250.224 | 1.03395× | 0.98960–1.04406 |
| perf_n2_q8192_p51200_bfloat16 | 1082.848 | 1028.096 | 1.05326× | 1.05184–1.05653 |
| perf_n2_q8192_p102400_bfloat16 | 1958.240 | 1949.648 | 1.00441× | 1.00372–1.04311 |
| perf_n4_q4096_p8192_bfloat16 | 230.208 | 215.520 | 1.06815× | 1.01865–1.06958 |
| perf_n4_q4096_p51200_bfloat16 | 1063.968 | 986.272 | 1.07878× | 1.07832–1.10623 |
| perf_n4_q4096_p102400_bfloat16 | 1885.568 | 1770.016 | 1.06528× | 1.06422–1.06787 |
| observed_short96_bfloat16 | 51.744 | 8.320 | 6.21923× | 6.20000–6.41667 |
| observed_tail16k_bfloat16 | 53.568 | 16.288 | 3.28880× | 3.26130–3.33400 |
| observed_tail32k_bfloat16 | 69.472 | 26.432 | 2.62833× | 2.55990–2.63471 |
| observed_short_batch3_bfloat16 | 52.416 | 12.448 | 4.21080× | 4.18909–4.27676 |
| observed_ragged54_bfloat16 | 78.400 | 73.856 | 1.06153× | 1.04208–1.06542 |
| observed_unequal2_bfloat16 | 601.952 | 596.992 | 1.00831× | 1.00599–1.01014 |
| observed_chunk_boundary_bfloat16 | 491.584 | 448.304 | 1.09654× | 1.08921–1.11127 |
| observed_chunk_tail64_bfloat16 | 628.384 | 636.896 | 0.98664× | 0.98426–0.98817 |
| observed_select_all_single_bfloat16 | N/A | N/A | N/A | N/A |
| observed_select_all_batch18_bfloat16 | N/A | N/A | N/A | N/A |

## TopK和pipeline完整明细（μs）

| Case | vLLM TopK | 正式生产 TopK | TopK加速比 | vLLM pipeline | 正式生产 pipeline | pipeline加速比 |
|---|---:|---:|---:|---:|---:|---:|
| perf_n1_q8192_p8192_bfloat16 | 206.080 | 214.528 | 0.96062× | 343.984 | 337.232 | 1.02002× |
| perf_n1_q8192_p51200_bfloat16 | 384.320 | 422.544 | 0.90954× | 892.672 | 920.000 | 0.97030× |
| perf_n1_q8192_p102400_bfloat16 | 571.504 | 708.832 | 0.80626× | 1557.808 | 1655.056 | 0.94124× |
| perf_n1_q16384_p8192_bfloat16 | 453.024 | 454.720 | 0.99627× | 778.752 | 763.232 | 1.02033× |
| perf_n1_q16384_p51200_bfloat16 | 785.920 | 885.008 | 0.88804× | 1903.584 | 1993.664 | 0.95482× |
| perf_n1_q16384_p102400_bfloat16 | 1162.240 | 1477.440 | 0.78666× | 3145.760 | 3497.968 | 0.89931× |
| perf_n2_q8192_p8192_bfloat16 | 398.880 | 414.080 | 0.96329× | 645.136 | 670.784 | 0.96176× |
| perf_n2_q8192_p51200_bfloat16 | 763.296 | 836.432 | 0.91256× | 1772.992 | 1800.528 | 0.98471× |
| perf_n2_q8192_p102400_bfloat16 | 1136.960 | 1398.240 | 0.81314× | 3055.728 | 3320.160 | 0.92036× |
| perf_n4_q4096_p8192_bfloat16 | 390.752 | 387.200 | 1.00917× | 607.920 | 605.584 | 1.00386× |
| perf_n4_q4096_p51200_bfloat16 | 753.152 | 814.560 | 0.92461× | 1753.360 | 1750.896 | 1.00141× |
| perf_n4_q4096_p102400_bfloat16 | 1128.864 | 1387.232 | 0.81375× | 2969.664 | 3170.432 | 0.93667× |
| observed_short96_bfloat16 | 8.288 | 11.488 | 0.72145× | 56.512 | 16.096 | 3.51093× |
| observed_tail16k_bfloat16 | 16.928 | 18.752 | 0.90273× | 65.408 | 29.472 | 2.21933× |
| observed_tail32k_bfloat16 | 21.088 | 25.472 | 0.82789× | 84.704 | 45.472 | 1.86277× |
| observed_short_batch3_bfloat16 | 13.888 | 13.696 | 1.01402× | 61.856 | 22.368 | 2.76538× |
| observed_ragged54_bfloat16 | 181.472 | 161.312 | 1.12498× | 249.408 | 228.416 | 1.09190× |
| observed_unequal2_bfloat16 | 524.000 | 635.360 | 0.82473× | 1097.184 | 1187.984 | 0.92357× |
| observed_chunk_boundary_bfloat16 | 522.944 | 537.728 | 0.97251× | 992.656 | 961.664 | 1.03223× |
| observed_chunk_tail64_bfloat16 | 536.512 | 632.832 | 0.84780× | 1163.424 | 1266.112 | 0.91890× |
| observed_select_all_single_bfloat16 | N/A | N/A | N/A | 24.384 | 6.144 | 3.96875× |
| observed_select_all_batch18_bfloat16 | N/A | N/A | N/A | 106.400 | 22.048 | 4.82583× |

## 实际配置和资源

| Case | 路径 | TR/BN/KT/warps | 寄存器/线程 | shared字节 | spill_count |
|---|---|---|---:|---:|---:|
| perf_n1_q8192_p8192_bfloat16 | key-major | 32/64/4/4 | 118 | 73728 | 0 |
| perf_n1_q8192_p51200_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n1_q8192_p102400_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n1_q16384_p8192_bfloat16 | key-major | 64/64/16/4 | 232 | 106496 | 0 |
| perf_n1_q16384_p51200_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n1_q16384_p102400_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n2_q8192_p8192_bfloat16 | key-major | 64/64/4/8 | 118 | 114688 | 0 |
| perf_n2_q8192_p51200_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n2_q8192_p102400_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n4_q4096_p8192_bfloat16 | key-major | 32/64/4/4 | 118 | 73728 | 0 |
| perf_n4_q4096_p51200_bfloat16 | key-major | 64/64/16/8 | 118 | 114688 | 0 |
| perf_n4_q4096_p102400_bfloat16 | key-major | 64/64/16/4 | 232 | 106496 | 0 |
| observed_short96_bfloat16 | key-major | 4/64/1/4 | 64 | 37888 | 0 |
| observed_tail16k_bfloat16 | key-major | 16/64/4/4 | 83 | 53248 | 0 |
| observed_tail32k_bfloat16 | key-major | 16/64/4/4 | 115 | 51200 | 0 |
| observed_short_batch3_bfloat16 | key-major | 8/64/4/4 | 66 | 43008 | 0 |
| observed_ragged54_bfloat16 | compact | 32/64/4/4 | 113 | 73728 | 0 |
| observed_unequal2_bfloat16 | key-major | 64/64/16/4 | 255 | 100352 | 44 |
| observed_chunk_boundary_bfloat16 | key-major | 64/64/16/4 | 232 | 106496 | 0 |
| observed_chunk_tail64_bfloat16 | key-major | 64/64/16/4 | 255 | 100352 | 44 |
| observed_select_all_single_bfloat16 | select-all | N/A | N/A | N/A | N/A |
| observed_select_all_batch18_bfloat16 | select-all | N/A | N/A | N/A | N/A |

## 环境和耗时

GPU UUID GPU-83ccf064-1bbb-6765-e356-c2a85323cbee；FlagTree实际模块3.6.0（/opt/flagtree），ptxas13.4.59、CUDA13.4兼容libcuda；Torch2.13.0+cu129、vLLM0.31.0+cu129。
3轮、warmup200ms、rep100ms，初始温度门槛80C；24次准入通过，另有一次无进程的滞后利用率10%等待10秒后通过，没有温度等待。计时温度31–46C，SM时钟1980–1980MHz，遥测中未见外来PID或实际热降频。
完整正确性556.62秒（约9分17秒）；首末benchmark遥测跨度819.07秒，包含case间编译、启动、验证和计时，不代表脚本完整墙钟耗时。
GPU本轮为GPU6，与上一轮候选GPU2不同；不混用两张卡或两个时段的绝对延迟构造加速比。本轮所有比值均为同卡、同进程、同工具链内的配对计时。

## 决策

三项整合保留，正式生产入口现成为下一轮优化基线。本轮仅记录审计和结果，不继续改动生产、TopK、调优空间或22组shape。H800结果不自动证明其他后端性能。
