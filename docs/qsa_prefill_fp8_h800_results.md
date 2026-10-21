# H800 QSA 新增10组FP8验证结果

[原始报告](https://github.com/cheersluvs/FlagGems-vllm/blob/cd132fad331f065721b54dc17432b7ca5caa0226/results/qsa_select_paged_prefill/h800/flagtree134-ab-20261020-111303/stage-summary.json)；[原始计时表](https://github.com/cheersluvs/FlagGems-vllm/blob/cd132fad331f065721b54dc17432b7ca5caa0226/results/qsa_select_paged_prefill/h800/flagtree134-ab-20261020-111303/triton-stage-ab.tsv)。
测试代码272e49992eb2501031e01f838bf9d15bcadff24a，报告提交cd132fad331f065721b54dc17432b7ca5caa0226。

219 passed、1 skipped；新增10个FP8正式入口对齐测试全部passed。跳过项为需要第二张SM90卡的设备守卫测试。仅新增10组FP8性能测试在本轮运行，旧22组BF16性能没有重跑。
本轮属于新增覆盖和基线测量，没有修改生产算子。FP8评分几何平均0.98231×，未体现整体快于原生vLLM；不能将其解释为本轮改坏了生产或用历史BF16加速比推算dtype间收益。

## 三阶段汇总

| 阶段 | 组数 | 几何平均加速比 | 快于原生组数 |
|---|---:|---:|---:|
| 评分kernel（NV验收，不含TopK） | 10 | 0.98231× | 1/10 |
| TopK | 10 | 0.91918× | 2/10 |
| 评分→TopK顺序执行 | 10 | 0.94879× | 2/10 |

加速比=同轮原生vLLM中位延迟/正式生产中位延迟，>1表示我方更快。几何平均按用例等权，不按捕获count加权，也不与旧BF16合并。pipeline单独计时，不由分阶段中位数相加。
TopK仍是仓库persistent_topk/TLE对原生vLLM CUDA扩展，双方读取相同冻结的原生FP32 logits；此轮未替换或优化TopK。

## 每组评分（μs）

| Case（省略fp8_及_float8_e4m3fn） | vLLM | 正式生产 | 加速比 | 三轮配对加速比范围 |
|---|---:|---:|---:|---:|
| first_chunk | 114.816 | 117.504 | 0.97712× | 0.97321–0.98006 |
| budget_exact | 290.272 | 313.760 | 0.92514× | 0.90858–0.92575 |
| dominant_long | 338.896 | 343.568 | 0.98640× | 0.98371–1.00123 |
| tail64 | 385.152 | 390.656 | 0.98591× | 0.98317–0.98694 |
| tail65 | 384.224 | 384.704 | 0.99875× | 0.97814–1.00599 |
| two_balanced | 395.264 | 418.912 | 0.94355× | 0.93953–0.94408 |
| two_skewed | 382.032 | 393.856 | 0.96998× | 0.95237–0.97355 |
| short_tail | 74.848 | 67.968 | 1.10122× | 1.07667–1.11111 |
| compact | 53.504 | 53.856 | 0.99346× | 0.97924–1.00360 |
| long100k | 1325.056 | 1392.544 | 0.95154× | 0.94321–0.96509 |

## 每组TopK和pipeline（μs）

| Case | vLLM TopK | 我方TopK | TopK加速比 | vLLM pipeline | 我方pipeline | pipeline加速比 |
|---|---:|---:|---:|---:|---:|---:|
| first_chunk | 351.584 | 345.088 | 1.01882× | 460.832 | 459.792 | 1.00226× |
| budget_exact | 525.376 | 538.368 | 0.97587× | 813.376 | 855.408 | 0.95086× |
| dominant_long | 540.352 | 562.080 | 0.96134× | 873.296 | 901.440 | 0.96878× |
| tail64 | 538.528 | 629.312 | 0.85574× | 920.096 | 1018.560 | 0.90333× |
| tail65 | 534.592 | 629.408 | 0.84936× | 900.992 | 987.264 | 0.91262× |
| two_balanced | 545.888 | 591.568 | 0.92278× | 939.232 | 1010.464 | 0.92951× |
| two_skewed | 535.088 | 629.952 | 0.84941× | 918.976 | 1027.328 | 0.89453× |
| short_tail | 92.992 | 103.744 | 0.89636× | 164.320 | 166.848 | 0.98485× |
| compact | 181.200 | 161.536 | 1.12173× | 223.616 | 207.648 | 1.07690× |
| long100k | 1162.064 | 1477.872 | 0.78631× | 2521.648 | 2863.008 | 0.88077× |

## 输入和调度核对

Q/cache实际指纹dtype均为torch.float8_e4m3fn，H=4、D=128、page=196、block_k=512、logits budget=512MiB。所有case的shape/stride/storage offset、实际chunk序列与冻结workload几何一致；logits实际为FP32。
8组来自上传日志记录的六元组，2组是compact和long100k的显式dtype变体。实际indexer dtype、cache/table布局和逐请求上下文未被原始日志完整记录，仍使用已声明的构造假设。64/65尾块由紧凑page-table与512MiB预算推导，并不是原始推理chunk trace。
9组走正式key-major矩形路径；compact组实际命中compact调度。下面是预热后已编译配置快照，不宣称其为每次chunk launch的逐条追踪。所有配置spill_count=0，无fractional createpolicy/evict_first/cp.async L2 hint。ncu/nsys不可用，静态资源不足以推断动态带宽、L2命中或occupancy。

| Case | 路径 | TR/BN/KT/warps | 寄存器/线程 | shared字节 |
|---|---|---|---:|---:|
| first_chunk | key-major | 32/64/4/4 | 123 | 40960 |
| budget_exact | key-major | 32/64/4/4 | 123 | 40960 |
| dominant_long | key-major | 32/64/4/4 | 123 | 40960 |
| tail64 | key-major | 16/64/4/4 | 109 | 26624 |
| tail65 | key-major | 16/64/4/4 | 109 | 26624 |
| two_balanced | key-major | 64/64/16/4 | 226 | 57344 |
| two_skewed | key-major | 16/64/4/4 | 109 | 26624 |
| short_tail | key-major | 32/64/4/4 | 123 | 40960 |
| compact | compact | 32/64/4/4 | 114 | 40960 |
| long100k | key-major | 64/64/16/4 | 226 | 57344 |

## 环境、完整性和验收口径

34份报告文件已通过Git blob SHA-1和长度校验。51份源码快照均通过SHA256校验，与manifest及本地实际测试提交一致；测试仓库干净。
预检、完整正确性和10个worker均使用正式导出flaggems_vllm.qsa_select_paged_prefill；canonical/candidate绑定一致。生产SHA256=6db2918b5b7dc1798244548972458a8c59745f967341a10663e2d704cf35687a，与之前BF16生产验证相同。
原生参考仍为固定vLLM提交7a877ae34dcf4a7d46de08eb16db4699e1e0047c及固定源码SHA256。双方评分实际使用/opt/flagtree/triton（模块3.6.0）、同一编译器库、ptxas13.4.59及CUDA13.4兼容libcuda；Torch2.13.0+cu129、vLLM0.31.0+cu129。安装包元数据中的Triton3.7.1不是本轮实际加载模块。
单卡GPU2：GPU-2d17872f-4327-2beb-8049-604ecf6a3d66。shared_gpu_exploratory=false；12次准入全部通过，未触发温度或利用率等待。3轮、预热200ms、重复100ms、初始温度上限80℃。
30个计时阶段、180个原始计时样本、120个遥测点及全部中位数/配对比/几何平均已复算。TopK公共logits前后哈希一致，计时前后数值验证均通过。计时遥测温度32–44℃，SM时钟1980MHz；进程均在各worker许可的自身PID中，未记录外来进程，所有性能守卫通过。
完整正确性563.39秒（9分23秒）；首末benchmark遥测跨度346.01秒，包含case间编译、启动、验证与计时，不能当作纯kernel累计时间或脚本完整墙钟耗时。
本轮两侧评分处于同一工具链，可作为NV score/kernel口径的FP8基线；“报告通过/acceptance_eligible”表示测量有效，不代表加速比大于1。H800结果也不代表其他平台。

## 后续建议

保留这10组作为FP8基线，后续优先研究评分路径：budget_exact为0.92514×、two_balanced为0.94355×、long100k为0.95154×。仅short_tail获得1.10122×；tail65和compact接近持平，不能仅凭3轮小差异解释微观原因。
后续评分优化仍须回归原有22组BF16，并分别汇总上游、观测BF16和FP8结果。TopK继续单列，不改动算法；本轮仅记录结果，无生产修改。
