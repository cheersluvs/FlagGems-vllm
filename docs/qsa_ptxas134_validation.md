# H800：只更换 ptxas 的 QSA 验证

检验原生 QSA 的 FlagTree 非法指令问题是否能仅靠新版 ptxas 解决。
这是评分正确性诊断，不是 benchmark，也不是完整 vLLM 兼容性验收。
正式 benchmark 仍仅有上游12组，本脚本不改变其 shape。

## 固定条件

- 原生 QSA 源码提交 `7a877ae34dcf4a7d46de08eb16db4699e1e0047c`，SHA256固定。
- 复用已有评分诊断的两组 prefill：H4/D128、H1/D32。
- 复用已有 decode 复现：H4/D128、query length=3、BF16/FP8、tiles=1/2/4/8，共8组。
- 每组5个独立子进程，prepare另用独立进程，失败的CUDA上下文不会传到下一项。
- 同一组各项共享持久化输入、独立FP32评分参考、原生kernel参数和grid。
- 不修改 FlagTree/vLLM/生产算子源码，不安装Python包，不更新驱动或CUDA运行库。

## 五项对照

1. `stock`：原生kernel由虚拟环境内的官方Triton编译并运行，正对照。
2. `flagtree-old`：现有FlagTree与默认旧ptxas；先保存PTX/CUBIN/ABI，再运行。
   预期复现非法指令，此项失败本身不导致最终失败。
3. `identical-ptx-new`：原PTX逐字节不变，用ptxas 13.4.59重新组装，在FlagTree下运行。
   不删除policy、不更改fraction、不更改`.version`、保留原assembler选项。
4. `new-cubin-stock`：同一份新CUBIN在官方Triton下加载运行，交叉排查launcher因素。
5. `fresh-jit-new`：现有FlagTree不改源码，仅配置`TRITON_PTXAS_PATH`指向新工具，
   使用新缓存重新编译原生kernel并执行，并确认仍生成fractional policy和async L2 hint。
   PTX版本由现有编译器正常推导，不手改PTX头。

两层结论独立保存。第3/4项通过但第5项失败，只能说明固定PTX的组装问题已解决，
不能声称现有FlagTree可直接配置运行。旧故障未复现或未生成policy，结论为inconclusive。
所有10组均为`config_only_validated`时退出0；未解决或证据不足退出1；GPU环境拒绝退出2。

## 工具与报告

从[NVIDIA CUDA 13.4.1 redistributables](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json)
下载Linux x86_64 nvcc 13.4.59包及nvdisasm 13.4.49包，校验固定SHA256，
仅提取ptxas和nvdisasm两个普通文件至独立临时目录，不覆盖系统工具。
PTX/机器码/原始命令保存在每项`*-artifacts.json`压缩包；新nvdisasm输出保留全部SASS，
不根据简单正则自动断言描述符初始化正确。报告记录编译器源文件与libtriton哈希、
实际assembler路径/版本、输入/原生源/PTX/CUBIN哈希和失败阶段。

入口：`bash tools/run_qsa_ptxas_h800.sh --flagtree /opt/flagtree`。
选用原vLLM 0.31.0虚拟环境Python，通过`QSA_STOCK_PYTHON`指定；
`CUDA_VISIBLE_DEVICES`可指定单个完整GPU UUID；未设置或为`auto`时自动选择空闲H800。
自动选择不构成独占预留；脚本在每个子进程前仍检查GPU闲置，
发现其他任务则停止，不结束任何其他进程。

runner会自动回传已有报告，包括失败报告。回传路径为
`results/qsa_select_paged_prefill/h800/ptxas-<UTC时间>/`，结果分支仍使用
`codex/qsa-h800-results-<UTC时间>`。原始输入`.pt`、缓存和工具保留在卡上，不上传。

## 2026-10-19：CUBIN加载崩溃的最小对照

报告 `c27bd3c5aa085d8f30a362d7fe384fb2850ee8cf` 完成10组：官方Triton/旧ptxas
全部评分通过，FlagTree/旧ptxas全部复现执行阶段非法指令。新工具重组装的原PTX
字节哈希保持不变，FlagTree全新JIT生成PTX 9.4；但全部新CUBIN在加载/launcher
初始化阶段收到SIGILL或SIGSEGV，未进入kernel执行，因此尚不能确认policy修复。
原CUBIN的ELF OSABI=0x33/ABI version=7，新产物为0x41/8，反汇编也提示Std ELF。
环境为570驱动、LD_LIBRARY_PATH配置CUDA12.9兼容库；加载兼容性是待验证假设。

下一步：`QSA_PTXAS_LOADER_ONLY=1 bash tools/run_qsa_ptxas_h800.sh`。
该模式只运行两个不含policy、异步拷贝和矩阵计算的最小kernel：空返回、写入123。
同一PTX 8.7分别交给旧ptxas与13.4.59，得到4份CUBIN。分别在独立进程中直接调用
`cuModuleLoadData`、`cuModuleGetFunction`、`cuLaunchKernel`、`cuCtxSynchronize`，
使用Torch建立CUDA上下文，但不使用Triton launcher或QSA kernel。
记录实际libcuda映射路径、cuDriverGetVersion、ELF头、API边界、退出信号和faulthandler栈。
工具、驱动、FlagTree安装保持原样；报告沿用自动回传路径。新最小CUBIN失败说明问题
也能在无policy条件下复现；最小CUBIN成功则需要继续检查QSA资源/元数据等加载差异。

## 报告84cab23：无policy的直接加载故障已复现

两组最小PTX的旧、新汇编器输入哈希各自一致。旧产物2/2通过，新产物2/2在
`cuModuleLoadData`内SIGSEGV，未进入kernel。实际映射为旧虚拟环境内的
`cuda-12.9/compat/libcuda.so.575.57.08`，`cuDriverGetVersion=12090`。
因此这一加载故障不需要QSA、fractional policy或Triton launcher即可复现；尚不能
据此判断新ptxas在匹配运行库下是否修复了原policy问题。

[NVIDIA forward-compat矩阵](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
列明570驱动可配合cuda-compat-13-4。下一轮设置：

```bash
export QSA_PTXAS_LOADER_ONLY=1
export QSA_CUDA_COMPAT134=1
bash tools/run_qsa_ptxas_h800.sh
```

runner通过`with_cuda_compat134.py`下载固定的Ubuntu24.04 amd64兼容包615.71.09，
校验NVIDIA仓库公布的SHA256后，仅用`dpkg-deb --extract`解包到临时目录。
启动全新Python进程，仅对本次诊断的子进程将该目录前置到LD_LIBRARY_PATH。
不执行包安装、post-install、ldconfig或内核驱动更改，不修改原虚拟环境文件。
报告增加`compat134.json`，记录包和libcuda哈希、目录及原/新库路径；loader还会
强制检查实际libcuda路径和Driver API版本，避免配置未生效却误判通过。
库准备失败也由原runner自动回传。先验证4个最小对照，通过后再运行10组原生QSA。
