# Production FP8 full10 CUDA Graph benchmark

## Scope board

| Item | Decision |
| --- | --- |
| Request | Benchmark current production QSA on the existing10 FP8 shapes using CUDA Graph mode |
| Editable | Shared diagnostic timer mode,stage CLI/report labels,CPU tests and dedicated runner |
| Read-only | Production SHA2560d7e80e240966565a7c78151c8344f17e0627e73920f5229aadcc0f0fb42f10e,FP8 shape definitions,compiler,TopK,tuning and tolerances |
| Isolation | Separate checkout/branch;leave unrelated untracked candidate work in the original checkout untouched |

## Implementation board

| Item | Decision |
| --- | --- |
| Timer | Existing benchmark.base.Benchmark.get_latency CUDAGRAPH branch calls triton.testing.do_bench_cudagraph(rep=...,return_mode=median);preserve ms->us conversion |
| Coverage | FlagTree,production candidate,fp8-prefill suite;existing native-vLLM baseline uses the same timer/compiler/toolchain |
| Stages | Score-only,TopK-only and actual sequential pipeline;production persistent_topk unchanged |
| Defaults | Kernel mode unchanged for old callers;CUDA Graph mode explicitly selected and recorded |
| Warmup/cache | do_bench_cudagraph uses internal warmup/replay calibration;Config.warm_up is not passed;no do_bench L2-flush policy claim |

## Validation board

| Item | Decision |
| --- | --- |
| Local | Exercise actual shared get_latency branch with mocked graph timer;check forwarding,units,Config restoration,mode validation,CLI and source binding |
| Card | Native/production preflight and timed-stage pre/post numerical checks;formal production entry binding,source/input fingerprints and GPU guards retained |
| Runtime | Two alternating rounds,rep20ms,graph timer default retry count unchanged. Skip separate pytest batch to save time;retain stage numerical checks |
| Reporting | Each case prints CUDA Graph score,TopK,pipeline latencies/speedups;store timer mode and timer source hash;auto-upload success/failure |
| Acceptance | Separate CUDA Graph comparison;existing kernel-mode performance acceptance remains unchanged and cannot be satisfied by graph results |

The Mac has no H800 access. Local checks do not establish CUDA Graph capture
compatibility of the installed native extension or TLE implementation;the card
run must pass capture and the normal post-timing checks. Do not silently fall
back to kernel timing if capture fails.

Local validation:31 CPU checks pass,including actual shared timer branch
selection,mode propagation to preflight/worker,no kernel fallback after capture
failure,10-case CLI selection and readable graph labels. Updated the stale
legacy default-suite assertion from upstream12 to the existing FP8 default10.
Shell/hooks pass. Production source SHA256 remains0d7e80e2;no GPU run on Mac.
