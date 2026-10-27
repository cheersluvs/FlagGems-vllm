# Integrate validated FP8 paired dot issue into production

## Scope board

| Item | Decision |
| --- | --- |
| Evidence | All10 actual-candidate report9be919: nine noncompact score ratios>1,compact0.9046;noncompact9 mean score1.020335/pipeline1.006456 |
| Editable | Production FP8 store-score kernel/pruner,NVIDIA FP8 config group,integration tests,production source pin and dedicated official-entry runner |
| Read-only | Original compact kernel/gate,BF16 kernels,public host API/validation,TopK,benchmark shapes and reference/tolerances |
| Historical | Preserve byte-identical pre-paired source0d7e80e2 in a named snapshot;old compile/hash-bound probes remain historical evidence |

## Implementation board

| Item | Decision |
| --- | --- |
| Kernel | Only FP8 H4/D128,TILE_R32,N64 split-query code issues right dot before left epilogue;other shapes/tiles retain original order |
| Configs | Add R32/N64/K16/P2/warps4/stages2;eligible production tuning keeps original P1 choices and validated R32/P2/warps4 choices;exclude unverified P2 shapes/warps for this domain |
| Fallback | Other H/D and dtype retain their old34 config candidates;compact untouched. Autotuning selects among candidates without shape-name dispatch |
| Cache | Existing tuner config_table_name hashes JIT source and config set;changed kernel/config group cannot reuse a previous best choice |
| Promotion scope | Source-level operator change only;no compiler edit,IR override or TopK change |

## Validation board

| Item | Decision |
| --- | --- |
| Local | Exact paired branch transfer after compile-time branch normalization,unchanged fallback/BF16/compact/host ASTs,config pruning/cache contracts and existing host tests |
| GPU | Official exported production entry,iteration correctness profile plus FP8 modes at K4 stages1/2 and K16 stage2;all10 native-vLLM/production stage benchmark |
| Timing | Same FlagTree/PTXAS13.4 on both score paths,two rounds,20ms warmup/rep;score excludes TopK;TopK and actual pipeline still reported |
| Evidence limit | Previous frozen candidate measurements do not prove production autotuner will choose the same config. Record actual tuned configs,resources,binding and per-stage speedups |
| Status | Mac cannot execute H800 validation;production-path GPU report required after publishing |

Local validation:32 production/host/accuracy-profile CPU checks passed,with
paired and original branches normalized and compared to the proven/frozen
source. Changed-file hooks,YAML,shell syntax and source pin checks pass.
The historical snapshot is for source audit;execute old hash-bound probes from
their original Git revision (e.g.b2882cc) with its corresponding YAML,not from
the new production checkout. Formal GPU correctness/performance is pending.
