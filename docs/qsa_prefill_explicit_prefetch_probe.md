<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Explicit next-K prefetch feasibility probe

Pre-code boards,2026-10-26. Report52201bb7 verified that loop-scoped stages2
adds256B page-ID shared staging without moving the actual K read before dot.
Return to original production0d7e80e2;neither that hint nor dot-order candidate
is part of this probe. optimization.md/deep_opt.md are absent.

|Project truth|Frozen decision|
|---|---|
|Editable|Five new private diagnostic files: kernel,driver,tests,runner,this board|
|Read-only|Production source,YAML,TopK,original references,fixtures,timers and GPU guards;all prior probes|
|One hypothesis|Load first K tile in a prologue;carry current keys/columns/live across a loop;issue next tile load before current dot;keep original math/store;carry next values at loop end|
|Loop policy|tl.range num_stages1 prevents automatic deepening on top of explicit one-tile lookahead;all captured launch options remain unchanged;this is one manual schedule,not a stage search|
|Activation|Only original rectangular BF16 H4/D128,R32/R64,BN64;FP8/compact/other regimes use exact original JIT|
|Compile evidence|Distinguish BF16 64x128 cache reads from int32 page-ID reads;accept either async copy or direct global load only when an actual K read precedes first dot;print K buffer slots,register/shared/spills and read families|
|Resource gate|Same static WGMMA work,one identifiable main loop,at most two identified64x128 BF16 shared K slots (zero allowed only with an identified register K operand),no larger total shared footprint,known resources and zero spills;save artifacts before rejection|
|Timing|Independent R32/n1 and R64/n4 cases;only passing configurations get score/TopK/pipeline and four A/A controls;2 rounds,20ms warmup/20ms rep|
|Safety|Existing pinned FlagTree/PTXAS13.4/CUDA13.4 compat,original input/compiler/source checks,exclusive GPU/PID and thermal guards unchanged|
|Autotune|Private fixed-config feasibility probe,exempt from tuning search;production still uses original libtuner/YAML|
|Profiler|Compiler IR/PTX/resource gate first;no new NCU pass in this round|
|Delivery|Safe child-bash runner,terminal data and automatic JSON/TXT/TSV/CSV report upload|

|Contract|Preserved|
|---|---|
|Math|Original full-D FP32 dots,per-head ReLU and reduction,serial grouped dot order,and original store AST unchanged|
|Addressing|Original vector page lookup and int64 physical-page address multiply;all original strides,bounds and valid-page checks retained|
|Lookahead bounds|Next tile masked by tile+1<last_tile before page-ID and cache reads;last iteration's speculative values never contribute to output;prologue uses first tile already proven nonempty by original early return|
|Loop carry|Current columns/live/keys match each tile's original values;next values replace them only after its original dot/store|
|Fallback|Same original JIT and launch constants for inactive routes,verified by exact PTX/cubin equality|
|Torch|Only diagnostics references/fingerprints;no production compute fallback|
|Validation limit|Small all-row boundary/canary checks precede full-workload compile gate;negative compile results do not claim full-workload numeric validation or performance|

|Path|Evidence/decision|
|---|---|
|CPU semantics|AST comparison of unchanged surrounding code/math/store;execute actual prologue/lookahead/carry AST under masked pointer simulator versus original across page/visibility/tile tails,padded strides,invalid IDs and masked final lookahead|
|Compiled gate|Save unique case/PID/counter artifacts;resource/motion rejection yields not_promising/exit0,no timings;continue with the other configuration|
|Executable gate|If compiled feasibility passes,run original pre/post score/selection/input/common-logit/binary/source/GPU checks and unchanged timers|
|Failure classification|Numeric/environment/compiler/runtime failure remains nonzero;only recognized completed compile rejection is a negative probe|
|Integration|None this round;IR ordering alone is not proof of final SASS overlap or performance|

Delivery validation:54 applicable CPU checks passed,including10 new checks.
The actual prologue/lookahead/carry/load AST was executed against the original
under a masked pointer simulator across516 cases(page16/64/196,tile/tail ranges,
visible boundaries,padded table/cache strides,invalid page IDs and final masked
lookahead). Consumed columns/live/keys and every effective table/cache read agree;
the next read is issued before current consumption. Unchanged prefix,full math
and store ASTs compare exactly. Source-level correctness does not guarantee GPU
lowering correctness; original on-card numerical checks remain mandatory.
The parser was also calibrated on verified52201bb baseline/candidate artifacts:
both contain one actual K loop read and zero before-dot K reads;only the hint
candidate has a separate metadata async copy. Register-K operand fixtures and
shared-buffer bounds are tested. Pre-commit,bash syntax and CLI help passed.
Production/YAML remain unchanged. GPU compile/resource/validation/timing are pending.
