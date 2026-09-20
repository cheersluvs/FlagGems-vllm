# BW1000 current prefill codegen audit

On the dedicated Hygon worktree, pull the probe commit, then run:

```bash
tools/hygon_prefill_next_run.sh codegen hygon_prefill_codegen_v1
```

The runner commits and pushes the report. Four representative production
shapes are compiled and launched once each in separate workers, using the
current non-TLE kernel and production geometry. The report records target
assembly availability, registers, spills, shared-memory bytes, barriers,
atomics, loads/stores, conversion opcodes, and representative instruction
lines. It does not change the operator, measure speed, or prove a hardware
bottleneck from static instruction counts alone.

Decision: a spill, unexpected scalar memory operation, duplicated conversion,
or excessive barrier sequence must be supported by both the codegen evidence
and an existing phase-cost result before changing layout/VEC/ISA. If none is
visible, move on rather than doing a blind parameter sweep.
