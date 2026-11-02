# 910B QSA restricted production routing

Status: implementation and local host checks complete; device validation pending.
The NVIDIA implementation and its draft PR are unchanged.

The production Triton kernel AST is unchanged. Only host dispatch changes:

| Policy | Gate | Otherwise |
| --- | --- | --- |
| Score tile128 | 910B, contiguous BF16 Q/K, H4/D128, page192, contiguous int32 table, width4096..25600 | tile32 |
| Chunk cap32 | Score tile128 gate, group TopK512, contiguous int32 request offsets, width4096..16384 | cap16 |

The existing logits budget still bounds the actual chunk, with a minimum of one
row even for a zero budget. Request-owner mapping, scoring arithmetic, output
contract, and TopK implementation are unchanged. The width intervals are a
restricted dispatch choice based on the sampled widths, not proof that every
width in those intervals has a measured performance benefit.

Use `run_qsa_lightning_910b.sh --suite benchmark --chunk32-production`.
It verifies the registered public entry and compares old tile32/cap16 policy
against actual production dispatch in the same process. The old control patches
only the two policy helpers; it does not change TopK or replace the public host.

The 16 cases include row counts31/32/33/63/64/65, ragged and empty requests,
zero/1MiB budgets, short/medium/long diagnostics, width4096/8192/16384 timings,
and H1/page256 fallback checks. Six cases have timing, ten only correctness.
CPU FP32 score checks and selected-score/range/uniqueness checks run outside
timing. TopK data, native adapter totals and actual launch routing are reported.
This is exploratory paired timing, not NVIDIA kernel-mode acceptance.

Local checks: original kernel AST fingerprint, 45 actual-host allocation/chunk
simulations, width/layout/dtype/device policy guards, formatting and syntax.
These checks do not establish NPU compilation, correctness or performance.
