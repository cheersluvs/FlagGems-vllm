# Ascend measurement scripts — NOT part of PR #684

This branch exists only to move measurement scripts to the 910B box, which has
no SSH but can reach GitHub. **Nothing here is ever merged.** The PR branch is
`deepseek-v4-quant-insert-metax-hygon`; these files must not appear on it, and
none of them is imported by the operator, the tests or the benchmark.

Fetch a script on the box without switching branches or touching the working
tree — `git show` reads the object database, not the checkout:

```bash
git fetch origin ascend-bench-scripts
git show origin/ascend-bench-scripts:bench-scripts/run_harness_crossover.py \
    > myowncode/run_harness_crossover.py
```

They live under `bench-scripts/` rather than `myowncode/` so that a stray
checkout of this branch can never collide with the box's untracked scratch
directory of the same name.

## What each one is for

| script | question it answers |
|---|---|
| `run_harness_ascend.py` | what does the operator cost, through the repo's own harness, on a card with no baseline to divide by |
| `run_harness_ab.py` | before vs after the tuning, with the old operator bound into the empty `torch_op` slot so the harness's own SpeedUp column becomes before/after |
| `run_harness_crossover.py` | the same, densely sampled from 64 to 1024 tokens, where the benchmark's shape list has a gap |
| `ab_optimisation.py` | the same before/after through a private round-robin timing loop — an independent check on the harness numbers, not a substitute for them |
| `eager_baseline.py` | an eager torch/torch_npu composition of the operator, the only candidate baseline on this card |
| `probe_baseline_vs_oracle.py` | is that baseline right? Compares it against the test file's oracle with the operator removed from the middle, so the argument is not circular |
| `check_vs_oracle.py` | the oracle comparison on its own |

## Two things a reader should not misread

**The SpeedUp column in the A/B runs is not a speedup over vLLM.** vLLM has no
kernel on this card and its portable Triton fallback does not compile here, so
there is no external baseline. That column is `c50ad93 / HEAD`: what the tuning
bought against an already-working Ascend kernel.

**`run_harness_*.py` drive `Benchmark` directly instead of going through
pytest.** The benchmark file gates on `@pytest.mark.skipif(not
VLLM_REF_AVAILABLE, ...)`, evaluated when the decorator is applied and therefore
fixed at import. The only way past it through pytest would be to register
something under `torch.ops._C.<op>` — pointing the baseline at a fake and
inventing a speedup. Driving the class directly invents nothing.
