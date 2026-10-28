### PR Category
Operator

### Type of Change
New Feature / Performance Optimization

### Description
Add `qsa_select_paged_prefill` for packed BF16/FP8 QSA queries on NVIDIA SM90.
The direct-call API writes compressed-group TopK indices with per-row visibility
and -1 padding, without a vLLM extension dependency in production.

Includes YAML autotuning, query-row K reuse, compact ragged tiles, BF16 head
grouping and FP8 vector stores/paired dot scheduling. The existing upstream
persistent_topk algorithm is retained; only missing-TLE reporting and Triton
radix-state zero-fill are added. Other backends remain explicitly unsupported.

### Issue
No linked issue.

### Progress
- [ ] Change is properly reviewed (1 reviewer required, 2 recommended).
- [ ] Change is responded to an issue.
- [x] Targeted QSA production tests and ten FP8 stage benchmarks passed on H800.

Validated production/test/config bytes correspond to a4f8ab9. The full retained
QSA file passed 140 tests; one optional second-SM90 case was skipped. This is
not a claim that every test in the repository ran. Subsequent changes only
update this PR's documentation/evidence. Submission-gate adaptation is pending.

### Performance
NV acceptance uses score-only `--mode kernel`, excluding TopK. Against native
vLLM, ten FP8 cases yielded arithmetic means of **1.2325x score**, **0.9572x
TopK**, and **1.0578x measured pipeline**. Nine score cases exceeded 1x; compact
was 0.9945x. Long100k's TopK cost leaves its pipeline at 0.9824x. TopK was not
optimized by this change.

Both score implementations used the same FlagTree/PTXAS13.4/CUDA compatibility
profile. Native TopK used the original CUDA extension; production used TLE.
Two alternating rounds, 20ms warmup/measurement; all 90 GPU guards passed and
all captured production kernels had zero spills.

Report: https://github.com/cheersluvs/FlagGems-vllm/tree/5a9978ddd6d8774ebff511c90c449092002171b9/results/qsa_select_paged_prefill/h800/nv-pr-revalidation-20261028-025220

Per-case score/TopK/pipeline times are in
`docs/qsa_select_paged_prefill/h800-pr-validation.json`.
Environment setup uses an isolated pinned FlagGems import dependency and
loads native vLLM before FlagGems-vllm; no compiler/schema patch is required.
