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
- [ ] Change is fully covered by a UT.

Preparation only: latest-upstream-base GPU validation and a compatible
FlagGems-vllm submission gate are pending. Do not mark historical results as
validation of this branch.

### Performance
Ten FP8 envelopes; NV acceptance uses score-only `--mode kernel`, excluding
TopK. Historical H800 source 16bc82f passed 49 iteration-profile tests and all
ten stage benchmarks: arithmetic mean score 1.2344x, TopK 0.9569x and measured
pipeline 1.0556x against native vLLM. The long100k pipeline remained 0.9838x.
These figures precede the PR's upstream TopK update and require revalidation.
See docs/qsa_select_paged_prefill/h800-historical-results.json for per-case
stage timings and docs/qsa_select_paged_prefill/README.md for scope/limitations.
