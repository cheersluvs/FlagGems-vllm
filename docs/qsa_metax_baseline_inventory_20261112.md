# MetaX QSA baseline and artifact inventory

Public source inspected on 2026-11-12: MetaX-MACA/vLLM-metax master commit
ca15eaa7373d8c60a59478fbebcc959ceb1d9c9d. Latest GitHub Release queried was
v0.28.0 (2026-10-15); its tree has no QSA/qwen4_exp paths. An unpublished
vendor wheel/image may differ from either source; inspect its contents first.

Source: https://github.com/MetaX-MACA/vLLM-metax/blob/ca15eaa7373d8c60a59478fbebcc959ceb1d9c9d/vllm_metax/models/qwen4_exp/metax/ops/qsa_indexer.py
Caller: https://github.com/MetaX-MACA/vLLM-metax/blob/ca15eaa7373d8c60a59478fbebcc959ceb1d9c9d/vllm_metax/models/qwen4_exp/metax/indexer_qsa.py

| Contract | Public development implementation |
| --- | --- |
| Entry | qsa_select_paged_prefill(q, k_cache, page_table, query_start_loc, visible_blocks, token_topk, compress_ratio, max_query_len, block_indices, max_seq_len) |
| Q/cache | Matching dtype; caller selects BF16 or float8_e4m3fn; FP8 uses RMSNormed Q/K without explicit per-token scales in this API |
| Shapes | Q [tokens,H,D], cache [pages,page_size,1,D], request page table and packed-query offsets |
| Output | INT32 compressed logical block indices [tokens, token_topk/compress_ratio] |
| Scoring | Triton _qsa_mqa_paged_prefill_kernel, visible key tiles and per-request query ranges |
| TopK | _C.cooperative_topk or _C.persistent_topk according to runtime predicates; do not assume which route runs on MACA |
| Chunking | Environment logits budget, width bounded by max_seq_len and page-table capacity |
| Expansion | Separate expand_qsa_block_indices; outside selection entry |
| Unknown | Installed vendor version, native extension ABI, hardware FP8 availability and actual runtime paths |

The Ascend INT8 prepared trial is not the MetaX baseline. Start from the matching
BF16/FP8 QSA contract after verifying actual vendor artifacts. No MetaX operator
implementation or performance claim is delivered by this inventory step.

## Remote inspection

The Mac cannot SSH to the user's box. Run tools/run_qsa_metax_vendor_inventory.sh
on the box in the namespace where /data/zhouxt is visible. It uses Python stdlib,
no GPU framework imports, package installation, image import/start, or extraction.
Inspect wheel METADATA and QSA source members without executing them. List local
Docker images and query mx-smi if present. Read Docker-save manifests and bounded
uncompressed layers, including nested vLLM wheels and QSA sources. Raw layers are
not a merged filesystem: deletions/whiteouts are not applied, so layer content is
not proof the package remains installed in the final image.

Defaults: depth6, 50000 files, 80 inspected artifacts, 120-second inventory budget,
20-second artifact-worker timeout, at most16 recent layers per archive, 512KiB
source/metadata entries. Large compressed archives are deferred to avoid expensive
multi-GB decompression. All limitations/errors are recorded. Negative results are
bounded to the scan, not a machine-wide absence claim. ZIP archives are names-only.
A standalone wheel is not labelled image-derived without archive provenance.

Reports include current Python distribution versions, package origin paths,
QSA function definitions/mentions, source hashes and bounded source text. Finding
a file does not prove importability, native TopK symbols, device correctness or
performance. Those are the next phase after identifying the relevant artifact.
The runner uploads reports to the user's fork even if the root is unavailable.
