# Bypass empty TopK history for a single key block

| Experiment | Contract |
| --- | --- |
| Scope | Prepared Ascend 910B mixed kernel in the probe harness |
| Control | Current fixed-sort snapshot fallback at WIDTH=512 |
| Candidate | Unpack the one sorted 512-proposal list inside its query iteration |
| Fixed work | Query16, block512 Cube, two ring slots, ready/free events, score, sort and output masking |
| Timing | Two interleaved rounds, three invocations per arm per round, per-window task-count validation |
| Metric | Prepared control kernel latency divided by prepared candidate kernel latency |

| Input and output | Validation |
| --- | --- |
| Shapes | 16, 17 and 128 queries with one 512-key block |
| Visibility | Zero, partial and full visibility plus padded query rows |
| Selection | FP32 Top512 scores and INT32 compressed-group indices |
| Padding | Invalid ranks retain -1 index and -inf score. Padded rows cannot write output. |
| Correctness | Existing independent score reference, tie-aware rank checks, exact scores and output guards before and after timing |

| Path | Evidence |
| --- | --- |
| Snapshot | Existing online merge, temporary and persistent history |
| Single-block | One sort32, two local mrgsort calls and one unpack_sort per query in IR. No exhaustion merge, fallback gather or persistent history allocation. |
| Scope of performance result | Device latency for a prepared kernel with no packing, dispatch or full-input CANN comparison |

Run with `--single-block` in the existing history probe. The candidate specializes
to logical WIDTH=512. Generalizing to an input with a wider logical width but
one scheduled block requires its own validation with the runtime block count.
