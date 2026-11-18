# Seed the TopK history from the first sorted block

| Experiment | Contract |
| --- | --- |
| Scope | Prepared Ascend 910B mixed kernel with two or three key blocks |
| Control | Existing fixed local sort and snapshot update for every block |
| Candidate | Peel the first Vector block and commit its sorted proposals directly to each query's history slot |
| Fixed work | Query16, block512 Cube, two ring slots, ready/free events, scoring, sorting, later snapshot merges and final unpack |
| Shapes | 17×769 and 80×1024 for two blocks, 17×1281 and 112×1536 for three blocks |
| Timing | Two interleaved rounds, three invocations per arm per round and task-count validation |

Each query writes all 512 packed proposals during the first block, including
invisible rows and keys. A lane with an invalid score remains negative infinity.
The final rank and row masks retain the existing padding semantics. The same
slot barriers separate the first write from subsequent history reads.

The IR check expects two static sorting sites, one snapshot merge site in the
rest of the loop, and one final unpack. The device checks retain independent
score and rank oracles and guard validation before and after timing. This is
a kernel probe with prepared inputs. The two- and three-block results support
a later full-input dispatch decision with packing and per-piece costs included.

Run with `--first-block-seed` under the existing 910B history probe wrapper.
