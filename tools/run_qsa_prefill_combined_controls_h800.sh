#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Child shell only; existing runner uploads complete and partial reports.
set -euo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$qsa_repo/tools/run_qsa_prefill_flagtree_ab_h800.sh" \
  --candidate query-joint-page --accuracy-profile iteration \
  --suite scored-prefill --mode kernel --rounds 2 \
  --combined-self-control-cycles 4 --kernel-warmup-ms 200 --kernel-rep-ms 100 \
  --max-initial-temp-c 80 --timeout 1800 --accuracy-timeout 7200 \
  --cooldown-timeout 900 --cooldown-interval 10 \
  --cases observed_short_batch3_bfloat16 "$@"
