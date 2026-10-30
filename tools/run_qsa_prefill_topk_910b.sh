#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Execute with bash, never source into the container login shell.
set -euo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export QSA_PYTHON=${QSA_PYTHON:-/flagos/bin/python3}
export QSA_UPLOAD_REPORT=${QSA_UPLOAD_REPORT:-1}
export PYTHONPATH="$qsa_repo/src:/opt/flagtree${PYTHONPATH:+:$PYTHONPATH}"
printf 'Candidate: repository Ascend top_k_per_row_prefill; score unchanged\n'
bash "$qsa_repo/tools/run_qsa_lightning_910b.sh" \
    --suite production --prefill-topk --timeout 600 "$@"
bash "$qsa_repo/tools/run_qsa_lightning_910b.sh" \
    --suite benchmark --prefill-topk --timeout 600 \
    --rounds 2 --warmup 2 --iterations 3 "$@"
