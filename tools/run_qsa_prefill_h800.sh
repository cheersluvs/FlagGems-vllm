#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
set -euo pipefail

# Run from any directory. PYTHON may select an existing card virtualenv.
# No packages are installed and no repository files are checked out here.
qsa_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$qsa_repo_dir"
qsa_python=${PYTHON:-python}
qsa_mode=${1:-all}
case "$qsa_mode" in
  quick|full|bench|all|torch-only) ;;
  *) echo "Usage: $0 [quick|full|bench|all|torch-only]" >&2; exit 2 ;;
esac

export PYTHONPATH="$qsa_repo_dir/src:$qsa_repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DNN_VENDOR=nvidia
# Keep the official baseline free from plugin operator replacement.
export VLLM_PLUGINS=""
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-512}

qsa_report_dir=$(mktemp -d "${TMPDIR:-/tmp}/qsa-prefill-h800.XXXXXX")
echo "Reports: $qsa_report_dir"
git rev-parse HEAD | tee "$qsa_report_dir/commit.txt"
git status --short -- src/flaggems_vllm/ops/qsa_select_paged_prefill.py \
  src/flaggems_vllm/ops/__init__.py src/flaggems_vllm/__init__.py \
  src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml \
  tests/qsa_select_prefill_utils.py tests/test_qsa_select_paged_prefill.py \
  benchmark/test_qsa_select_paged_prefill_perf.py

"$qsa_python" - <<'PY' | tee "$qsa_report_dir/environment.txt"
import importlib
import importlib.metadata
import os
from pathlib import Path

import torch
import triton
import flaggems_vllm

op = importlib.import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
expected = Path.cwd() / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
assert Path(op.__file__).resolve() == expected.resolve(), op.__file__
assert flaggems_vllm.vendor_name == "nvidia", flaggems_vllm.vendor_name
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_capability() == (9, 0), "SM90 device required"
assert flaggems_vllm.qsa_select_paged_prefill is op.qsa_select_paged_prefill
print("device:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("triton:", triton.__version__)
try:
    print("vllm:", importlib.metadata.version("vllm"))
except importlib.metadata.PackageNotFoundError:
    print("vllm: not installed")
print("source:", op.__file__)
print("logits budget MiB:", os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"])
print("score tune candidates:", len(flaggems_vllm.runtime.get_tuned_config("qsa_select_paged_prefill_score")))
PY

if [[ "$qsa_mode" != torch-only ]]; then
  export QSA_REQUIRE_VLLM_REFERENCE=1
  qsa_ref_commit=7a877ae34dcf4a7d46de08eb16db4699e1e0047c
  qsa_ref_file="$qsa_report_dir/qsa_indexer_vllm.py"
  if [[ -n "${QSA_VLLM_REFERENCE_FILE:-}" ]]; then
    cp -- "$QSA_VLLM_REFERENCE_FILE" "$qsa_ref_file"
  else
    curl --fail --location --retry 2 \
      "https://raw.githubusercontent.com/vllm-project/vllm/$qsa_ref_commit/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py" \
      --output "$qsa_ref_file"
  fi
  export QSA_VLLM_REFERENCE_FILE="$qsa_ref_file"
  "$qsa_python" - <<'PY' | tee "$qsa_report_dir/reference.txt"
from tests.qsa_select_prefill_utils import REFERENCE_COMMIT, REFERENCE_SHA256, vllm_reference
reference = vllm_reference()
print("reference commit:", REFERENCE_COMMIT)
print("reference SHA256:", REFERENCE_SHA256)
print("reference source:", reference.__file__)
PY
else
  export QSA_REQUIRE_VLLM_REFERENCE=0
fi

if [[ "$qsa_mode" == quick || "$qsa_mode" == all || "$qsa_mode" == torch-only ]]; then
  "$qsa_python" -m pytest -q -ra tests/test_qsa_select_paged_prefill.py --quick \
    --record json --output "$qsa_report_dir/accuracy-quick.json" \
    2>&1 | tee "$qsa_report_dir/accuracy-quick.txt"
fi
if [[ "$qsa_mode" == full || "$qsa_mode" == all ]]; then
  "$qsa_python" -m pytest -q -ra tests/test_qsa_select_paged_prefill.py \
    --record json --output "$qsa_report_dir/accuracy-full.json" \
    2>&1 | tee "$qsa_report_dir/accuracy-full.txt"
fi
if [[ "$qsa_mode" == bench || "$qsa_mode" == all ]]; then
  "$qsa_python" -m pytest -q -s benchmark/test_qsa_select_paged_prefill_perf.py \
    --level core --mode operator --warmup 100 --iter 500 \
    --record json --output "$qsa_report_dir/benchmark-core.json" \
    2>&1 | tee "$qsa_report_dir/benchmark-core.txt"
fi
echo "Finished. Return the text/JSON files in: $qsa_report_dir"
