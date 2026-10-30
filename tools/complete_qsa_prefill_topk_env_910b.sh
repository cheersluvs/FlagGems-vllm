#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Complete dependency checks without reinstalling CANN or FlagTree.
set -eo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_root=$(realpath -- "${QSA_ENV_ROOT:?Set QSA_ENV_ROOT to the prepared environment}")
case "$qsa_root" in
    /opt/qsa-topk-env.*) ;;
    *) echo 'Unexpected QSA_ENV_ROOT'; exit 1 ;;
esac
test -f "$qsa_root/activate.sh"
exec > >(tee -a "$qsa_root/complete.txt") 2>&1
trap 'printf "Failed line=%s command=%s\n" "$LINENO" "$BASH_COMMAND"' ERR
source "$qsa_root/activate.sh"
qsa_python=$QSA_PYTHON
mkdir -p "$qsa_root/python-deps"
printf '%s\n' \
    'cann-shmem==1.6.0 --hash=sha256:308a1c5a3dfe039d77d1159e27f3b1c16f38f4990a19c1fc68c7c052dee2b3f5' \
    > "$qsa_root/cann-shmem-requirements.txt"
"$qsa_python" -m pip install --no-deps --require-hashes \
    --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/ \
    --target "$qsa_root/python-deps" -r "$qsa_root/cann-shmem-requirements.txt"
if ! grep -q '# QSA_SHMEM_PYTHON_DEPS' "$qsa_root/activate.sh"; then
    {
        printf '# QSA_SHMEM_PYTHON_DEPS\n'
        printf 'export PYTHONPATH=%q:"${PYTHONPATH:-}"\n' "$qsa_root/python-deps"
    } >> "$qsa_root/activate.sh"
fi
source "$qsa_root/activate.sh"
export QSA_ENV_ROOT="$qsa_root"
export PYTHONPATH="$qsa_repo/src:${PYTHONPATH:-}"
"$qsa_python" - <<'PY'
import json
import os
from pathlib import Path
import torch
import shmem
import torch_npu
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.dsa.ascend.custom_ops as custom
from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import CUSTOM_OPS_BITCODE
root = Path(os.environ["QSA_ENV_ROOT"])
assert Path(triton.__file__).resolve().is_relative_to(root / "flagtree"), triton.__file__
assert Path(shmem.__file__).resolve().is_relative_to(root / "python-deps"), shmem.__file__
assert callable(tle.dsa.ascend.raw), "TLE raw callable missing"
for name in ("sort_1d_pack", "unpack_sort", "merge_exhaust_sort4"):
    assert hasattr(custom, name), name
bitcode = Path(CUSTOM_OPS_BITCODE)
assert bitcode.is_file() and bitcode.stat().st_size > 0, str(bitcode)
result = dict(
    triton=dict(version=triton.__version__, source=triton.__file__),
    torch=dict(version=torch.__version__, source=torch.__file__),
    torch_npu=dict(version=torch_npu.__version__, source=torch_npu.__file__),
    ascend_home=os.environ.get("ASCEND_HOME_PATH"),
    shmem=shmem.__file__,
    custom_ops=custom.__file__, bitcode=str(bitcode), bitcode_bytes=bitcode.stat().st_size,
    scope="Import/bitcode checks only; kernel compilation and device execution not yet verified",
)
(root / "verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
PY
printf 'Activation command: source %q\n' "$qsa_root/activate.sh"
bash "$qsa_repo/tools/run_qsa_lightning_910b.sh" \
    --suite benchmark --prefill-topk --environment-only --device "${QSA_DEVICE:-0}" --timeout 180
printf '\nNext: source %q\nbash %q --device %q\n' \
    "$qsa_root/activate.sh" "$qsa_repo/tools/run_qsa_prefill_topk_910b.sh" "${QSA_DEVICE:-0}"
