#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
set -euo pipefail

# Container-local baseline environment. No apt install, Docker or nvcc needed.
# Installs only into a new temporary directory; host driver is not modified.
qsa_mode=${1:-all}
case "$qsa_mode" in
  setup|all) ;;
  *) echo "Usage: $0 [setup|all]" >&2; exit 2 ;;
esac
if [[ $(uname -m) != x86_64 ]]; then
  echo "This pinned setup targets Linux x86_64; a different wheel/deb is needed." >&2
  exit 2
fi
for qsa_command in curl dpkg-deb sha256sum; do
  command -v "$qsa_command" >/dev/null || {
    echo "Required command missing: $qsa_command" >&2
    exit 2
  }
done
qsa_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_base_python=${PYTHON:-python}
qsa_env_root=$(mktemp -d "${TMPDIR:-/tmp}/qsa-vllm031-h800.XXXXXX")
echo "Environment and setup log: $qsa_env_root"
exec > >(tee -a "$qsa_env_root/setup.log") 2>&1
trap 'echo "Setup/run failed at line $LINENO. Keep logs in $qsa_env_root." >&2' ERR

# Fixed package and SHA256 from NVIDIA's Ubuntu 24.04 x86_64 Packages index.
qsa_compat_name=cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb
qsa_compat_sha=d6571727935e52731a415e22df4d0f0a8717f35c9937d4c4a9fe286297dceaab
curl --fail --location --retry 2 \
  "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$qsa_compat_name" \
  --output "$qsa_env_root/$qsa_compat_name"
printf '%s  %s\n' "$qsa_compat_sha" "$qsa_env_root/$qsa_compat_name" | sha256sum --check
dpkg-deb --extract "$qsa_env_root/$qsa_compat_name" "$qsa_env_root/compat-root"
qsa_compat_dir="$qsa_env_root/compat-root/usr/local/cuda-12.9/compat"
test -e "$qsa_compat_dir/libcuda.so.1"

"$qsa_base_python" -m venv "$qsa_env_root/venv"
qsa_python="$qsa_env_root/venv/bin/python"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$qsa_compat_dir:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export VLLM_PLUGINS=""
"$qsa_python" -m pip install --index-url https://pypi.org/simple --upgrade pip uv

# Explicit CUDA backend avoids resolving PyPI's default CUDA 13 Torch build.
qsa_wheel_url='https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f'
"$qsa_env_root/venv/bin/uv" pip install --python "$qsa_python" \
  --default-index https://pypi.org/simple --torch-backend=cu129 \
  "$qsa_wheel_url" pytest packaging PyYAML sqlalchemy pybind11 scipy einops distro
"$qsa_env_root/venv/bin/uv" pip check --python "$qsa_python"
"$qsa_python" -m pip freeze > "$qsa_env_root/requirements.lock"

# A reusable activation file affects only the shell that sources it.
{
  printf 'source %q\n' "$qsa_env_root/venv/bin/activate"
  printf 'unset PYTHONPATH PYTHONHOME\n'
  printf 'export PYTHONNOUSERSITE=1\n'
  printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
  printf 'export VLLM_PLUGINS=""\n'
  printf 'export PYTHON=%q\n' "$qsa_python"
} > "$qsa_env_root/activate_qsa.sh"

# Save Python to a file so Triton can retrieve the JIT function's source.
cat > "$qsa_env_root/preflight.py" <<'PY'
from pathlib import Path
import os
import sys

import torch
import triton
import triton.language as tl
import vllm
import vllm._C_stable_libtorch  # noqa: F401
import vllm._custom_ops  # noqa: F401

assert vllm.__version__.startswith("0.31.0"), vllm.__version__
assert torch.version.cuda == "12.9", (torch.__version__, torch.version.cuda)
for module in (torch, triton, vllm):
    assert Path(module.__file__).is_relative_to(Path(sys.prefix)), module.__file__
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (9, 0)
for name in ("persistent_topk", "cooperative_topk"):
    assert hasattr(torch.ops._C, name), name

a = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
torch.testing.assert_close(a @ a, torch.full_like(a, 16))


@triton.jit
def copy_kernel(source, target, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    tl.store(target + index, tl.load(source + index))


target = torch.empty_like(a)
copy_kernel[(1,)](a, target, BLOCK=256)
torch.testing.assert_close(target, a)
torch.cuda.synchronize()
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Triton:", triton.__version__)
print("vLLM:", vllm.__version__, vllm.__file__)
print("GPU:", torch.cuda.get_device_name())
print("Compatibility libraries:", os.environ["LD_LIBRARY_PATH"])
print("CUDA extension, BF16 matmul and Triton JIT preflight: PASS")
PY
"$qsa_python" "$qsa_env_root/preflight.py" | tee "$qsa_env_root/preflight.txt"

echo "Reuse environment: source $qsa_env_root/activate_qsa.sh"
if [[ "$qsa_mode" == all ]]; then
  # The existing runner freezes the source hash and makes an unavailable
  # official reference a hard failure. No Torch substitution for vLLM.
  PYTHON="$qsa_python" bash "$qsa_repo_dir/tools/run_qsa_prefill_h800.sh" all
fi
