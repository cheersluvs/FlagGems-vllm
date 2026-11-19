#!/usr/bin/env bash
set -uo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
report=$(mktemp -d /tmp/qsa-metax-installed.XXXXXXXX) || exit 1
cd /tmp || exit 1
echo "Report: $report"
if command -v mx-smi >/dev/null; then mx-smi > "$report/mx-smi.txt" 2>&1; fi
selected=
candidates=("${VIRTUAL_ENV:-/nonexistent}/bin/python" "${CONDA_PREFIX:-/nonexistent}/bin/python" python3 /opt/conda/bin/python /flagos/bin/python /opt/conda/envs/*/bin/python)
if [[ -n ${QSA_PYTHON:-} ]]; then candidates=("$QSA_PYTHON"); fi
for candidate in "${candidates[@]}"; do
    interpreter=$(command -v "$candidate") || continue
    [[ -x $interpreter ]] || continue
    [[ -n $selected ]] || selected=$interpreter
    if "$interpreter" -c 'import importlib.util as u; import sys; sys.exit(u.find_spec("vllm_metax") is None)' 2>/dev/null; then
        selected=$interpreter
        break
    fi
done
[[ -n $selected ]] || { echo "No Python found; set QSA_PYTHON."; exit 2; }
export VLLM_PLUGINS=${VLLM_PLUGINS:-metax}
"$selected" -u - <<'PY' 2>&1 | tee "$report/environment.txt"
import base64
import hashlib
import importlib.metadata as md
import importlib.util
import subprocess
import sys
from pathlib import Path

print("Python", sys.executable, sys.version)
for name in ("flaggems-vllm", "flagtree", "triton", "torch", "vllm", "vllm-metax"):
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        print("DIST MISSING", name)
        continue
    print("DIST", name, dist.version, dist.locate_file(""))
    if name in ("flagtree", "triton"):
        spec = importlib.util.find_spec("triton")
        if spec and spec.origin:
            active = Path(spec.origin).parent / "compiler" / "compiler.py"
            for entry in dist.files or ():
                if entry.hash and dist.locate_file(entry).resolve() == active.resolve() and active.is_file():
                    actual = hashlib.new(entry.hash.mode, active.read_bytes()).digest()
                    match = base64.urlsafe_b64encode(actual).decode().rstrip("=") == entry.hash.value
                    print("COMPILER RECORD", name, "match=", match, active)
for name in ("torch", "triton", "flaggems_vllm", "triton.experimental.tle", "vllm", "vllm_metax", "vllm_metax.models.qwen4_exp.metax.ops.qsa_indexer"):
    code = "import importlib; m=importlib.import_module(%r); print('IMPORT OK', m.__name__, getattr(m,'__file__',None), getattr(m,'__version__',None))" % name
    if name == "flaggems_vllm":
        code += "; print('vendor', getattr(m,'vendor_name',None), 'QSA export', callable(getattr(m,'qsa_select_paged_prefill',None)))"
    try:
        proc = subprocess.run([sys.executable, "-c", code], timeout=120)
        if proc.returncode:
            print("IMPORT FAILED", name, proc.returncode)
    except subprocess.TimeoutExpired:
        print("IMPORT TIMEOUT", name)
PY
env_rc=${PIPESTATUS[0]}
"$selected" -u "$repo/tools/probe_qsa_metax_native.py" --suite installed \
    --report "$report" --device "${QSA_DEVICE:-0}" 2>&1 | tee "$report/console.txt"
status=${PIPESTATUS[0]}
(( env_rc == 0 )) || status=$env_rc
echo "$status" > "$report/exit-code.txt"
git -C "$repo" rev-parse HEAD > "$report/code-commit.txt" || exit 1
stamp=$(date -u +%Y%m%d-%H%M%S)-$$
branch=codex/qsa-metax-results-$stamp
tree=$(mktemp -d /tmp/qsa-metax-upload.XXXXXXXX) || exit 1
git -C "$repo" worktree add -b "$branch" "$tree" HEAD || exit 1
dest=results/qsa_select_paged_prefill/metax/installed-$stamp
mkdir -p "$tree/$dest" && cp -R "$report/." "$tree/$dest/" && \
    git -C "$tree" add -- "$dest" && git -C "$tree" -c user.name='QSA GPU reports' \
    -c user.email=qsa-reports@users.noreply.github.com commit -m "Report installed MetaX environment and QSA baseline" || exit 1
echo "Local report: $report; result worktree: $tree; test exit: $status"
git -C "$tree" push https://github.com/cheersluvs/FlagGems-vllm.git "HEAD:refs/heads/$branch" || exit 1
echo "结果分支：$branch"
git -C "$tree" rev-parse HEAD
exit "$status"
