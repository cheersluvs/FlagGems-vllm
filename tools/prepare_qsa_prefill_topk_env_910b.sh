#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Execute with bash. CANN uses a separate prefix but updates container install records.
set -eo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_python=${QSA_PYTHON:-/flagos/bin/python3}
if [[ -n "${QSA_ENV_ROOT:-}" ]]; then
    qsa_root=$(realpath -- "$QSA_ENV_ROOT")
    case "$qsa_root" in
        /opt/qsa-topk-env.*) ;;
        *) echo 'QSA_ENV_ROOT must identify an existing /opt/qsa-topk-env.* directory'; exit 1 ;;
    esac
    test -d "$qsa_root/downloads"
else
    qsa_root=$(mktemp -d /opt/qsa-topk-env.XXXXXXXX)
fi
# mktemp creates mode 0700; CANN requires its root-owned ancestor to be 0755.
# Change only this dedicated environment directory, never parent trees recursively.
chmod 0755 -- "$qsa_root"
exec > >(tee -a "$qsa_root/prepare.txt") 2>&1
trap 'printf "Failed line=%s command=%s\n" "$LINENO" "$BASH_COMMAND"' ERR
trap 'qsa_status=$?; if (( qsa_status )); then printf "Preparation exit=%s; retained directory: %s\n" "$qsa_status" "$qsa_root"; fi' EXIT
printf 'Independent environment: %s\n' "$qsa_root"
"$qsa_python" - <<'PY'
import platform
import sys
assert sys.version_info[:2] == (3, 11), sys.version
assert platform.machine() == "aarch64", platform.machine()
print("Base Python:", sys.executable, sys.version)
PY
for qsa_tool in wget sha256sum; do
    if ! command -v "$qsa_tool"; then
        printf 'Missing tool: %s; install wget and coreutils before retrying.\n' "$qsa_tool"
        exit 1
    fi
done
mkdir -p "$qsa_root/downloads" "$qsa_root/cann" "$qsa_root/flagtree"
qsa_base_url='https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0'
for qsa_package in toolkit 910b-ops; do
    qsa_installer="$qsa_root/downloads/Ascend-cann-${qsa_package}_9.1.0_linux-aarch64.run"
    wget -c --header='Referer: https://www.hiascend.com/' \
        "$qsa_base_url/$(basename "$qsa_installer")" -O "$qsa_installer"
    # --quiet skips stdin prompts and accepts the vendor EULA per installer docs.
    bash "$qsa_installer" --quiet --install --install-path="$qsa_root/cann" </dev/null
    if [[ "$qsa_package" == toolkit ]]; then
        qsa_cann_env=''
        for qsa_path in "$qsa_root/cann/cann/set_env.sh" \
            "$qsa_root/cann/ascend-toolkit/set_env.sh" \
            "$qsa_root/cann/cann-9.1.0/set_env.sh" "$qsa_root/cann/set_env.sh"; do
            if [[ -f "$qsa_path" ]]; then qsa_cann_env=$qsa_path; break; fi
        done
        if [[ -z "$qsa_cann_env" ]]; then
            echo 'Cannot locate installed CANN set_env.sh; inspect installation log.'
            exit 1
        fi
        source "$qsa_cann_env"
    fi
done
qsa_wheel='flagtree-0.7.1a1+ascend3.5-cp311-cp311-linux_aarch64.whl'
qsa_wheel_url='https://resource.flagos.net/repository/flagos-pypi-hosted/packages/flagtree/0.7.1a1+ascend3.5'
wget -c "$qsa_wheel_url/$qsa_wheel" -O "$qsa_root/downloads/$qsa_wheel"
printf '%s  %s\n' \
    c57ff79b99ca2f751fcbd8db420ea1e6dd46e186c124f4e35a18a83a6cfe877b \
    "$qsa_root/downloads/$qsa_wheel" | sha256sum -c -
"$qsa_python" -m pip install --no-deps --target "$qsa_root/flagtree" \
    "$qsa_root/downloads/$qsa_wheel"
{
    printf '# Source this file to select the isolated QSA test environment.\n'
    printf 'source %q\n' "$qsa_cann_env"
    printf 'export QSA_PYTHON=%q\n' "$qsa_python"
    printf 'export QSA_FLAGTREE_PYTHONPATH=%q\n' "$qsa_root/flagtree"
    printf 'export PYTHONPATH=%q:"${PYTHONPATH:-}"\n' "$qsa_root/flagtree:$qsa_repo/src"
    printf 'export QSA_UPLOAD_REPORT=1\n'
} > "$qsa_root/activate.sh"
export QSA_ENV_ROOT="$qsa_root"
bash "$qsa_repo/tools/complete_qsa_prefill_topk_env_910b.sh"
