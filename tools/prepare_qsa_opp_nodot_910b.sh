#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Reuse identical OPP files under a dot-free path for the CANN JSON-path workaround.
set -eo pipefail
qsa_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qsa_root=$(realpath -- "${QSA_ENV_ROOT:?Set QSA_ENV_ROOT to the prepared environment}")
test -f "$qsa_root/activate.sh"
source "$qsa_root/activate.sh"
qsa_source_opp="$qsa_root/cann/cann-9.1.0/opp"
test -d "$qsa_source_opp"
qsa_opp=$(mktemp -d /opt/qsa-opp-910b-XXXXXXXX)
chmod 0755 -- "$qsa_opp"
printf 'OPP source: %s\nDot-free hardlink directory: %s\n' "$qsa_source_opp" "$qsa_opp"
# Same filesystem is required. Fail on cross-device links; do not copy gigabytes silently.
cp -al -- "$qsa_source_opp/." "$qsa_opp/"
export QSA_OPP_SOURCE="$qsa_source_opp"
export QSA_OPP_PATH="$qsa_opp"
"$QSA_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
source, target = (Path(os.environ[k]) for k in ("QSA_OPP_SOURCE", "QSA_OPP_PATH"))
assert "." not in str(target.resolve()), str(target)
relative = Path("built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_transformer/lightning_indexer")
files = sorted((target / relative).glob("*"))
assert len([p for p in files if p.suffix == ".o"]) == 4, str(target / relative)
verified = []
for path in files:
    if path.suffix not in (".json", ".o"):
        continue
    old = source / path.relative_to(target)
    assert os.path.samefile(old, path), (str(old), str(path))
    assert "." not in str(path.resolve().parent), str(path)
    if path.suffix == ".json":
        metadata = json.loads(path.read_text())
        binary = path.with_suffix(".o")
        assert metadata["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest(), str(binary)
    verified.append(str(path))
print("Identical hardlinked LightningIndexer artifacts:", json.dumps(verified), flush=True)
PY
{
    printf '# CANN workaround: OPP directory contains no dot; files are unchanged hardlinks.\n'
    printf 'export QSA_OPP_PATH=%q\n' "$qsa_opp"
    printf 'export ASCEND_OPP_PATH=%q\n' "$qsa_opp"
} > "$qsa_root/opp-nodot.sh"
if ! grep -q '# QSA_OPP_NODOT' "$qsa_root/activate.sh"; then
    {
        printf '# QSA_OPP_NODOT\n'
        printf 'source %q\n' "$qsa_root/opp-nodot.sh"
    } >> "$qsa_root/activate.sh"
fi
source "$qsa_root/opp-nodot.sh"
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset QSA_OPP_TRACE
bash "$qsa_repo/tools/run_qsa_lightning_910b.sh" \
    --suite opp-diagnostic --device "${QSA_DEVICE:-0}" --timeout 180
