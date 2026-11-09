#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
: "${QSA_MODEL:?Set QSA_MODEL to a checkpoint directory visible inside qsa-image-check}"
: "${QSA_VISIBLE_DEVICES:?Set QSA_VISIBLE_DEVICES to the devices assigned to this run}"
qsa_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
qsa_source=${QSA_OPP_CONTAINER:-wuyuqing-vllm0240}
qsa_target=${QSA_IMAGE_CONTAINER:-qsa-image-check}
qsa_opp=${QSA_OPP_SOURCE:-/opt/qsa-opp-910b-qw4qRghU}
qsa_driver=/usr/local/Ascend/driver/lib64
qsa_report=$(mktemp -d /tmp/qsa-ascend-model-forward-XXXXXXXX)
printf 'Report directory: %s\n' "$qsa_report"
qsa_tmp=
qsa_engine_args=${QSA_ENGINE_ARGS:-'{}'}
cleanup() {
    if [[ -n "$qsa_tmp" ]]; then
        docker exec "$qsa_target" rm -rf -- "$qsa_tmp" || true
    fi
}
trap cleanup EXIT

docker exec "$qsa_target" test -r "$QSA_MODEL/config.json" ||
    { echo "Checkpoint config is unreadable inside $qsa_target: $QSA_MODEL/config.json" >&2; exit 1; }
test -r "$qsa_driver/driver/libascend_hal.so" ||
    { echo "Host HAL is unreadable: $qsa_driver/driver/libascend_hal.so" >&2; exit 1; }
docker exec "$qsa_source" test -d "$qsa_opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" ||
    { echo "910B OPP is missing inside $qsa_source: $qsa_opp" >&2; exit 1; }

qsa_tmp=$(docker exec "$qsa_target" mktemp -d /tmp/qsa-model-forward-XXXXXXXX)
docker cp "$qsa_driver" "$qsa_target:$qsa_tmp/lib64"
docker cp "$qsa_tools/probe_qsa_ascend_model_forward.py" "$qsa_target:$qsa_tmp/"
docker exec "$qsa_target" mkdir -- "$qsa_tmp/opp"
docker exec "$qsa_source" tar -C "$qsa_opp" -cf - . |
    docker exec -i "$qsa_target" tar -C "$qsa_tmp/opp" -xf -

docker exec -i -e ASCEND_RT_VISIBLE_DEVICES="$QSA_VISIBLE_DEVICES" "$qsa_target" \
    /bin/bash --noprofile --norc -s -- "$qsa_tmp" "$QSA_MODEL" "$qsa_engine_args" \
    2>&1 <<'CONTAINER' | tee "$qsa_report/trace.log"
set -euo pipefail
qsa_tmp=$1
export ASCEND_OPP_PATH="$qsa_tmp/opp"
export LD_LIBRARY_PATH="$qsa_tmp/lib64/driver:$qsa_tmp/lib64/common:$qsa_tmp/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$qsa_tmp${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/local/python3.12.13/bin/python3.12 "$qsa_tmp/probe_qsa_ascend_model_forward.py" \
    --model "$2" --engine-args "$3"
CONTAINER
