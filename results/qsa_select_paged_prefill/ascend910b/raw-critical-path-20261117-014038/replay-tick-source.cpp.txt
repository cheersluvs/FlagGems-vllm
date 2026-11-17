// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "qsa_raw_cube_query_pack_910b.cpp"

extern "C" {
__aicore__ __attribute__((always_inline)) void _mlir_ciface_qsa_raw_replay_tick(
    uint64_t q, uint64_t key, uint64_t weight, int64_t step, uint64_t out) {
  (void)q;
  (void)key;
  (void)weight;
  using namespace AscendC;
  GlobalTensor<int32_t> marker;
  marker.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(out), 1);
  marker.SetValue(0, static_cast<int32_t>(step));
  PipeBarrier<PIPE_ALL>();
}
__aicore__ __attribute__((always_inline)) void qsa_replay_tick_mixed(
    uint64_t q, uint64_t key, uint64_t weight, int64_t step, uint64_t out)
    __asm__("_mlir_ciface_qsa_raw_replay_tick.cube");
__aicore__ __attribute__((always_inline)) void qsa_replay_tick_mixed(
    uint64_t q, uint64_t key, uint64_t weight, int64_t step, uint64_t out) {
  _mlir_ciface_qsa_raw_replay_tick(q, key, weight, step, out);
}
}
