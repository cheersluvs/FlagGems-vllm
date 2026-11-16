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

static_assert(QSA_QUERY_PACK == 16);
#define QSA_ABI_STRING_IMPL(name) #name
#define QSA_ABI_STRING(name) QSA_ABI_STRING_IMPL(name)

extern "C" {
// The mixed caller requests the Cube suffix. Both entries belong to this one
// translation unit, so no duplicate CANN header globals need to be merged.
__aicore__ __attribute__((always_inline)) void qsa_raw_mixed_chain_entry(
    uint64_t q, uint64_t key, uint64_t weight, int64_t scale_bits, uint64_t out)
    __asm__(QSA_ABI_STRING(QSA_SYMBOL(_mlir_ciface_qsa_raw_chain_address)) ".cube");

__aicore__ __attribute__((always_inline)) void qsa_raw_mixed_chain_entry(
    uint64_t q, uint64_t key, uint64_t weight, int64_t scale_bits, uint64_t out) {
  QSA_SYMBOL(_mlir_ciface_qsa_raw_chain_address)(q, key, weight, scale_bits, out);
}
}
