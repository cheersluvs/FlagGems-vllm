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
namespace {
template <bool Chain, typename OutT>
__aicore__ inline void run_block512(__gm__ int8_t *q, __gm__ int8_t *key,
                                  __gm__ half *weight, int64_t scale_bits,
                                  __gm__ OutT *out) {
  // This private allocator is safe only in the pointer-only wrapper of this
  // feasibility probe. Do not combine it with enclosing Triton local buffers.
  TPipe pipe;
  TBuf<TPosition::A1> q_buf, weight_buf;
  TBuf<TPosition::B1> key_buf, middle_buf;
  TBuf<TPosition::A2> a_buf;
  TBuf<TPosition::B2> b_buf;
  TBuf<TPosition::CO1> acc_buf;
  pipe.InitBuffer(q_buf, M * K);
  pipe.InitBuffer(key_buf, N * K);
  pipe.InitBuffer(a_buf, M * K);
  pipe.InitBuffer(b_buf, N * K);
  pipe.InitBuffer(acc_buf, M * N * sizeof(int32_t));
  if constexpr (Chain) {
    pipe.InitBuffer(weight_buf, M * O * sizeof(half));
    pipe.InitBuffer(middle_buf, M * N * sizeof(half));
  }
  auto q_l1 = q_buf.Get<int8_t>();
  auto key_l1 = key_buf.Get<int8_t>();
  auto a = a_buf.Get<int8_t>();
  auto b = b_buf.Get<int8_t>();
  auto acc = acc_buf.Get<int32_t>();
  GlobalTensor<int8_t> q_gm, key_gm;
  q_gm.SetGlobalBuffer(q, M * K);
  DataCopy(q_l1, q_gm, M * K);
  if constexpr (Chain) {
    GlobalTensor<half> w_gm;
    w_gm.SetGlobalBuffer(weight, M * O);
    DataCopy(weight_buf.Get<half>(), w_gm, M * O);
  }
  // Q and weights stay in L1. L0A/B/C still follow the original per-tile reuse.
  for (int key_tile = 0; key_tile < 8; ++key_tile) {
    key_gm.SetGlobalBuffer(key + key_tile * N * K, N * K);
    DataCopy(key_l1, key_gm, N * K);
    fence<HardEvent::MTE2_MTE1>();
    load_matrix<int8_t>(a, q_l1, M, K, false);
    LoadData2DParams ld;
    ld.startIndex = 0;
    ld.repeatTimes = (N / 16) * (K / 32);
    ld.srcStride = 1;
    ld.dstGap = 0;
    ld.ifTranspose = false;
    LoadData(b, key_l1, ld);
    fence<HardEvent::MTE1_M>();
    MmadParams mm;
    mm.m = M;
    mm.n = N;
    mm.k = K;
    mm.cmatrixInitVal = true;
    mm.cmatrixSource = false;
    Mmad(acc, a, b, mm);
    PipeBarrier<PIPE_M>();
    fence<HardEvent::M_FIX>();
    if constexpr (!Chain) {
      store_result<int32_t>(out, acc);
    } else {
      auto middle = middle_buf.Get<half>();
      DataCopyCO12DstParams p;
      p.mSize = M;
      p.nSize = N;
      p.dstStride = M;
      p.srcStride = M;
      p.quantPre = QuantMode_t::DEQF16;
      p.reluPre = 1;
      p.channelSplit = 0;
      p.nz2ndEn = 0;
      SetFixpipePreQuantFlag(static_cast<uint64_t>(scale_bits));
      DataCopy(middle, acc, p);
      fence<HardEvent::FIX_MTE1>();
      // Reuse L0A/B/C only after the first Mmad/Fixpipe has completed.
      auto a_half = a.ReinterpretCast<half>();
      auto b_half = b.ReinterpretCast<half>();
      ld.repeatTimes = M / 16;
      ld.ifTranspose = true;
      LoadData(a_half, weight_buf.Get<half>(), ld);
      load_matrix<half>(b_half, middle, M, N, true);
      fence<HardEvent::MTE1_M>();
      mm.m = O;
      mm.k = M;
      auto acc_float = acc.ReinterpretCast<float>();
      Mmad(acc_float, a_half, b_half, mm);
      PipeBarrier<PIPE_M>();
      fence<HardEvent::M_FIX>();
      store_result<float, O>(out + key_tile * O * N, acc_float);
    }
  }
}
} // namespace

extern "C" {
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_qsa_raw_block512_address_query16(
    uint64_t q, uint64_t key, uint64_t weight, int64_t scale_bits, uint64_t out) {
  run_block512<true, float>(reinterpret_cast<__gm__ int8_t *>(q),
                           reinterpret_cast<__gm__ int8_t *>(key),
                           reinterpret_cast<__gm__ half *>(weight), scale_bits,
                           reinterpret_cast<__gm__ float *>(out));
}

__aicore__ __attribute__((always_inline)) void qsa_raw_block512_mixed_entry(
    uint64_t q, uint64_t key, uint64_t weight, int64_t scale_bits, uint64_t out)
    __asm__("_mlir_ciface_qsa_raw_block512_address_query16.cube");

__aicore__ __attribute__((always_inline)) void qsa_raw_block512_mixed_entry(
    uint64_t q, uint64_t key, uint64_t weight, int64_t scale_bits, uint64_t out) {
  _mlir_ciface_qsa_raw_block512_address_query16(q, key, weight, scale_bits, out);
}
}
