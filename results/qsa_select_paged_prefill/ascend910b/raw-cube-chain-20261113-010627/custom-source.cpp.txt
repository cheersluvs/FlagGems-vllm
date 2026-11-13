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

#include "kernel_operator.h"

// Ranked memref ABI used by the CANN custom-op C interface.
template <typename T, int Rank> struct QsaMemRef {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

namespace {
using namespace AscendC;
constexpr int M = 16;
constexpr int K = 128;
constexpr int N = 64;
constexpr IsResetLoad3dConfig LOAD_CONFIG = {true, true};

template <HardEvent Event> __aicore__ inline void fence() {
  SetFlag<Event>(EVENT_ID0);
  WaitFlag<Event>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void load_matrix(LocalTensor<T> dst, LocalTensor<T> src,
                                   int rows, int columns, bool transpose) {
  LoadData3DParamsV2<T> p;
  p.l1H = rows / 16;
  p.l1W = 16;
  p.channelSize = columns;
  p.padList[0] = p.padList[1] = p.padList[2] = 0;
  p.padList[3] = 255;
  p.mExtension = rows;
  p.kExtension = columns;
  p.mStartPt = p.kStartPt = 0;
  p.strideW = p.strideH = 1;
  p.filterW = p.filterH = 1;
  p.filterSizeW = p.filterSizeH = 0;
  p.dilationFilterW = p.dilationFilterH = 1;
  p.enTranspose = transpose;
  p.fMatrixCtrl = 0;
  LoadData<T, LOAD_CONFIG>(dst, src, p);
}

template <typename T>
__aicore__ inline void store_result(__gm__ T *out, LocalTensor<T> acc) {
  GlobalTensor<T> gm;
  gm.SetGlobalBuffer(out, M * N);
  DataCopyCO12DstParams p;
  p.mSize = M;
  p.nSize = N;
  p.dstStride = N;
  p.srcStride = M;
  p.quantPre = QuantMode_t::NoQuant;
  p.nz2ndEn = true;
  p.reluPre = 0;
  p.channelSplit = 0;
  SetFixpipeNz2ndFlag(1, 0, 0);
  DataCopy(gm, acc, p);
  fence<HardEvent::FIX_M>();
  PipeBarrier<PIPE_ALL>();
}

template <bool Chain, typename OutT>
__aicore__ inline void run(__gm__ int8_t *q, __gm__ int8_t *key,
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
    pipe.InitBuffer(weight_buf, M * M * sizeof(half));
    pipe.InitBuffer(middle_buf, M * N * sizeof(half));
  }
  auto q_l1 = q_buf.Get<int8_t>();
  auto key_l1 = key_buf.Get<int8_t>();
  auto a = a_buf.Get<int8_t>();
  auto b = b_buf.Get<int8_t>();
  auto acc = acc_buf.Get<int32_t>();
  GlobalTensor<int8_t> q_gm, key_gm;
  q_gm.SetGlobalBuffer(q, M * K);
  key_gm.SetGlobalBuffer(key, N * K);
  DataCopy(q_l1, q_gm, M * K);
  DataCopy(key_l1, key_gm, N * K);
  if constexpr (Chain) {
    GlobalTensor<half> w_gm;
    w_gm.SetGlobalBuffer(weight, M * M);
    DataCopy(weight_buf.Get<half>(), w_gm, M * M);
  }
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
    mm.k = M;
    auto acc_float = acc.ReinterpretCast<float>();
    Mmad(acc_float, a_half, b_half, mm);
    PipeBarrier<PIPE_M>();
    fence<HardEvent::M_FIX>();
    store_result<float>(out, acc_float);
  }
}
} // namespace

extern "C" {
__aicore__ __attribute__((always_inline)) void _mlir_ciface_qsa_raw_qk(
    QsaMemRef<__gm__ int8_t, 1> *q, QsaMemRef<__gm__ int8_t, 1> *key,
    QsaMemRef<__gm__ half, 1> *weight, int64_t scale_bits,
    QsaMemRef<__gm__ int32_t, 1> *out) {
  run<false, int32_t>(q->aligned + q->offset, key->aligned + key->offset,
                      weight->aligned + weight->offset, scale_bits,
                      out->aligned + out->offset);
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_qsa_raw_chain(
    QsaMemRef<__gm__ int8_t, 1> *q, QsaMemRef<__gm__ int8_t, 1> *key,
    QsaMemRef<__gm__ half, 1> *weight, int64_t scale_bits,
    QsaMemRef<__gm__ float, 1> *out) {
  run<true, float>(q->aligned + q->offset, key->aligned + key->offset,
                  weight->aligned + weight->offset, scale_bits,
                  out->aligned + out->offset);
}
}
