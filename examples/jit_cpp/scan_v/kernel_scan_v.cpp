/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <acl/acl.h>
#include <runtime/rt.h>

#include <pto/pto-inst.hpp>

#include "inter_core_flag.hpp"
#include "kernel_utils.h"

using namespace pto;

constexpr unsigned UB_SIZE = 0x30000;  // 192KB UB of A2A3


/**
 * @brief Kernel implementation for scan operation on a single cube.
 *
 * The implemetation follows the ScanMCSSA algorithm described in [1]
 *
 * [1] Parallel Scan on Ascend AI Accelerators
 * (https://arxiv.org/abs/2505.15112v1).
 *
 * @tparam InputT Input data type. Supports `fp16` or `fp32`
 * @tparam OutputT Output data type 'fp32`
 * @tparam tile_size Size of the square matrix
 *
 * @param x Input matrix in GM
 * @param o Ones matrix in GM
 * @param u Upper triangular matrix in GM
 * @param l Lower triangular matrix in GM
 * @param s Output matrix in GM, also used as intermediate buffer for C1
 */
template <typename InputT, typename OutputT, uint32_t tile_size>
AICORE void runKernelScanMCSSA(__gm__ InputT* x, __gm__ OutputT* s, __gm__ int32_t* shift_chunk, uint32_t scan_size) {
#if (__CHECK_FEATURE_AT_PRECOMPILE || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)))

  set_mask_norm();
  set_vector_mask(-1, -1);
  
  using Shape = pto::Shape<1, 1, 1, tile_size, tile_size>;
  using Stride = pto::Stride<1, 1, 1, tile_size, 1>;
  using GlobalDataIn = pto::GlobalTensor<InputT, Shape, Stride, Layout::ND>;
  using GlobalDataOut = pto::GlobalTensor<OutputT, Shape, Stride, Layout::ND>;
  using GlobalDataIndex = pto::GlobalTensor<int32_t, Shape, Stride, Layout::ND>;

  const uint32_t elePerTile = tile_size * tile_size;

  using TileDataIn = Tile<TileType::Vec, InputT, 1, elePerTile,
                          BLayout::RowMajor>;
  using TileDataOut = Tile<TileType::Vec, OutputT, 1, elePerTile,
                           BLayout::RowMajor>;

  using TileDataIndex = Tile<TileType::Vec, int32_t, 1, elePerTile,
                             BLayout::RowMajor>;

  GlobalDataIn xGlobal(x);
  GlobalDataOut sGlobal(s);
  GlobalDataIndex shiftChunkGlobal(shift_chunk);

  const uint32_t tile_ub_offset = 0x0;
  const uint32_t tile_byte_size = elePerTile * sizeof(OutputT);
  TileDataIn xVecTile;
  TileDataOut sVecTile;
  TileDataOut temp0Tile;
  TileDataOut temp1Tile;
  TileDataIndex shiftChunkTile;
  TileDataIndex tempIndexTile;
  TASSIGN(xVecTile, tile_ub_offset);
  TASSIGN(sVecTile, tile_ub_offset + tile_byte_size);
  TASSIGN(shiftChunkTile, tile_ub_offset + 2 * tile_byte_size);
  TASSIGN(temp0Tile, tile_ub_offset + 3 * tile_byte_size);
  TASSIGN(temp1Tile, tile_ub_offset + 4 * tile_byte_size);
  TASSIGN(tempIndexTile, tile_ub_offset + 5 * tile_byte_size );

  TLOAD(xVecTile, xGlobal);
  TLOAD(shiftChunkTile, shiftChunkGlobal);

  // Wait for load to complete before doing addition
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // Step 1
  set_vector_mask(0, 0x40);
  // One iteration calculates 256B of data in 8
  // blocks of 32B
  vcadd(sVecTile.data(), xVecTile.data(), 
    8, // repeat times 
    8, // dst repeat stride 
    2, // src block stride
    1, // src repeat stride 32B block?
    false // mode
  );
  // vcgadd(sVecTile.data(), xVecTile.data(), 
  //   1, // repeat times 
  //   8, // dst repeat stride 
  //   1, // src block stride
  //   1 // src repeat stride 32B block?
  // );

  pipe_barrier(PIPE_V);

  TGATHER(temp0Tile, sVecTile, shiftChunkTile, tempIndexTile);

  pipe_barrier(PIPE_V);

  // Step 2
  set_vector_mask(0, 0x20);

  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

  // One iteration calculates 256B of data in 8
  // blocks of 32B
  vcadd(temp1Tile.data(), xVecTile.data(), 
    8, // repeat times 
    8, // dst repeat stride 
    2, // src block stride
    1, // src repeat stride 32B block?
    false // mode
  );

  pipe_barrier(PIPE_V);

  TADD(sVecTile, temp0Tile, temp1Tile);
  
  // pipe_barrier(PIPE_V);
  
  // TGATHER(temp0Tile, sVecTile, shiftChunkTile, tempIndexTile);

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  TSTORE(sGlobal, temp1Tile);
  // TSTORE(sGlobal, sVecTile);
  
#endif
}

template <typename T>
AICORE void run_scan_mcssa(__gm__ T* x,
                           __gm__ float* s, 
                           __gm__ int32_t* shift_chunk,
                           uint32_t scan_size,
                           uint32_t tile_size) {
  static_assert(std::is_same_v<T, half> or std::is_same_v<T, float>,
                "scan_mcssa supports only fp16/fp32.");
  switch (tile_size) {
    case 16:
      runKernelScanMCSSA<T, float, 16>(x, s, shift_chunk, scan_size);
      break;
    case 32:
      runKernelScanMCSSA<T, float, 32>(x, s, shift_chunk, scan_size);
      break;
    case 64:
      runKernelScanMCSSA<T, float, 64>(x, s, shift_chunk, scan_size);
      break;
    case 96:
      runKernelScanMCSSA<T, float, 96>(x, s, shift_chunk, scan_size);
      break;
    case 128:
      runKernelScanMCSSA<T, float, 128>(x, s, shift_chunk, scan_size);
      break;
  }
}

// extern "C" __global__ AICORE void scan_mcssa_fp16(
//     __gm__ void* x, __gm__ void* o, __gm__ void* u, __gm__ void* l,
//     __gm__ void* s, uint32_t scan_size, uint32_t tile_size,
//     __gm__ float* scan_core_buf, __gm__ uint8_t* ffts_addr) {
//   run_scan_mcssa((__gm__ half*)x, (__gm__ half*)o, (__gm__ half*)u,
//                  (__gm__ half*)l, (__gm__ float*)s, scan_size, tile_size,
//                  scan_core_buf, ffts_addr);
// }

extern "C" __global__ AICORE void scan_mcssa_fp32(
    __gm__ float* x,
    __gm__ float* s, 
    __gm__ int32_t* shift_chunk,
    uint32_t scan_size, uint32_t tile_size) {
  run_scan_mcssa(x, s, shift_chunk, scan_size, tile_size);
}

extern "C" void scan_fp32(uint32_t blockDim, void* stream, void* x, void* s, void* shift_chunk, uint32_t scan_size,
                          uint32_t tile_size) {

  scan_mcssa_fp32<<<blockDim, nullptr, stream>>>(
      (float*)x, (float*)s, (int32_t*)shift_chunk, 
      scan_size,
      tile_size);
}
