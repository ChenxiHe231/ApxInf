#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace apxinf::experimental::parallel_moe_layout {

constexpr int kExperts = 256;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPerLane = 4;
constexpr int kSegment = kThreads * kPerLane;
constexpr int kTile = 128;

inline int blocks_for(int routes) {
  return (routes + kSegment - 1) / kSegment;
}

// Warp-local stable ranks and block counts adapted from the L1/058 submission.
// Use kernel boundaries
// between the global passes so replay does not need a resident-grid barrier.
__global__ void count_and_rank(const int32_t* ids, int routes, int max_padded,
                               int32_t* block_counts, int32_t* ranks,
                               int32_t* permutation) {
  __shared__ int32_t warp_counts[kWarps][kExperts];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  for (int w = 0; w < kWarps; ++w) warp_counts[w][threadIdx.x] = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_padded;
       i += gridDim.x * blockDim.x) {
    permutation[i] = -1;
  }
  __syncthreads();

  int row_base = blockIdx.x * kSegment + warp * (32 * kPerLane) + lane;
  int expert[kPerLane];
  int rank[kPerLane];
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    int row = row_base + j * 32;
    int id = row < routes ? ids[row] : -1;
    expert[j] = id;
    unsigned valid = __ballot_sync(0xffffffff, id >= 0 && id < kExperts);
    if (id >= 0 && id < kExperts) {
      unsigned peers = __match_any_sync(valid, id);
      unsigned lower = peers & ((1u << lane) - 1);
      int previous = warp_counts[warp][id];
      rank[j] = previous + __popc(lower);
      __syncwarp(valid);
      if (!lower) warp_counts[warp][id] = previous + __popc(peers);
    }
    // The valid-lane set can change in the next batch. All lanes must
    // observe the previous leaders' writes, including newly valid lanes.
    __syncwarp();
  }
  __syncthreads();

  int total = 0;
  for (int w = 0; w < kWarps; ++w) total += warp_counts[w][threadIdx.x];
  block_counts[blockIdx.x * kExperts + threadIdx.x] = total;
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    int id = expert[j];
    if (id >= 0 && id < kExperts) {
      int prior = 0;
      for (int w = 0; w < warp; ++w) prior += warp_counts[w][id];
      ranks[row_base + j * 32] = prior + rank[j];
    }
  }
}

__global__ void prefix_and_tiles(int blocks, int32_t* block_counts,
                                 int32_t* offsets, int32_t* tile_groups,
                                 int32_t* tile_limits, int32_t* num_tiles) {
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ typename Scan::TempStorage scratch;
  int expert = threadIdx.x;
  int count = 0;
  for (int block = 0; block < blocks; ++block) {
    int32_t& entry = block_counts[block * kExperts + expert];
    int local = entry;
    entry = count;
    count += local;
  }
  int tile_count = (count + kTile - 1) / kTile;
  int first_tile, all_tiles;
  Scan(scratch).ExclusiveSum(tile_count, first_tile, all_tiles);
  offsets[expert] = first_tile * kTile;
  if (expert == 0) *num_tiles = all_tiles;
  for (int tile = 0; tile < tile_count; ++tile) {
    tile_groups[first_tile + tile] = expert;
    tile_limits[first_tile + tile] = first_tile * kTile + min((tile + 1) * kTile, count);
  }
}

__global__ void scatter(const int32_t* ids, const int32_t* ranks,
                        const int32_t* block_prefix, const int32_t* offsets,
                        int routes, int32_t* permutation) {
  int row_base = blockIdx.x * kSegment + threadIdx.x;
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    int row = row_base + j * kThreads;
    if (row < routes) {
      int id = ids[row];
      if (id >= 0 && id < kExperts) {
        int position = offsets[id] + block_prefix[blockIdx.x * kExperts + id] + ranks[row];
        permutation[position] = row;
      }
    }
  }
}

inline void launch(const int32_t* ids, int routes, int max_padded,
                   int32_t* block_counts, int32_t* ranks, int32_t* offsets,
                   int32_t* permutation, int32_t* tile_groups,
                   int32_t* tile_limits, int32_t* num_tiles, cudaStream_t stream) {
  int blocks = blocks_for(routes);
  count_and_rank<<<blocks, kThreads, 0, stream>>>(
      ids, routes, max_padded, block_counts, ranks, permutation);
  prefix_and_tiles<<<1, kThreads, 0, stream>>>(
      blocks, block_counts, offsets, tile_groups, tile_limits, num_tiles);
  scatter<<<blocks, kThreads, 0, stream>>>(
      ids, ranks, block_counts, offsets, routes, permutation);
}
}  // namespace apxinf::experimental::parallel_moe_layout
