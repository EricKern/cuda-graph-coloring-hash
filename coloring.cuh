#pragma once
// 1. copy permuted matrix to device
//    (In the following: tile = partition)
//    tile_boundaries: array of indices giving the starting index of each partition.
//    Partitions are internally grouped. Starting with tile-edge nodes
//    followed by intra-tile nodes

//    intra_tile_sep: array of indices giving the starting index of the first
//    intra-tile node in each partition.
#include <cooperative_groups.h>
#include <hash.cuh>

#include <cub/cub.cuh>

namespace cg = cooperative_groups;

template <typename IndexType>
__forceinline__ __device__
void Partition2ShMem(IndexType* shMemRows,
                     IndexType* shMemCols,
                     IndexType* row_ptr,  // global mem
                     IndexType* col_ptr,  // global mem
                     IndexType* tile_boundaries){
  // determine which tile to load with blockId
  uint partNr = blockIdx.x;

  uint partSize = tile_boundaries[partNr+1] - tile_boundaries[partNr] + 1;
  // +1 to load one additional row_ptr value to determine size of last row

  uint part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  // put row_ptr partition in shMem
  for (uint i = threadIdx.x; i < partSize; i += blockDim.x){
    shMemRows[i] = row_ptr[i + part_offset];
  }

  cg::this_thread_block().sync();

  uint num_cols = shMemRows[partSize-1] - shMemRows[0];
  uint col_offset = shMemRows[0];
  for (uint i = threadIdx.x; i < num_cols; i += blockDim.x){
    shMemCols[i] = col_ptr[i + col_offset];
  }

  cg::this_thread_block().sync();
  // If partition contains n nodes then we now have
  // n+1 elements of row_ptr in shMem and the col_ptr values for
  // n nodes
}
struct Sum_Counters {
  __host__ __device__ __forceinline__ Counters
  operator()(const Counters& a, const Counters& b) const {
    Counters tmp;
    for (auto i = 0; i < max_bitWidth; ++i) {
      tmp.m[i] = a.m[i] + b.m[i];
    }
    return tmp;
  }
};

struct Max_Counters {
  __host__ __device__ __forceinline__ Counters
  operator()(const Counters& a, const Counters& b) const {
    Counters tmp;
    for (auto i = 0; i < max_bitWidth; ++i) {
      tmp.m[i] = max(a.m[i], b.m[i]);
    }
    return tmp;
  }
};

__device__ unsigned int retirementCount = 0;

template <typename IndexType>
__global__
void coloring1Kernel(IndexType* row_ptr,  // global mem
                     IndexType* col_ptr,  // global mem
                     IndexType* tile_boundaries,
                     IndexType* intra_tile_sep,   // <- useless (?)
                     uint m_rows,
                     uint tile_max_nodes,
                     uint tile_max_edges,
                     Counters* d_results){
  //uint number_of_tiles = gridDim.x;

  extern __shared__ IndexType shMem[];
  IndexType* shMemRows = shMem;                      // tile_max_nodes elements
  IndexType* shMemCols = &shMemRows[tile_max_nodes]; // tile_max_edges elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  // thread local array to count collisions
  Counters node_collisions;
  

  // Global mem access in tile_boundaries !!!!!
  uint partNr = blockIdx.x;
  uint local_rows = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  // for distance 1 we don't have to distinguish between intra tile nodes and
  // border nodes
  for (uint i = threadIdx.x; i < local_rows; i += blockDim.x){
    auto const glob_row = tile_boundaries[partNr] + i;
    auto const row_begin = shMemRows[i];
		auto const row_end = shMemRows[i + 1];

    for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
      auto const col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index

      auto const row_hash = hash(glob_row, static_k_param);
      auto const col_hash = hash(col, static_k_param);

      # pragma unroll max_bitWidth
      for(auto bit_w = 1; bit_w <= max_bitWidth; ++bit_w){
        uint mask = (1u << bit_w) - 1;
        if((row_hash & mask) == (col_hash & mask)){
          node_collisions.m[bit_w-1] += 1;
        }
        // else if {
        //   // if hashes differ in lower bits they also differ when increasing
        //   // the bit_width
        //   break;
        // }
      }
    }
  }

  cg::this_thread_block().sync();
  // Now each thread has counted all local collisions
  // and we can do sum and max reduction for desired output

  // // template BlockDim is bad
  typedef cub::BlockReduce<Counters, 512> BlockReduce;
  using TempStorageT = typename BlockReduce::TempStorage;

  TempStorageT temp_storage = reinterpret_cast<TempStorageT&>(shMem);

  Counters sum_accu = BlockReduce(temp_storage).Reduce(node_collisions,
                                                  Sum_Counters(), local_rows);
  cg::this_thread_block().sync();

  Counters max_accu = BlockReduce(temp_storage).Reduce(node_collisions,
                                                  Max_Counters(), local_rows);
  cg::this_thread_block().sync();

  if(threadIdx.x == 0){
    d_results[blockIdx.x] = sum_accu;
    d_results[blockIdx.x + gridDim.x] = max_accu;
  }

  __threadfence();

  __shared__ bool amLast;
  // Thread 0 takes a ticket
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
    // If the ticket ID is equal to the number of blocks, we are the last
    // block!
    amLast = (ticket == gridDim.x - 1);
  }

  cg::this_thread_block().sync();

  // The last block sums the results of all other blocks
  if (amLast && (threadIdx.x < gridDim.x)) {

    // Counters sum_accu = d_results[threadIdx.x];
  Counters sum_accu = BlockReduce(temp_storage).Reduce(d_results[threadIdx.x],
                                                  Sum_Counters(), gridDim.x);
  cg::this_thread_block().sync();

  Counters max_accu = BlockReduce(temp_storage).Reduce(d_results[threadIdx.x + gridDim.x],
                                                  Max_Counters(), gridDim.x);
  cg::this_thread_block().sync();

    if (threadIdx.x == 0) {
      d_results[0] = sum_accu;
      d_results[1] = max_accu;

      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}