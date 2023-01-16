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
#include <coloringCounters.cuh>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>


namespace apa22_coloring {

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
  IndexType* shMemRows = shMem;                      // tile_max_nodes +1 elements
  IndexType* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  // thread local array to count collisions
  Counters current_collisions;
  Counters total_collisions;
  Counters max_collisions;
  

  // Global mem access in tile_boundaries !!!!!
  uint partNr = blockIdx.x;
  IndexType n_tileNodes = tile_boundaries[partNr+1]
                          - tile_boundaries[partNr];

  // for distance 1 we don't have to distinguish between intra tile nodes and
  // border nodes
  for (uint i = threadIdx.x; i < n_tileNodes; i += blockDim.x){
    auto const glob_row = tile_boundaries[partNr] + i;
    auto const row_begin = shMemRows[i];
		auto const row_end = shMemRows[i + 1];

    auto const row_hash = hash(glob_row, static_k_param);
    
    for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
      auto const col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if(col == glob_row)
        continue;
      auto const col_hash = hash(col, static_k_param);

      # pragma unroll max_bit_width
      for(auto counter_idx = 0; counter_idx < max_bit_width; ++counter_idx){
        auto shift_val = start_bit_width + counter_idx;

        std::make_unsigned_t<IndexType> mask = (1u << shift_val) - 1;
        if((row_hash & mask) == (col_hash & mask)){
          current_collisions.m[counter_idx] += 1;
        }
        // else if {
        //   // if hashes differ in lower bits they also differ when increasing
        //   // the bit_width
        //   break;
        // }
      }
    }
    Sum_Counters sum;
    Max_Counters max;
    total_collisions = sum(current_collisions, total_collisions);
    max_collisions = max(current_collisions, max_collisions);
    current_collisions = Counters{};
  }

  cg::this_thread_block().sync();
  // Now each thread has counted all local collisions
  // and we can do sum and max reduction for desired output

  // // template BlockDim is bad
  typedef cub::BlockReduce<Counters, THREADS> BlockReduce;
  using TempStorageT = typename BlockReduce::TempStorage;

  // TempStorageT temp_storage = *reinterpret_cast<TempStorageT*>(shMem);
  __shared__ TempStorageT temp_storage;

  Counters sum_accu = BlockReduce(temp_storage).Reduce(total_collisions,
                                                  Sum_Counters(), n_tileNodes);
  cg::this_thread_block().sync();

  Counters max_accu = BlockReduce(temp_storage).Reduce(max_collisions,
                                                  Max_Counters(), n_tileNodes);
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
  if (amLast) {

    Counters sum_accu;
    Counters max_accu;
    
    uint div = gridDim.x / (blockDim.x - 1);
    uint rem = gridDim.x % (blockDim.x - 1);
    uint block_valid;

    // if (threadIdx.x == 0) {
    //   sum_accu = d_results[threadIdx.x];
    //   max_accu = d_results[threadIdx.x + gridDim.x];
    // }

    int num_iters = rem == 0 ? div : div + 1;
    // complicated for loop in case we have more blocks than threads
    for (uint nr_reduction = 0; nr_reduction < num_iters; ++nr_reduction){
      if(nr_reduction < div){
        block_valid = blockDim.x;
      } else if (nr_reduction == div){
        block_valid = rem;
      } else {
        block_valid = 0;
      }

      // evlt. <=
      int local_tid = threadIdx.x - 1;
      if (threadIdx.x != 0 && local_tid < block_valid) {
        sum_accu = d_results[local_tid + nr_reduction * (blockDim.x - 1)];
        max_accu = d_results[local_tid + gridDim.x + nr_reduction * (blockDim.x - 1)];
      }

      sum_accu = BlockReduce(temp_storage).Reduce(sum_accu,
                                                      Sum_Counters(), block_valid + 1);
      cg::this_thread_block().sync();

      max_accu = BlockReduce(temp_storage).Reduce(max_accu,
                                                      Max_Counters(), block_valid + 1);
      cg::this_thread_block().sync();
      
    }

    if (threadIdx.x == 0) {
      d_results[0] = sum_accu;
      d_results[1] = max_accu;

      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

// needs double the ammount of ShMem
template <typename IndexType>
__global__
void coloring2Kernel(IndexType* row_ptr,  // global mem
                     IndexType* col_ptr,  // global mem
                     IndexType* tile_boundaries,
                     IndexType* intra_tile_sep,   // <- useless (?)
                     uint m_rows,
                     uint tile_max_nodes,
                     uint tile_max_edges,
                     Counters* d_results){
  //uint number_of_tiles = gridDim.x;

  extern __shared__ IndexType shMem[];
  IndexType* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexType* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexType* shMemWorkspace = &shMemCols[tile_max_edges]; //tile_max_nodes+tile_max_edges elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  // thread local array to count collisions
  Counters node_collisions;

  // Global mem access in tile_boundaries !!!!!
  uint partNr = blockIdx.x;
  IndexType n_tileNodes = tile_boundaries[partNr+1]
                          - tile_boundaries[partNr];

  for (uint i = threadIdx.x; i < n_tileNodes; i += blockDim.x){
    auto const glob_row = tile_boundaries[partNr] + i;
    auto const row_begin = shMemRows[i];
		auto const row_end = shMemRows[i + 1];

    auto const num_local_vertices = (row_end - row_begin) + 1;

    auto const local_workspace_begin = i + row_begin - shMemRows[0];
    auto const local_workspace_end = i + row_end - shMemRows[0];
    
    shMemWorkspace[local_workspace_begin] = hash(glob_row, static_k_param);
    
    for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
      auto const col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if(col == glob_row)
        continue;

      shMemWorkspace[i + col_idx - shMemRows[0] + 1]
        = hash(col, static_k_param);

      // sort hashes in workspace_
			thrust::sort(thrust::seq,
			             shMemWorkspace + local_workspace_begin,
			             shMemWorkspace + local_workspace_end,
			             brev_cmp<IndexType>{});


      # pragma unroll max_bit_width
      for(auto counter_idx = 0; counter_idx < max_bit_width; ++counter_idx){
        auto shift_val = start_bit_width + counter_idx;
        std::make_unsigned_t<IndexType> mask = (1u << shift_val) - 1;
        Counters::value_type max_current = 0;
        Counters::value_type max_so_far = 0;

        auto hash = shMemWorkspace[local_workspace_begin];
        for (auto vertex_idx = 1; vertex_idx < num_local_vertices;
              ++vertex_idx) {
          auto next_hash = shMemWorkspace[local_workspace_begin + vertex_idx];

          if((hash & mask) == (next_hash & mask)){
            max_current += 1;
          } else {

            max_so_far = (max_current > max_so_far) ? max_current : max_so_far;
            max_current = 0;
            hash = next_hash;
          }
        }
        max_so_far = (max_current > max_so_far) ? max_current : max_so_far;
        node_collisions.m[counter_idx] = max_so_far;
      }
    }
  }

  cg::this_thread_block().sync();
  // Now each thread has counted all local collisions
  // and we can do sum and max reduction for desired output

  // // template BlockDim is bad
  typedef cub::BlockReduce<Counters, THREADS> BlockReduce;
  using TempStorageT = typename BlockReduce::TempStorage;

  // TempStorageT temp_storage = reinterpret_cast<TempStorageT&>(shMem);
  __shared__ TempStorageT temp_storage;

  Counters sum_accu = BlockReduce(temp_storage).Reduce(node_collisions,
                                                  Sum_Counters(), n_tileNodes);
  cg::this_thread_block().sync();

  Counters max_accu = BlockReduce(temp_storage).Reduce(node_collisions,
                                                  Max_Counters(), n_tileNodes);
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

} // end apa22_coloring