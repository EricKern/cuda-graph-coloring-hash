#pragma once

#include <cstdint>    // for mask

#include <cooperative_groups.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

#include "hash.cuh"
#include "coloring_counters.cuh"
#include "odd_even_sort.hpp"
#include "util.cuh"


namespace apa22_coloring {

namespace cg = cooperative_groups;


template <typename IndexT>
__forceinline__ __device__
void Partition2ShMem(IndexT* shMemRows,
                     IndexT* shMemCols,
                     IndexT* row_ptr,     // global mem
                     IndexT* col_ptr,     // global mem
                     IndexT part_offset,  // tile_boundaries[part_nr]
                     int n_tileNodes) {
  int part_size = n_tileNodes + 1;
  // +1 to load one additional row_ptr value to determine size of last row

  // put row_ptr partition in shMem
  for (int i = threadIdx.x; i < part_size; i += blockDim.x) {
    shMemRows[i] = row_ptr[i + part_offset];
  }

  cg::this_thread_block().sync();

  int num_cols = shMemRows[part_size-1] - shMemRows[0];
  IndexT col_offset = shMemRows[0];
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    shMemCols[i] = col_ptr[i + col_offset];
  }

  cg::this_thread_block().sync();
  // If partition contains n nodes then we now have
  // n+1 elements of row_ptr in shMem and the col_ptr values for
  // n nodes
}

template <typename IndexT>
__forceinline__ __device__
void D1CollisionsLocal(IndexT* shMemRows,
                       IndexT* shMemCols,
                       const int n_tileNodes,
                       IndexT part_offset,    // tile_boundaries[part_nr]
                       int hash_param,
                       Counters& total,
                       Counters& max) {
  Counters current_collisions;

  // for distance 1 we don't have to distinguish between intra tile nodes and
  // border nodes
  for (int i = threadIdx.x; i < n_tileNodes; i += blockDim.x) {
    IndexT glob_row = part_offset + i;
    IndexT row_begin = shMemRows[i];
		IndexT row_end = shMemRows[i + 1];

    const auto row_hash = hash(glob_row, hash_param);
    
    for (IndexT col_idx = row_begin; col_idx < row_end; ++col_idx) {
      IndexT col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if (col != glob_row) {
        const auto col_hash = hash(col, hash_param);

        #pragma unroll num_bit_widths
        for (int counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx) {
          int shift_val = start_bit_width + counter_idx;

          std::uint32_t mask = (1u << shift_val) - 1u;
          if ((row_hash & mask) == (col_hash & mask)) {
            current_collisions.m[counter_idx] += 1;
          }
          // else {
          //   // if hashes differ in lower bits they also differ when increasing
          //   // the bit_width
          //   break;
          // }
        }
      }
    }
    Sum_Counters sum_ftor;
    Max_Counters max_ftor;
    total = sum_ftor(current_collisions, total);
    max = max_ftor(current_collisions, max);
    current_collisions = Counters{};
  }

}

template <typename CountT,
          typename ReductionOp>
__forceinline__ __device__
CountT BlockLoadThreadReduce(CountT* g_addr, int num_elem, ReductionOp red_op) {
  CountT sum = 0;
  for (int i = threadIdx.x; i < num_elem; i += blockDim.x) {
    sum = red_op(g_addr[i], sum);
  }
  return sum;
}

template <typename CubReduceT,
          typename CountT,
          typename ReductionOp>
__forceinline__ __device__
void LastReduction(CountT* blocks_res_ptr, Counters* out_counters, int n_tiles,
                       ReductionOp op,
                       typename CubReduceT::TempStorage& temp_storage) {
  const int elem_p_hash_fn = num_bit_widths * n_tiles;
  // Initial reduction during load in BlockLoadThreadReduce limits thread
  // local results to at most blockDim.x

  // Last block reduction
  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    int hash_offset = k * elem_p_hash_fn;
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      CountT* start_addr = blocks_res_ptr + hash_offset + i * n_tiles;

      CountT accu = BlockLoadThreadReduce(start_addr, n_tiles, op);
      accu = CubReduceT(temp_storage).Reduce(accu, op);
      cg::this_thread_block().sync();
      if (threadIdx.x == 0) {
        out_counters[k].m[i] = accu;
      }
    }
  }
}

__device__ unsigned int retirementCount = 0;

// We rely on guarantees of tiling i.e. constant tile size
// (n_rows + n_cols + 1) = constant, NumTiles = gridDim.x
template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1Kernel(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  using CountT = Counters::value_type;
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                  part_offset, n_tileNodes);
  typedef cub::BlockReduce<CountT, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, n_tileNodes, part_offset,
                      start_hash + k, total_collisions, max_collisions);

    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      CountT l1_sum = BlockReduceT(temp_storage)
                            .Reduce(total_collisions.m[i], cub::Sum{});
      cg::this_thread_block().sync();
      CountT l1_max = BlockReduceT(temp_storage)
                            .Reduce(max_collisions.m[i], cub::Max{});
      cg::this_thread_block().sync();

      // block_total* holds mem for all intermediate results of each block.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      // Results for each hash function are stored after each other.
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = l1_sum;
        blocks_max1[idx] = l1_max;
      }
    }
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

  // The last block reduces the results of all other blocks
  if (amLast) {
    LastReduction<BlockReduceT>(blocks_total1, d_total, gridDim.x, cub::Sum{},
                                temp_storage);
    LastReduction<BlockReduceT>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}


template <typename IndexT,
          typename WorkspaceT>  // same as hash output
__forceinline__ __device__
void D2CollisionsLocal(IndexT* shMemRows,
                       IndexT* shMemCols,
                       WorkspaceT* shMemWorkspace,
                       const int n_tileNodes,
                       IndexT part_offset,    // tile_boundaries[part_nr]
                       const int hash_param,
                       Counters& total,
                       Counters& max) {
  Counters current_collisions;

  for (int i = threadIdx.x; i < n_tileNodes; i += blockDim.x) {
    IndexT glob_row = part_offset + i;
    IndexT row_begin = shMemRows[i];
		IndexT row_end = shMemRows[i + 1];

    const int thread_ws_begin = i + row_begin - shMemRows[0];
    // const int thread_ws_end = i + row_end - shMemRows[0];
    // const int thread_ws_len = thread_ws_end - thread_ws_begin;

    WorkspaceT* thread_ws = shMemWorkspace + thread_ws_begin;

    thread_ws[0] = __brev(hash(glob_row, hash_param));
    
    int j = 1;
    for (IndexT col_idx = row_begin; col_idx < row_end; ++col_idx) {
      IndexT col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if (col != glob_row) {
        thread_ws[j] = __brev(hash(col, hash_param));
        ++j;
      }
    }

    thrust::sort(thrust::seq, thread_ws, thread_ws + j);

    // We define node collisions at dist 2 as maximum number of equal patterns
    // in a group - 1
    // e.g. for bitWidth 2 we get the list {00, 01, 01, 01, 10, 11}
    // In this case we have 3x "01" as biggest group. So collisions would be 2.
    // We subtract 1 because if each group has 1 element the coloring would be
    // good so there are no collisions

    // Now we actually do reduce_by_key of list with all ones. With bitmask applied to
    // sorted list as keys. Then we find max in reduced array.

    #pragma unroll num_bit_widths
    for(int counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx){
      auto shift_val = start_bit_width + counter_idx;

      std::uint32_t mask = (1u << shift_val) - 1;
      int max_so_far = 1;
      int current = 1;

      auto group_start_hash = __brev(thread_ws[0]);
      for (auto edge_idx = 1; edge_idx < j; ++edge_idx) {
        auto next_hash = __brev(thread_ws[edge_idx]);

        if((group_start_hash & mask) == (next_hash & mask)){
          current += 1;
        } else {
          group_start_hash = next_hash;

          max_so_far = (current > max_so_far) ? current : max_so_far;
          current = 1;
        }
      }
      max_so_far = (current > max_so_far) ? current : max_so_far;
      // Put max_so_far - 1 in counter for current bit width
      current_collisions.m[counter_idx] = max_so_far - 1;
    }

    Sum_Counters sum_ftor;
    Max_Counters max_ftor;
    total = sum_ftor(current_collisions, total);
    max = max_ftor(current_collisions, max);
    current_collisions = Counters{};
  }

  // remove if shared mem is not reused. Depends if cub doesn't require that
  // all threads have results present when calling block reduce.
  cg::this_thread_block().sync();

}


template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2Kernel(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters::value_type* blocks_total2,
                     Counters::value_type* blocks_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {
  using HashT = std::uint32_t;
  using CountT = Counters::value_type;
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements

  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  // n_tileNodes + n_tileEdges + 1 elements
  // Since shMemCols is 32 or 64 bit, allignment for HashT which is smaller
  // is fine.
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                  part_offset, n_tileNodes);

  typedef cub::BlockReduce<CountT, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // Counters total1, max1;
    Counters total2, max2;
    // D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
    //                   total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, n_tileNodes,
                      part_offset, start_hash + k, total2, max2);
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      // CountT l1_sum1 = BlockReduceT(temp_storage)
      //                  .Reduce(total1.m[i], cub::Sum{});
      // cg::this_thread_block().sync();
      // CountT l1_max1 = BlockReduceT(temp_storage)
      //                  .Reduce(max1.m[i], cub::Max{});
      // cg::this_thread_block().sync();
      CountT l1_sum2 = BlockReduceT(temp_storage)
                       .Reduce(total2.m[i], cub::Sum{});
      cg::this_thread_block().sync();
      CountT l1_max2 = BlockReduceT(temp_storage)
                       .Reduce(max2.m[i], cub::Max{});
      cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        // blocks_total1[idx] = l1_sum1;
        // blocks_max1[idx] = l1_max1;
        blocks_total2[idx] = l1_sum2;
        blocks_max2[idx] = l1_max2;
      }
    }
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

  // The last block reduces the results of all other blocks
  if (amLast) {
    // LastReduction<BlockReduceT>(blocks_total1, d_total1, gridDim.x, cub::Sum{}, temp_storage);
    // LastReduction<BlockReduceT>(blocks_max1, d_max1, gridDim.x, cub::Max{}, temp_storage);

    LastReduction<BlockReduceT>(blocks_total2, d_total2, gridDim.x, cub::Sum{}, temp_storage);
    LastReduction<BlockReduceT>(blocks_max2, d_max2, gridDim.x, cub::Max{}, temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

template <typename IndexT,
          typename WorkspaceT>
__forceinline__ __device__
void D2CollisionsSortNet(IndexT* shMemRows,
                         IndexT* shMemCols,
                         WorkspaceT* shMemWorkspace,
                         const int max_node_degree,
                         const int n_tileNodes,
                         IndexT part_offset,    // tile_boundaries[part_nr]
                         const int hash_param,
                         Counters& total,
                         Counters& max) {
  Counters current_collisions;

  const int mem_per_node = roundUp(max_node_degree + 1, 32) + 1;

  for (int i = threadIdx.x; i < n_tileNodes; i += blockDim.x) {
    IndexT glob_row = part_offset + i;
    IndexT row_begin = shMemRows[i];
		IndexT row_end = shMemRows[i + 1];

    const int thread_ws_begin = i * mem_per_node;
    // const int thread_ws_end = i + row_end - shMemRows[0];
    // const int thread_ws_len = thread_ws_end - thread_ws_begin;

    WorkspaceT* thread_ws = shMemWorkspace + thread_ws_begin;

    thread_ws[0] = __brev(hash(glob_row, hash_param));
    
    int j = 1;
    for (IndexT col_idx = row_begin; col_idx < row_end; ++col_idx) {
      IndexT col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if (col != glob_row) {
        thread_ws[j] = __brev(hash(col, hash_param));
        ++j;
      }
    }
    for (int k = j; k < mem_per_node; ++k) {
      thread_ws[k] = cub::Traits<WorkspaceT>::MAX_KEY;
      // Max value of uint32 so that padding elements are at the very end of 
      // the sorted bitreversed array
    }

    // sorting network to avoid bank conflicts
    odd_even_merge_sort(thread_ws, mem_per_node);


    // We define node collisions at dist 2 as maximum number of equal patterns
    // in a group - 1
    // e.g. for bitWidth 2 we get the list {00, 01, 01, 01, 10, 11}
    // In this case we have 3x "01" as biggest group. So collisions would be 2.
    // We subtract 1 because if each group has 1 element the coloring would be
    // good so there are no collisions

    // Now we actually do reduce_by_key of list with all ones. With bitmask applied to
    // sorted list as keys. Then we find max in reduced array.

    #pragma unroll num_bit_widths
    for(int counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx){
      auto shift_val = start_bit_width + counter_idx;

      uint32_t mask = (1u << shift_val) - 1;
      int max_so_far = 1;
      int current = 1;

      auto group_start_hash = __brev(thread_ws[0]);
      for (auto edge_idx = 1; edge_idx < j; ++edge_idx) {
        auto next_hash = __brev(thread_ws[edge_idx]);

        if((group_start_hash & mask) == (next_hash & mask)){
          current += 1;
        } else {
          group_start_hash = next_hash;

          max_so_far = (current > max_so_far) ? current : max_so_far;
          current = 1;
        }
      }
      max_so_far = (current > max_so_far) ? current : max_so_far;
      // Put max_so_far - 1 in counter for current bit width
      current_collisions.m[counter_idx] = max_so_far - 1;
    }

    Sum_Counters sum_ftor;
    Max_Counters max_ftor;
    total = sum_ftor(current_collisions, total);
    max = max_ftor(current_collisions, max);
    current_collisions = Counters{};
  }

  // remove if shared mem is not reused. Depends if cub doesn't require that
  // all threads have results present when calling block reduce.
  cg::this_thread_block().sync();

}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2KernelBank(IndexT* row_ptr,  // global mem
                         IndexT* col_ptr,  // global mem
                         IndexT* tile_boundaries,
                         int max_node_degree,
                         Counters::value_type* blocks_total1,
                         Counters::value_type* blocks_max1,
                         Counters::value_type* blocks_total2,
                         Counters::value_type* blocks_max2,
                         Counters* d_total1,
                         Counters* d_max1,
                         Counters* d_total2,
                         Counters* d_max2) {
  using HashT = std::uint32_t;
  using CountT = Counters::value_type;
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileNodes = tile_boundaries[partNr+1] - part_offset;
  const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  // Since shMemCols is 32 or 64 bit, allignment for HashT which is smaller
  // is fine. Tiling guarantees that shMemWorkspace is enough

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, part_offset,
                  n_tileNodes);

  typedef cub::BlockReduce<CountT, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // Counters total1, max1;
    Counters total2, max2;
    // D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
    //                   total1, max1);
    D2CollisionsSortNet(shMemRows, shMemCols, shMemWorkspace,
                          max_node_degree, n_tileNodes, part_offset,
                          start_hash + k, total2, max2);
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      // CountT l1_sum1 = BlockReduceT(temp_storage)
      //                  .Reduce(total1.m[i], cub::Sum{});
      // cg::this_thread_block().sync();
      // CountT l1_max1 = BlockReduceT(temp_storage)
      //                  .Reduce(max1.m[i], cub::Max{});
      // cg::this_thread_block().sync();
      CountT l1_sum2 = BlockReduceT(temp_storage)
                       .Reduce(total2.m[i], cub::Sum{});
      cg::this_thread_block().sync();
      CountT l1_max2 = BlockReduceT(temp_storage)
                       .Reduce(max2.m[i], cub::Max{});
      cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        // blocks_total1[idx] = l1_sum1;
        // blocks_max1[idx] = l1_max1;
        blocks_total2[idx] = l1_sum2;
        blocks_max2[idx] = l1_max2;
      }
    }
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

  // The last block reduces the results of all other blocks
  if (amLast) {
    // LastReduction<BlockReduceT>(blocks_total1, d_total1, gridDim.x, cub::Sum{}, temp_storage);
    // LastReduction<BlockReduceT>(blocks_max1, d_max1, gridDim.x, cub::Max{}, temp_storage);

    LastReduction<BlockReduceT>(blocks_total2, d_total2, gridDim.x, cub::Sum{}, temp_storage);
    LastReduction<BlockReduceT>(blocks_max2, d_max2, gridDim.x, cub::Max{}, temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

} // end apa22_coloring