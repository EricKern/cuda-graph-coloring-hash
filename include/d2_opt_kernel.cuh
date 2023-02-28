#pragma once
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>

#include "coloring.cuh"
#include "coloring_counters.cuh"
#include "sort_algorithms.hpp"


namespace apa22_coloring {

template <typename IndexT,
          typename WorkspaceT>  // same as hash output
__forceinline__ __device__
void D2CollisionsLocalThrustSrt(IndexT* shMemRows,
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

}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2KernelThrust(IndexT* row_ptr,  // global mem
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

  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // Counters total1, max1;
    Counters total2, max2;

    D2CollisionsLocalThrustSrt(shMemRows, shMemCols, shMemWorkspace, n_tileNodes,
                      part_offset, start_hash + k, total2, max2);
    Counters l1_sum2 = BlockReduceT(temp_storage)
                      .Reduce(total2, Sum_Counters{});
    cg::this_thread_block().sync();
    Counters l1_max2 = BlockReduceT(temp_storage)
                      .Reduce(max2, Max_Counters{});
    cg::this_thread_block().sync();

    // soa has an int array for each hash function.
    // In this array all block reduce results of first counter are stored
    // contiguous followed by the reduce results of the next counter ...
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;

        blocks_total2[idx] = l1_sum2.m[i];
        blocks_max2[idx] = l1_max2.m[i];
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
    auto& temp_storage_last =
        reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
    LastReduction<BlockReduceTLast>(blocks_total2, d_total2, gridDim.x,
                                    cub::Sum{}, temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max2, d_max2, gridDim.x,
                                    cub::Max{}, temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

template <typename IndexT,
          typename WorkspaceT>  // same as hash output
__forceinline__ __device__
void D2CollisionsLocalSNetSmall(IndexT* shMemRows,
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

    odd_even_merge_sort(thread_ws, j);

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
}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2SortNetSmall(IndexT* row_ptr,  // global mem
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

  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    Counters total2, max2;

    D2CollisionsLocalSNetSmall(shMemRows, shMemCols, shMemWorkspace, n_tileNodes,
                      part_offset, start_hash + k, total2, max2);
    Counters l1_sum2 = BlockReduceT(temp_storage)
                      .Reduce(total2, Sum_Counters{});
    cg::this_thread_block().sync();
    Counters l1_max2 = BlockReduceT(temp_storage)
                      .Reduce(max2, Max_Counters{});
    cg::this_thread_block().sync();

    // soa has an int array for each hash function.
    // In this array all block reduce results of first counter are stored
    // contiguous followed by the reduce results of the next counter ...
    if(threadIdx.x == 0){
      #pragma unroll num_bit_widths
      for (int i = 0; i < num_bit_widths; ++i) {
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;

        blocks_total2[idx] = l1_sum2.m[i];
        blocks_max2[idx] = l1_max2.m[i];
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
    auto& temp_storage_last =
        reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
    LastReduction<BlockReduceTLast>(blocks_total2, d_total2, gridDim.x,
                                    cub::Sum{}, temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max2, d_max2, gridDim.x,
                                    cub::Max{}, temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

}