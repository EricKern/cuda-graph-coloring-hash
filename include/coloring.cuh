#pragma once

#include <cooperative_groups.h>
#include <hash.cuh>
#include <cstdint>    // for mask
#include <coloringCounters.cuh>
#include <odd_even_sort.hpp>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>


namespace apa22_coloring {

namespace cg = cooperative_groups;


template <typename IndexT>
__forceinline__ __device__
void Partition2ShMem(IndexT* shMemRows,
                     IndexT* shMemCols,
                     IndexT* row_ptr,     // global mem
                     IndexT* col_ptr,     // global mem
                     IndexT* tile_boundaries) {
  // Determine which tile to load with blockId
  const int part_nr = blockIdx.x;

  int part_size = tile_boundaries[part_nr+1] - tile_boundaries[part_nr] + 1;
  // +1 to load one additional row_ptr value to determine size of last row

  IndexT part_offset = tile_boundaries[part_nr];   // offset in row_ptr array
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
                       IndexT* tile_boundaries,
                       int hash_param,
                       Counters& total,
                       Counters& max) {
  Counters current_collisions;
  // Global mem access in tile_boundaries !!!!!
  const int part_nr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[part_nr+1] - tile_boundaries[part_nr];

  // for distance 1 we don't have to distinguish between intra tile nodes and
  // border nodes
  for (int i = threadIdx.x; i < n_tileNodes; i += blockDim.x) {
    IndexT glob_row = tile_boundaries[part_nr] + i;
    IndexT row_begin = shMemRows[i];
		IndexT row_end = shMemRows[i + 1];

    const auto row_hash = hash(glob_row, hash_param);
    
    for (IndexT col_idx = row_begin; col_idx < row_end; ++col_idx) {
      IndexT col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if (col != glob_row) {
        const auto col_hash = hash(col, hash_param);

        # pragma unroll num_bit_widths
        for (int counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx) {
          int shift_val = start_bit_width + counter_idx;

          std::uint32_t mask = (1u << shift_val) - 1u;
          if ((row_hash & mask) == (col_hash & mask)) {
            current_collisions.m[counter_idx] += 1;
          }
          // else if {
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

  cg::this_thread_block().sync();
}

template <typename ReductionOp>
__forceinline__ __device__
int BlockLoadThreadReduce(int* g_addr, int num_elem, ReductionOp red_op) {
  int sum = 0;
  for (int i = threadIdx.x; i < num_elem; i += blockDim.x) {
    sum = red_op(g_addr[i], sum);
  }
  return sum;
}

template <typename CubReduceT,
          typename ReductionOp>
__forceinline__ __device__
void D1LastReduction(SOACounters* soa, Counters* out_counters, ReductionOp op,
                     typename CubReduceT::TempStorage& temp_storage) {
  int num_valid = cub::min(blockDim.x, gridDim.x);
  // Initial reduction during load in BlockLoadThreadReduce limits thread
  // local results to at most blockDim.x

  // Last block reduction
  for (int k = 0; k < hash_params.len; ++k) {
    for (int i = 0; i < num_bit_widths; ++i) {
      int* start_addr = soa->m[k] + i * gridDim.x;
      int accu = BlockLoadThreadReduce(start_addr, gridDim.x, op);
      accu = CubReduceT(temp_storage).Reduce(accu, op, num_valid);
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
template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1Kernel(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     SOACounters* soa_total,
                     SOACounters* soa_max,
                     Counters* d_total,
                     Counters* d_max) {
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  for (int k = 0; k < hash_params.len; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total_collisions, max_collisions);

    for (int i = 0; i < num_bit_widths; ++i) {
      int l1_sum = BlockReduceT(temp_storage)
                       .Reduce(total_collisions.m[i], cub::Sum{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_max = BlockReduceT(temp_storage)
                       .Reduce(max_collisions.m[i], cub::Max{}, n_tileNodes);
      cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        soa_total->m[k][partNr + i * gridDim.x] = l1_sum;
        soa_max->m[k][partNr + i * gridDim.x] = l1_max;
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
    D1LastReduction<BlockReduceT>(soa_total, d_total, cub::Sum{}, temp_storage);
    D1LastReduction<BlockReduceT>(soa_max, d_max, cub::Max{}, temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}


template <typename IndexT>
__forceinline__ __device__
void D2CollisionsLocal(IndexT* shMemRows,
                       IndexT* shMemCols,
                       IndexT* shMemWorkspace,
                       IndexT* tile_boundaries,
                       int hash_param,
                       Counters& total,
                       Counters& max) {
  Counters current_collisions;
  // Global mem access in tile_boundaries !!!!!
  const int part_nr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[part_nr+1] - tile_boundaries[part_nr];

  for (int i = threadIdx.x; i < n_tileNodes; i += blockDim.x) {
    IndexT glob_row = tile_boundaries[part_nr] + i;
    IndexT row_begin = shMemRows[i];
		IndexT row_end = shMemRows[i + 1];

    const int thread_ws_begin = i + row_begin - shMemRows[0];
    // const int thread_ws_end = i + row_end - shMemRows[0];
    // const int thread_ws_len = thread_ws_end - thread_ws_begin;

    IndexT* thread_ws = shMemWorkspace + thread_ws_begin;

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

    # pragma unroll num_bit_widths
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


template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2Kernel(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  IndexT* shMemWorkspace = &shMemCols[n_tileEdges]; // n_tileNodes + n_tileEdges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    for (int i = 0; i < num_bit_widths; ++i) {
      int l1_sum1 = BlockReduceT(temp_storage)
                       .Reduce(total1.m[i], cub::Sum{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_max1 = BlockReduceT(temp_storage)
                       .Reduce(max1.m[i], cub::Max{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_sum2 = BlockReduceT(temp_storage)
                       .Reduce(total2.m[i], cub::Sum{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_max2 = BlockReduceT(temp_storage)
                       .Reduce(max2.m[i], cub::Max{}, n_tileNodes);
      cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        soa_total1->m[k][partNr + i * gridDim.x] = l1_sum1;
        soa_max1->m[k][partNr + i * gridDim.x] = l1_max1;
        soa_total2->m[k][partNr + i * gridDim.x] = l1_sum2;
        soa_max2->m[k][partNr + i * gridDim.x] = l1_max2;
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
    D1LastReduction<BlockReduceT>(soa_total1, d_total1, cub::Sum{}, temp_storage);
    D1LastReduction<BlockReduceT>(soa_max1, d_max1, cub::Max{}, temp_storage);

    D1LastReduction<BlockReduceT>(soa_total2, d_total2, cub::Sum{}, temp_storage);
    D1LastReduction<BlockReduceT>(soa_max2, d_max2, cub::Max{}, temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

__device__ __forceinline__
int roundUp(int numToRound, int multiple) {
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

template <typename IndexT>
__forceinline__ __device__
void D2CollisionsLocalBank(IndexT* shMemRows,
                           IndexT* shMemCols,
                           IndexT* shMemWorkspace,
                           IndexT* tile_boundaries,
                           int max_node_degree,
                           int hash_param,
                           Counters& total,
                           Counters& max) {
  Counters current_collisions;
  // Global mem access in tile_boundaries !!!!!
  const int part_nr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[part_nr+1] - tile_boundaries[part_nr];
  const int mem_per_node = roundUp(max_node_degree + 1, 32) + 1;

  for (int i = threadIdx.x; i < n_tileNodes; i += blockDim.x) {
    IndexT glob_row = tile_boundaries[part_nr] + i;
    IndexT row_begin = shMemRows[i];
		IndexT row_end = shMemRows[i + 1];

    const int thread_ws_begin = i * mem_per_node;
    // const int thread_ws_end = i + row_end - shMemRows[0];
    // const int thread_ws_len = thread_ws_end - thread_ws_begin;

    IndexT* thread_ws = shMemWorkspace + thread_ws_begin;

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
      thread_ws[k] = __brev(-2);
    }

    // thrust::sort(thrust::seq, thread_ws, thread_ws + mem_per_node);
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

    # pragma unroll num_bit_widths
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

template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2KernelBank(IndexT* row_ptr,  // global mem
                         IndexT* col_ptr,  // global mem
                         IndexT* tile_boundaries,
                         int max_node_degree,
                         SOACounters* soa_total1,
                         SOACounters* soa_max1,
                         SOACounters* soa_total2,
                         SOACounters* soa_max2,
                         Counters* d_total1,
                         Counters* d_max1,
                         Counters* d_total2,
                         Counters* d_max2) {
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  IndexT* shMemWorkspace = &shMemCols[n_tileEdges]; // n_tileNodes + n_tileEdges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocalBank(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                          max_node_degree, hash_params.val[k], total2, max2);
    for (int i = 0; i < num_bit_widths; ++i) {
      int l1_sum1 = BlockReduceT(temp_storage)
                       .Reduce(total1.m[i], cub::Sum{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_max1 = BlockReduceT(temp_storage)
                       .Reduce(max1.m[i], cub::Max{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_sum2 = BlockReduceT(temp_storage)
                       .Reduce(total2.m[i], cub::Sum{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_max2 = BlockReduceT(temp_storage)
                       .Reduce(max2.m[i], cub::Max{}, n_tileNodes);
      cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        soa_total1->m[k][partNr + i * gridDim.x] = l1_sum1;
        soa_max1->m[k][partNr + i * gridDim.x] = l1_max1;
        soa_total2->m[k][partNr + i * gridDim.x] = l1_sum2;
        soa_max2->m[k][partNr + i * gridDim.x] = l1_max2;
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
    D1LastReduction<BlockReduceT>(soa_total1, d_total1, cub::Sum{}, temp_storage);
    D1LastReduction<BlockReduceT>(soa_max1, d_max1, cub::Max{}, temp_storage);

    D1LastReduction<BlockReduceT>(soa_total2, d_total2, cub::Sum{}, temp_storage);
    D1LastReduction<BlockReduceT>(soa_max2, d_max2, cub::Max{}, temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

} // end apa22_coloring