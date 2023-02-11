#pragma once

#include "coloring.cuh"

namespace apa22_coloring {

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void d1KernelDblTmp(IndexT* row_ptr,
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
  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, n_tileNodes, part_offset,
                      start_hash + k, total_collisions, max_collisions);

    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      CountT l1_sum = BlockReduceT(temp_storage1)
                            .Reduce(total_collisions.m[i], cub::Sum{});
      CountT l1_max = BlockReduceT(temp_storage2)
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
                                temp_storage1);
    LastReduction<BlockReduceT>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                temp_storage2);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}


template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void d1KernelStructRed(IndexT* row_ptr,
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
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, n_tileNodes, part_offset,
                      start_hash + k, total_collisions, max_collisions);

    Counters l1_sum = BlockReduceT(temp_storage)
                        .Reduce(total_collisions, Sum_Counters{});
    cg::this_thread_block().sync();
    Counters l1_max = BlockReduceT(temp_storage)
                        .Reduce(max_collisions, Max_Counters{});
    cg::this_thread_block().sync();

      // block_total* holds mem for all intermediate results of each block.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      // Results for each hash function are stored after each other.
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = l1_sum.m[i];
        blocks_max1[idx] = l1_max.m[i];
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
  auto& temp_storage_last =
      reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  // The last block reduces the results of all other blocks
  if (amLast) {
    LastReduction<BlockReduceTLast>(blocks_total1, d_total, gridDim.x, cub::Sum{},
                                temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

template <typename IndexT>
__forceinline__ __device__
void D1CollisionsLocal2(IndexT* shMemRows,
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
          else {
            // if hashes differ in lower bits they also differ when increasing
            // the bit_width
            break;
          }
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

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void d1KernelStructRedBreak(IndexT* row_ptr,
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
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal2(shMemRows, shMemCols, n_tileNodes, part_offset,
                      start_hash + k, total_collisions, max_collisions);

    Counters l1_sum = BlockReduceT(temp_storage)
                        .Reduce(total_collisions, Sum_Counters{});
    cg::this_thread_block().sync();
    Counters l1_max = BlockReduceT(temp_storage)
                        .Reduce(max_collisions, Max_Counters{});
    cg::this_thread_block().sync();

      // block_total* holds mem for all intermediate results of each block.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      // Results for each hash function are stored after each other.
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = l1_sum.m[i];
        blocks_max1[idx] = l1_max.m[i];
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
  auto& temp_storage_last =
      reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  // The last block reduces the results of all other blocks
  if (amLast) {
    LastReduction<BlockReduceTLast>(blocks_total1, d_total, gridDim.x, cub::Sum{},
                                temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

}