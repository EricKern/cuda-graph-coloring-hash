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
#include <cstdint>    // for mask
#include <coloringCounters.cuh>

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
  int part_nr = blockIdx.x;

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

__device__ unsigned int retirementCount = 0;

template <typename IndexT>
__global__
void coloring1Kernel(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total,
                     SOACounters* soa_max,
                     Counters* d_total,
                     Counters* d_max){
  const IndexT partNr = blockIdx.x;

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  // using BlockReduceT = cub::BlockReduce<int, THREADS>;
  // using TempStorageT = typename BlockReduceT::TempStorage;
  
  typedef cub::BlockReduce<int, THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

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

  // The last block sums the results of all other blocks
  if (amLast) {
    int num_valid = cub::min(blockDim.x, gridDim.x);

    // Last block sum
    for (int k = 0; k < hash_params.len; ++k) {
      for (int i = 0; i < num_bit_widths; ++i) {
        int* start_addr = soa_total->m[k] + i * gridDim.x;
        int total_l2 = BlockLoadThreadReduce(start_addr, gridDim.x, cub::Sum{});
        total_l2 =
            BlockReduceT(temp_storage).Reduce(total_l2, cub::Sum{}, num_valid);
        cg::this_thread_block().sync();
        if (threadIdx.x == 0) {
          d_total[k].m[i] = total_l2;
        }
      }
    }
    // Last block max
    for (int k = 0; k < hash_params.len; ++k) {
      for (int i = 0; i < num_bit_widths; ++i) {
        int* start_addr = soa_max->m[k] + i * gridDim.x;
        int max_l2 = BlockLoadThreadReduce(start_addr, gridDim.x, cub::Max{});
        max_l2 =
            BlockReduceT(temp_storage).Reduce(max_l2, cub::Max{}, num_valid);
        cg::this_thread_block().sync();
        if (threadIdx.x == 0) {
          d_max[k].m[i] = max_l2;
        }
      }
    }

    if (threadIdx.x == 0) {
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


      # pragma unroll num_bit_widths
      for(auto counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx){
        auto shift_val = start_bit_width + counter_idx;
        std::make_unsigned_t<IndexType> mask = (1u << shift_val) - 1;
        int max_current = 0;
        int max_so_far = 0;

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