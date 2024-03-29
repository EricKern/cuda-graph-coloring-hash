#pragma once
#include <cub/cub.cuh>

#include "coloring_counters.cuh"
#include "coloring.cuh"
#include "coop_launch.cuh"
#include "warp_hash.cuh"

using namespace apa22_coloring;

template <int THREADS,
          int N_HASHES,
          int N_BITW,
          typename CountT,
          typename SCountT>
__forceinline__ __device__
void Smem_ReductionCoop(SCountT* shMem_collisions,
                    int ceil_nodes, int tiles, int blk_i,
                    CountT* blocks_total1,
                    CountT* blocks_max1) {
  // Reduction
  // Must pad shared memory per bit width to full multiple of groups_per_warp
  const int warp_id = threadIdx.x / 32;
  const int seg_end = ceil_nodes * N_HASHES;
  for (int i = warp_id; i < N_BITW; i += THREADS / 32) {
    CountT sum_red = 0;
    CountT max_red = 0;
    SCountT* start = &shMem_collisions[ceil_nodes * N_HASHES * i];

    for (int warp_chunk = 0; warp_chunk < seg_end; warp_chunk += 32) {
      
      CountT node_col = static_cast<CountT>(start[warp_chunk + cub::LaneId()]);
      sum_red = cub::Sum{}(sum_red, node_col);
      max_red = cub::Max{}(max_red, node_col);
    }
    ShflGroupsDown<N_HASHES>(sum_red, cub::Sum{});
    ShflGroupsDown<N_HASHES>(max_red, cub::Max{});

    if (cub::LaneId() < N_HASHES) {
      int idx = cub::LaneId() * N_HASHES + i;
      blocks_total1[idx] = cub::Sum{}(sum_red, blocks_total1[idx]);
      blocks_max1[idx] = cub::Max{}(max_red, blocks_max1[idx]);
    }
  }
}

template <typename CountT,  
        int N_HASHES, 
        int N_BITW>
__device__ __forceinline__
void init_countersCoop(CountT* arr1, CountT* arr2) {
  for (int i = threadIdx.x; i < N_HASHES * N_BITW; i += blockDim.x) {
    arr1[i] = 0;
    arr2[i] = 0;
  }
  cg::this_thread_block().sync();
}

template <int THREADS,
          int BLK_SM,
          int N_HASHES = 16,
          int START_HASH = 3,
          typename CountT = int,
          typename SCountT = char,     // also consider this during tiling
          int N_BITW = 8,
          int START_BITW = 3,
          typename IndexT>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1coopWarp(IndexT* row_ptr,
                   IndexT* col_ptr,
                   IndexT* tile_boundaries,   // gridDim.x + 1 elements
                   int tiles,
                   CountT* blocks_total1,
                   CountT* blocks_max1,
                   Counters* d_total,
                   Counters* d_max) {

    __shared__ CountT coop_total1[N_HASHES * N_BITW];
    __shared__ CountT coop_max1[N_HASHES * N_BITW];
    init_countersCoop<CountT, N_HASHES, N_BITW>(coop_total1, coop_max1);

    extern __shared__ IndexT shMem[];

    const int groups_per_warp = 32 / N_HASHES;
    
    int my_start, my_work;
    work_distribution(tiles, my_start, my_work);
    for(int blk_i = my_start; blk_i < my_start+my_work; ++blk_i) {
      const int partNr = blk_i;
      const IndexT part_offset = tile_boundaries[partNr];
      const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];
      const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];
      const int ceil_nodes = roundUp(n_tileNodes, groups_per_warp);

      IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
      IndexT* shMemCols = &shMemRows[n_tileNodes+1];
      SCountT* shMem_collisions = reinterpret_cast<SCountT*>(&shMemCols[n_tileEdges]);
      zero_smem(shMem_collisions, ceil_nodes * N_BITW * N_HASHES);
      
      Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                      part_offset, n_tileNodes);

      D1WarpCollisions<THREADS, N_HASHES, START_HASH, N_BITW, START_BITW, SCountT>(
                      n_tileNodes, ceil_nodes, part_offset, shMemRows, shMemCols,
                      shMem_collisions);
      cg::this_thread_block().sync();

      Smem_ReductionCoop<THREADS, N_HASHES, N_BITW>(shMem_collisions,
                                          ceil_nodes, tiles, blk_i,
                                          coop_total1,
                                          coop_max1);
      cg::this_thread_block().sync();
    }
    //write to gloal memory
    const int warp_id = threadIdx.x / 32;
    for (int i = warp_id; i < N_BITW; i += THREADS / 32) {
        if (cub::LaneId() < N_HASHES) {
            const int elem_p_hash_fn = N_BITW * gridDim.x;
            int idx = cub::LaneId() * N_HASHES + i;
            int idxOut = cub::LaneId() * elem_p_hash_fn + i * gridDim.x + blockIdx.x; 
            blocks_total1[idxOut] = coop_total1[idx];
            blocks_max1[idxOut] = coop_max1[idx];
        }
    }
    cg::this_grid().sync();

    typedef cub::BlockReduce<CountT, THREADS> BlockReduceT;
    auto& temp_storage = reinterpret_cast<typename BlockReduceT::TempStorage&>(shMem);
    // The last block reduces the results of all other blocks
    if (blockIdx.x == 0) {
      LastReduction<BlockReduceT>(blocks_total1, d_total, gridDim.x, cub::Sum{},
                                  temp_storage);
      LastReduction<BlockReduceT>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                  temp_storage);
    }
}


template <int THREADS,
          int BLK_SM,
          int N_HASHES = 16,
          int START_HASH = 3,
          typename CountT = int,
          typename SCountT = char,     // also consider this during tiling
          int N_BITW = 8,
          int START_BITW = 3,
          typename IndexT>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2coopWarp(IndexT* row_ptr,  // global mem
                   IndexT* col_ptr,  // global mem
                   IndexT* tile_boundaries,
                   int tiles,
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

  __shared__ CountT coop_total2[N_BITW * N_HASHES];
  __shared__ CountT coop_max2[N_BITW * N_HASHES];
  init_countersCoop<CountT, N_HASHES, N_BITW>(coop_total2, coop_max2);

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  const int groups_per_warp = 32 / N_HASHES;
  const int single_node_list_elems = (roundUp(max_node_degree, 32) + 1) * N_HASHES;

  int my_start, my_work;
  work_distribution(tiles, my_start, my_work);
  for(int blk_i = my_start; blk_i < my_start+my_work; ++blk_i) {
    const int partNr = blk_i;
    const IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
    const int n_tileNodes = tile_boundaries[partNr+1] - part_offset;
    const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];
    const int ceil_nodes = roundUp(n_tileNodes, groups_per_warp);
  
    const int list_space = single_node_list_elems * n_tileNodes;

    IndexT* shMemRows = shMem;  // n_tileNodes +1 elements
    IndexT* shMemCols = &shMemRows[n_tileNodes + 1];
    HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
    SCountT* shMem_collisions = reinterpret_cast<SCountT*>(&shMemWorkspace[list_space]);
    zero_smem(shMem_collisions, ceil_nodes * N_BITW * N_HASHES);

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                    part_offset, n_tileNodes);
  
    D2WarpCollisionsConflict<THREADS, N_HASHES, START_HASH, N_BITW, START_BITW>(
      n_tileNodes, ceil_nodes, part_offset, shMemRows,
      shMemCols, shMemWorkspace, shMem_collisions);
    
    cg::this_thread_block().sync();

    Smem_ReductionCoop<THREADS, N_HASHES, N_BITW>(shMem_collisions,
                                            ceil_nodes, tiles, blk_i,
                                            coop_total2,
                                            coop_max2);
    cg::this_thread_block().sync();

  } // blk_i

  //write to gloal memory
  const int warp_id = threadIdx.x / 32;
  for (int i = warp_id; i < N_BITW; i += THREADS / 32) {
      if (cub::LaneId() < N_HASHES) {
          const int elem_p_hash_fn = N_BITW * gridDim.x;
          int idx = cub::LaneId() * N_HASHES + i;
          int idxOut = cub::LaneId() * elem_p_hash_fn + i * gridDim.x + blockIdx.x; 
          blocks_total2[idxOut] = coop_total2[idx];
          blocks_max2[idxOut] = coop_max2[idx];
      }
  }

  cg::this_grid().sync();

  // The last block reduces the results of all other blocks

  typedef cub::BlockReduce<CountT, THREADS> BlockReduceT;
  auto& temp_storage =
      reinterpret_cast<typename BlockReduceT::TempStorage&>(shMem);
  if (blockIdx.x == 0) {
    LastReduction<BlockReduceT>(blocks_total2, d_total2, gridDim.x,
                                    cub::Sum{}, temp_storage);
    LastReduction<BlockReduceT>(blocks_max2, d_max2, gridDim.x, cub::Max{},
                                    temp_storage);
  }
}