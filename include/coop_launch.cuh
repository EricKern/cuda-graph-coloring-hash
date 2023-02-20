#pragma once
#include "coloring.cuh"
#include "d2_opt_kernel.cuh"

namespace apa22_coloring {

namespace cg = cooperative_groups;

__host__
int get_coop_grid_size(void* kernel, int threads, size_t shMem_bytes,
                       bool print=false) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel,
                                                threads, shMem_bytes);
  if (print) {
    std::printf("numBlocksPerSm: %d\n", numBlocksPerSm);
    std::printf("gridSize.x: %d\n", deviceProp.multiProcessorCount*numBlocksPerSm);
  }
  return deviceProp.multiProcessorCount*numBlocksPerSm;
}

__device__ __forceinline__
void init_counters(Counters* arr1, Counters* arr2) {
  for (int i = threadIdx.x; i < num_hashes * num_bit_widths; i += blockDim.x) {
    int struct_nr = i / num_bit_widths;
    int elem_nr = i % num_bit_widths;
    arr1[struct_nr].m[elem_nr] = 0;
    arr2[struct_nr].m[elem_nr] = 0;
  }
  cg::this_thread_block().sync();
}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1coop(IndexT* row_ptr,
                   IndexT* col_ptr,
                   IndexT* tile_boundaries,   // gridDim.x + 1 elements
                   int tiles,
                   Counters::value_type* blocks_total1,
                   Counters::value_type* blocks_max1,
                   Counters* d_total,
                   Counters* d_max) {
  using CountT = Counters::value_type;

  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  __shared__ Counters coop_total1[num_hashes];
  __shared__ Counters coop_max1[num_hashes];
  init_counters(coop_total1, coop_max1);

  extern __shared__ IndexT shMem[];

  for(int blk_i = blockIdx.x; blk_i < tiles; blk_i += gridDim.x) {
    const int partNr = blk_i;
    const IndexT part_offset = tile_boundaries[partNr];
    const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

    IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
    IndexT* shMemCols = &shMemRows[n_tileNodes+1];

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                    part_offset, n_tileNodes);

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
          coop_total1[k].m[i] = cub::Sum{}(l1_sum.m[i], coop_total1[k].m[i]);
          coop_max1[k].m[i] = cub::Max{}(l1_max.m[i], coop_max1[k].m[i]);
        }
      }
    }
  }
  if(threadIdx.x == 0) {
    #pragma unroll num_hashes
    for (int k = 0; k < num_hashes; ++k) {
      #pragma unroll num_bit_widths
      for (int i = 0; i < num_bit_widths; ++i) {
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = coop_total1[k].m[i];
        blocks_max1[idx] = coop_max1[k].m[i];
      }
    }
  }
  cg::this_grid().sync();

  auto& temp_storage_last =
      reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  // The last block reduces the results of all other blocks
  if (blockIdx.x == 0) {
    LastReduction<BlockReduceTLast>(blocks_total1, d_total, gridDim.x, cub::Sum{},
                                temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                temp_storage_last);
  }
}


template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2coop(IndexT* row_ptr,  // global mem
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
  using CountT = Counters::value_type;

  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  __shared__ Counters coop_total2[num_hashes];
  __shared__ Counters coop_max2[num_hashes];
  init_counters(coop_total2, coop_max2);

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  // Since shMemCols is 32 or 64 bit, allignment for HashT which is smaller
  // is fine. Tiling guarantees that shMemWorkspace is enough

  for(int blk_i = blockIdx.x; blk_i < tiles; blk_i += gridDim.x) {
    const int partNr = blk_i;
    const IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
    const int n_tileNodes = tile_boundaries[partNr+1] - part_offset;
    const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

    IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
    HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                    part_offset, n_tileNodes);

    #pragma unroll num_hashes
    for (int k = 0; k < num_hashes; ++k) {
      // Counters total1, max1;
      Counters total2, max2;
      // D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
      //                   total1, max1);
      D2CollisionsLocalSNetSmall(shMemRows, shMemCols, shMemWorkspace, n_tileNodes,
                      part_offset, start_hash + k, total2, max2);
      // CountT l1_sum1 = BlockReduceT(temp_storage)
      //                  .Reduce(total1.m[i], cub::Sum{});
      // cg::this_thread_block().sync();
      // CountT l1_max1 = BlockReduceT(temp_storage)
      //                  .Reduce(max1.m[i], cub::Max{});
      // cg::this_thread_block().sync();
      Counters l1_sum2 = BlockReduceT(temp_storage)
                      .Reduce(total2, Sum_Counters{});
      cg::this_thread_block().sync();
      Counters l1_max2 = BlockReduceT(temp_storage)
                      .Reduce(max2, Max_Counters{});
      cg::this_thread_block().sync();
      #pragma unroll num_bit_widths
      for (int i = 0; i < num_bit_widths; ++i) {
        if(threadIdx.x == 0){
          // blocks_total1[idx] = l1_sum1;
          // blocks_max1[idx] = l1_max1;
          coop_total2[k].m[i] = cub::Sum{}(l1_sum2.m[i], coop_total2[k].m[i]);
          coop_max2[k].m[i] = cub::Max{}(l1_max2.m[i], coop_max2[k].m[i]);
        }
      } // bit_w
    } // hash_nr
  } // blk_i

  if(threadIdx.x == 0){
    #pragma unroll num_hashes
    for (int k = 0; k < num_hashes; ++k) {
      #pragma unroll num_bit_widths
      for (int i = 0; i < num_bit_widths; ++i) {
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;

        blocks_total2[idx] = coop_total2[k].m[i];
        blocks_max2[idx] = coop_max2[k].m[i];
      }
    }
  }
  cg::this_grid().sync();

  // The last block reduces the results of all other blocks

  auto& temp_storage_last =
      reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  if (blockIdx.x == 0) {
    LastReduction<BlockReduceTLast>(blocks_total2, d_total2, gridDim.x,
                                    cub::Sum{}, temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max2, d_max2, gridDim.x, cub::Max{},
                                    temp_storage_last);
  }
}

// =====================================================
// Base version built to Coop Launchs
// =====================================================

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1coopBase(IndexT* row_ptr,
                   IndexT* col_ptr,
                   IndexT* tile_boundaries,   // gridDim.x + 1 elements
                   int tiles,
                   Counters::value_type* blocks_total1,
                   Counters::value_type* blocks_max1,
                   Counters* d_total,
                   Counters* d_max) {
  using CountT = Counters::value_type;

  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  extern __shared__ IndexT shMem[];

  for(int blk_i = blockIdx.x; blk_i < tiles; blk_i += gridDim.x) {
    const int partNr = blk_i;
    const IndexT part_offset = tile_boundaries[partNr];
    const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

    IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
    IndexT* shMemCols = &shMemRows[n_tileNodes+1];

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                    part_offset, n_tileNodes);

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
      if(threadIdx.x == 0) {
        #pragma unroll num_bit_widths
        for (int i = 0; i < num_bit_widths; ++i) {
          const int elem_p_hash_fn = num_bit_widths * tiles;
          //        hash segment         bit_w segment   block value
          //        __________________   _____________   ___________
          int idx = k * elem_p_hash_fn + i * tiles     + blk_i;
          blocks_total1[idx] = l1_sum.m[i];
          blocks_max1[idx] = l1_max.m[i];
        }
      }
    }
  }
  cg::this_grid().sync();

  auto& temp_storage_last =
      reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  // The last block reduces the results of all other blocks
  if (blockIdx.x == 0) {
    LastReduction<BlockReduceTLast>(blocks_total1, d_total, tiles, cub::Sum{},
                                temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max1, d_max, tiles, cub::Max{},
                                temp_storage_last);
  }
}


template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2coopBase(IndexT* row_ptr,  // global mem
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
  using CountT = Counters::value_type;

  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Counters::value_type, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  // Since shMemCols is 32 or 64 bit, allignment for HashT which is smaller
  // is fine. Tiling guarantees that shMemWorkspace is enough

  for(int blk_i = blockIdx.x; blk_i < tiles; blk_i += gridDim.x) {
    const int partNr = blk_i;
    const IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
    const int n_tileNodes = tile_boundaries[partNr+1] - part_offset;
    const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

    IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
    HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                    part_offset, n_tileNodes);

    #pragma unroll num_hashes
    for (int k = 0; k < num_hashes; ++k) {
      // Counters total1, max1;
      Counters total2, max2;
      // D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
      //                   total1, max1);
      D2CollisionsLocalSNetSmall(shMemRows, shMemCols, shMemWorkspace, n_tileNodes,
                      part_offset, start_hash + k, total2, max2);
        // CountT l1_sum1 = BlockReduceT(temp_storage)
        //                  .Reduce(total1.m[i], cub::Sum{});
        // cg::this_thread_block().sync();
        // CountT l1_max1 = BlockReduceT(temp_storage)
        //                  .Reduce(max1.m[i], cub::Max{});
        // cg::this_thread_block().sync();
      Counters l1_sum2 = BlockReduceT(temp_storage)
                      .Reduce(total2, Sum_Counters{});
      cg::this_thread_block().sync();
      Counters l1_max2 = BlockReduceT(temp_storage)
                      .Reduce(max2, Max_Counters{});
      cg::this_thread_block().sync();

      if(threadIdx.x == 0){
        #pragma unroll num_bit_widths
        for (int i = 0; i < num_bit_widths; ++i) {
          const int elem_p_hash_fn = num_bit_widths * tiles;
          //        hash segment         bit_w segment   block value
          //        __________________   _____________   ___________
          int idx = k * elem_p_hash_fn + i * tiles + blk_i;

          blocks_total2[idx] = l1_sum2.m[i];
          blocks_max2[idx] = l1_max2.m[i];
        }
      }
    } // hash_nr
  } // blk_i

  cg::this_grid().sync();

  // The last block reduces the results of all other blocks

  auto& temp_storage_last =
      reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  if (blockIdx.x == 0) {
    LastReduction<BlockReduceTLast>(blocks_total2, d_total2, tiles,
                                    cub::Sum{}, temp_storage_last);
    LastReduction<BlockReduceTLast>(blocks_max2, d_max2, tiles, cub::Max{},
                                    temp_storage_last);
  }
}

} // end apa22_coloring