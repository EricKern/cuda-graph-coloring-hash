#pragma once
#include "coloring.cuh"


using namespace apa22_coloring;

// coloring1OnlyLoad
// coloring1OnlyHash        including load
// coloring1LoadWrite       load and write but no hash
// coloring1HashWrite       all above but no reduction
// coloring1FirstReduce     all above but with block reduction

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1OnlyLoad(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                  part_offset, n_tileNodes);
}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1OnlyHash(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
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

    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      // a counter should never have the value of -1 so we should never write
      // just so nothing gets optimized away
      if(total_collisions.m[i] == -1){
        printf("You shouldn't see this in the output");
      }
      if(max_collisions.m[i] == -1){
        printf("You shouldn't see this in the output");
      }
    }
  }
}

// No reduction no hash just load and immediate write
template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1LoadWrite(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr,
                  part_offset, n_tileNodes);
  #pragma unroll num_hashes
  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = total_collisions.m[i];
        blocks_max1[idx] = max_collisions.m[i];
      }
    }
  }
}

// Still no reduction
template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1HashWrite(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
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
    
    #pragma unroll num_bit_widths
    for (int i = 0; i < num_bit_widths; ++i) {
      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = total_collisions.m[i];
        blocks_max1[idx] = max_collisions.m[i];
      }
    }
  }
}

// We rely on guarantees of tiling i.e. constant tile size
// (n_rows + n_cols + 1) = constant, NumTiles = gridDim.x
template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1FirstReduce(IndexT* row_ptr,
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
}