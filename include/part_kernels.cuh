#pragma once
#include "coloring.cuh"


using namespace apa22_coloring;

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
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
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
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total_collisions, max_collisions);

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
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total_collisions, max_collisions);

    for (int i = 0; i < num_bit_widths; ++i) {
      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = k * i;
        blocks_max1[idx] = k + i;
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
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

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2OnlyLoad(IndexT* row_ptr,  // global mem
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
}


template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2OnlyHashD1(IndexT* row_ptr,  // global mem
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];


  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
  }
}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2OnlyHashD2(IndexT* row_ptr,  // global mem
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total2, max2;
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
  }
}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2OnlyHash(IndexT* row_ptr,  // global mem
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
  }
}


template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2HashWrite(IndexT* row_ptr,  // global mem
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
    for (int i = 0; i < num_bit_widths; ++i) {
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = i * k;
        blocks_max1[idx] = k - i;
        blocks_total2[idx] = i - k;
        blocks_max2[idx] = i + k;
      }
    }
}

template <int THREADS,
          int BLK_SM,
          typename IndexT,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2FirstRed(IndexT* row_ptr,  // global mem
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
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  const int n_tileEdges =
      row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
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
        const int elem_p_hash_fn = num_bit_widths * gridDim.x;
        //        hash segment         bit_w segment   block value
        //        __________________   _____________   ___________
        int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
        blocks_total1[idx] = l1_sum1;
        blocks_max1[idx] = l1_max1;
        blocks_total2[idx] = l1_sum2;
        blocks_max2[idx] = l1_max2;
      }
    }
  }
}



