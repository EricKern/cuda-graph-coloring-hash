#pragma once

#include <coloring.cuh>

template <typename IndexT>
__global__
void coloring1OnlyLoad(IndexT* row_ptr,  // global mem
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
  

}

template <typename IndexT>
__global__
void coloring1NoReduceNoWrite(IndexT* row_ptr,  // global mem
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
  
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  for (int k = 0; k < hash_params.len; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total_collisions, max_collisions);

  }
}

template <typename IndexT>
__global__
void coloring1NoReduce(IndexT* row_ptr,  // global mem
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
  
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  for (int k = 0; k < hash_params.len; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total_collisions, max_collisions);

    for (int i = 0; i < num_bit_widths; ++i) {
      if(threadIdx.x == 0){
        soa_total->m[k][partNr + i * gridDim.x] = k + i;
        soa_max->m[k][partNr + i * gridDim.x] = k * i;
      }
    }
  }
}

template <typename IndexT>
__global__
void coloring1NoLastReduce(IndexT* row_ptr,  // global mem
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
  
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

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

}

template <typename IndexT>
__global__
void coloring2OnlyLoad(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexT* shMemWorkspace = &shMemCols[tile_max_edges]; // tile_max_nodes + tile_max_edges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

}

template <typename IndexT>
__global__
void coloring2NoReduceNoWriteOnlyD1(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexT* shMemWorkspace = &shMemCols[tile_max_edges]; // tile_max_nodes + tile_max_edges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const IndexT partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    }
  
}

template <typename IndexT>
__global__
void coloring2NoReduceNoWriteOnlyD2(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexT* shMemWorkspace = &shMemCols[tile_max_edges]; // tile_max_nodes + tile_max_edges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const IndexT partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    }
  
}

template <typename IndexT>
__global__
void coloring2NoReduceNoWrite(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexT* shMemWorkspace = &shMemCols[tile_max_edges]; // tile_max_nodes + tile_max_edges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const IndexT partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    }
  
}

template <typename IndexT>
__global__
void coloring2NoReduce(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexT* shMemWorkspace = &shMemCols[tile_max_edges]; // tile_max_nodes + tile_max_edges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const IndexT partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    for (int i = 0; i < num_bit_widths; ++i) {
      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        soa_total1->m[k][partNr + i * gridDim.x] = k * i;
        soa_max1->m[k][partNr + i * gridDim.x] = k + i;
        soa_total2->m[k][partNr + i * gridDim.x] = k - i;
        soa_max2->m[k][partNr + i * gridDim.x] = i - k;
      }
    }
  }
}


template <typename IndexT>
__global__
void coloring2NoLastReduce(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // tile_max_nodes +1 elements
  IndexT* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  IndexT* shMemWorkspace = &shMemCols[tile_max_edges]; // tile_max_nodes + tile_max_edges + 1 elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  const IndexT partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

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
}