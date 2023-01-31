#include <cub/cub.cuh>

#include "coloring.cuh"
#include "coloringCounters.cuh"

using namespace apa22_coloring;

struct Accu{
  int a[num_bit_widths];
};

struct Sum_Last {
  __forceinline__ __device__ Accu operator() (const Accu &a, const Accu &b) {
      Accu tmp;
      for (int i = 0; i < num_bit_widths; i++){
        tmp.a[i] = a.a[i] + b.a[i];
      }
      return tmp;
  }
};

struct Max_Last {
  __forceinline__ __device__ Accu operator() (const Accu &a, const Accu &b) {
      Accu tmp;
      for (int i = 0; i < num_bit_widths; i++){
        tmp.a[i] = a.a[i] > b.a[i] ? a.a[i] : b.a[i]; 
      }
      return tmp;
  }
};

template <typename CubReduceT,
          typename ReductionOp1,
          typename ReductionOp2>
__forceinline__ __device__
void D1LastReduction2(SOACounters* soa, Counters* out_counters, ReductionOp1 op1,
                      ReductionOp2 op2, typename CubReduceT::TempStorage& temp_storage) {
  int num_valid = cub::min(blockDim.x, gridDim.x);
  // Initial reduction during load in BlockLoadThreadReduce limits thread
  // local results to at most blockDim.x

  // Last block reduction
  for (int k = 0; k < hash_params.len; ++k) {
    int* start_addr[num_bit_widths];
    Accu accu{};
    for (int i = 0; i < num_bit_widths; ++i) {
      start_addr[i] = soa->m[k] + i * gridDim.x;
      accu.a[i] = BlockLoadThreadReduce(start_addr[i], gridDim.x, op1);
    }
    accu = CubReduceT(temp_storage).Reduce(accu, op2, num_valid);
    cg::this_thread_block().sync();
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_bit_widths; i++){
        out_counters[k].m[i] = accu.a[i];
      }
    }
  }
}


template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1KernelDoubleTemp(IndexT* row_ptr,
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
  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;

  for (int k = 0; k < hash_params.len; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total_collisions, max_collisions);

    for (int i = 0; i < num_bit_widths; ++i) {
      int l1_sum = BlockReduceT(temp_storage1)
                       .Reduce(total_collisions.m[i], cub::Sum{}, n_tileNodes);
      int l1_max = BlockReduceT(temp_storage2)
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
    D1LastReduction<BlockReduceT>(soa_total, d_total, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(soa_max, d_max, cub::Max{}, temp_storage2);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2Kernel2Temp(IndexT* row_ptr,  // global mem
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

  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;


  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    for (int i = 0; i < num_bit_widths; ++i) {
      int l1_sum1 = BlockReduceT(temp_storage1)
                       .Reduce(total1.m[i], cub::Sum{}, n_tileNodes);
      int l1_max1 = BlockReduceT(temp_storage2)
                       .Reduce(max1.m[i], cub::Max{}, n_tileNodes);
      cg::this_thread_block().sync();
      int l1_sum2 = BlockReduceT(temp_storage1)
                       .Reduce(total2.m[i], cub::Sum{}, n_tileNodes);
      int l1_max2 = BlockReduceT(temp_storage2)
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
    D1LastReduction<BlockReduceT>(soa_total1, d_total1, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(soa_max1, d_max1, cub::Max{}, temp_storage2);

    D1LastReduction<BlockReduceT>(soa_total2, d_total2, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(soa_max2, d_max2, cub::Max{}, temp_storage2);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}

template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2Kernel4Temp(IndexT* row_ptr,  // global mem
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

  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;
  __shared__ typename BlockReduceT::TempStorage temp_storage3;
  __shared__ typename BlockReduceT::TempStorage temp_storage4;


  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    for (int i = 0; i < num_bit_widths; ++i) {
      int l1_sum1 = BlockReduceT(temp_storage1)
                       .Reduce(total1.m[i], cub::Sum{}, n_tileNodes);
      int l1_max1 = BlockReduceT(temp_storage2)
                       .Reduce(max1.m[i], cub::Max{}, n_tileNodes);
      int l1_sum2 = BlockReduceT(temp_storage3)
                       .Reduce(total2.m[i], cub::Sum{}, n_tileNodes);
      int l1_max2 = BlockReduceT(temp_storage4)
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
    D1LastReduction<BlockReduceT>(soa_total1, d_total1, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(soa_max1, d_max1, cub::Max{}, temp_storage2);

    D1LastReduction<BlockReduceT>(soa_total2, d_total2, cub::Sum{}, temp_storage3);
    D1LastReduction<BlockReduceT>(soa_max2, d_max2, cub::Max{}, temp_storage4);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }

}


template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1KernelCustomReduce(IndexT* row_ptr,
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
  
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  for (int k = 0; k < hash_params.len; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total_collisions, max_collisions);

    auto l1_sum = BlockReduceT(temp_storage)
                       .Reduce(total_collisions, Sum_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_max = BlockReduceT(temp_storage)
                       .Reduce(max_collisions, Max_Counters(), n_tileNodes);
    cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
    if(threadIdx.x == 0){
      for (int i = 0; i < num_bit_widths; ++i) {
        soa_total->m[k][partNr + i * gridDim.x] = l1_sum.m[i];
        soa_max->m[k][partNr + i * gridDim.x] = l1_max.m[i];
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction<BlockReduceTLast>(soa_total, d_total, cub::Sum{}, temp_storage_last);
    D1LastReduction<BlockReduceTLast>(soa_max, d_max, cub::Max{}, temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring1KernelCustomReduceLast(IndexT* row_ptr,
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
  
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Accu, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  for (int k = 0; k < hash_params.len; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total_collisions, max_collisions);

    auto l1_sum = BlockReduceT(temp_storage)
                       .Reduce(total_collisions, Sum_Counters(), n_tileNodes);
    auto l1_max = BlockReduceT(temp_storage)
                       .Reduce(max_collisions, Max_Counters(), n_tileNodes);
    cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
    if(threadIdx.x == 0){
      for (int i = 0; i < num_bit_widths; ++i) {
        soa_total->m[k][partNr + i * gridDim.x] = l1_sum.m[i];
        soa_max->m[k][partNr + i * gridDim.x] = l1_max.m[i];
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction2<BlockReduceTLast>(soa_total, d_total, cub::Sum{}, Sum_Last(), temp_storage_last);
    D1LastReduction2<BlockReduceTLast>(soa_max, d_max, cub::Max{}, Max_Last(), temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2KernelCustomReduce(IndexT* row_ptr,  // global mem
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
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceTLast;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    auto l1_sum1 = BlockReduceT(temp_storage)
                     .Reduce(total1, Sum_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_max1 = BlockReduceT(temp_storage)
                     .Reduce(max1, Max_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_sum2 = BlockReduceT(temp_storage)
                     .Reduce(total2, Sum_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_max2 = BlockReduceT(temp_storage)
                     .Reduce(max2, Max_Counters(), n_tileNodes);
    cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        for (int i = 0; i < num_bit_widths; ++i) {
          soa_total1->m[k][partNr + i * gridDim.x] = l1_sum1.m[i];
          soa_max1->m[k][partNr + i * gridDim.x] = l1_max1.m[i];
          soa_total2->m[k][partNr + i * gridDim.x] = l1_sum2.m[i];
          soa_max2->m[k][partNr + i * gridDim.x] = l1_max2.m[i];
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction<BlockReduceTLast>(soa_total1, d_total1, cub::Sum{}, temp_storage_last);
    D1LastReduction<BlockReduceTLast>(soa_max1, d_max1, cub::Max{}, temp_storage_last);

    D1LastReduction<BlockReduceTLast>(soa_total2, d_total2, cub::Sum{}, temp_storage_last);
    D1LastReduction<BlockReduceTLast>(soa_max2, d_max2, cub::Max{}, temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}

template <typename IndexT,
          int THREADS,
          int BLK_SM,
          cub::BlockReduceAlgorithm RED_ALGO=cub::BLOCK_REDUCE_WARP_REDUCTIONS>
__global__
__launch_bounds__(THREADS, BLK_SM)
void coloring2KernelCustomReduceLast(IndexT* row_ptr,  // global mem
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
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Accu, THREADS, RED_ALGO> BlockReduceTLast;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < hash_params.len; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, hash_params.val[k],
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      hash_params.val[k], total2, max2);
    auto l1_sum1 = BlockReduceT(temp_storage)
                     .Reduce(total1, Sum_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_max1 = BlockReduceT(temp_storage)
                     .Reduce(max1, Max_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_sum2 = BlockReduceT(temp_storage)
                     .Reduce(total2, Sum_Counters(), n_tileNodes);
    cg::this_thread_block().sync();
    auto l1_max2 = BlockReduceT(temp_storage)
                     .Reduce(max2, Max_Counters(), n_tileNodes);
    cg::this_thread_block().sync();

      // soa has an int array for each hash function.
      // In this array all block reduce results of first counter are stored
      // contiguous followed by the reduce results of the next counter ...
      if(threadIdx.x == 0){
        for (int i = 0; i < num_bit_widths; ++i) {
          soa_total1->m[k][partNr + i * gridDim.x] = l1_sum1.m[i];
          soa_max1->m[k][partNr + i * gridDim.x] = l1_max1.m[i];
          soa_total2->m[k][partNr + i * gridDim.x] = l1_sum2.m[i];
          soa_max2->m[k][partNr + i * gridDim.x] = l1_max2.m[i];
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction2<BlockReduceTLast>(soa_total1, d_total1, cub::Sum{}, Sum_Last(), temp_storage_last);
    D1LastReduction2<BlockReduceTLast>(soa_max1, d_max1, cub::Max{}, Max_Last(), temp_storage_last);

    D1LastReduction2<BlockReduceTLast>(soa_total2, d_total2, cub::Sum{}, Sum_Last(), temp_storage_last);
    D1LastReduction2<BlockReduceTLast>(soa_max2, d_max2, cub::Max{}, Max_Last(), temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}
