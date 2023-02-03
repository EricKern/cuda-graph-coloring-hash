#include <cub/cub.cuh>

#include "coloring.cuh"
#include "coloring_counters.cuh"

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
void D1LastReduction2(Counters::value_type* block_res, Counters* out_counters, ReductionOp1 op1,
                      ReductionOp2 op2, typename CubReduceT::TempStorage& temp_storage) {
  const int elem_p_hash_fn = num_bit_widths * gridDim.x;
  // Initial reduction during load in BlockLoadThreadReduce limits thread
  // local results to at most blockDim.x

  // Last block reduction
  for (int k = 0; k < num_hashes; ++k) {
    int* start_addr[num_bit_widths];
    int hash_offset = k * elem_p_hash_fn;
    Accu accu{};
    for (int i = 0; i < num_bit_widths; ++i) {
      start_addr[i] = block_res + hash_offset + i * gridDim.x;
      accu.a[i] = BlockLoadThreadReduce(start_addr[i], gridDim.x, op1);
    }
    accu = CubReduceT(temp_storage).Reduce(accu, op2);
    cg::this_thread_block().sync();
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_bit_widths; i++){
        out_counters[k].m[i] = accu.a[i];
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
void coloring1KernelDoubleTemp(IndexT* row_ptr,
                     IndexT* col_ptr,
                     IndexT* tile_boundaries,   // gridDim.x + 1 elements
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  using CountT = Counters::value_type;
  const int partNr = blockIdx.x;
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];
  
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  
  typedef cub::BlockReduce<CountT, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;

  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
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
    D1LastReduction<BlockReduceT>(blocks_total1, d_total, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(blocks_max1, d_max, cub::Max{}, temp_storage2);

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
void coloring2Kernel2Temp(IndexT* row_ptr,  // global mem
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
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
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
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<CountT, THREADS, RED_ALGO> BlockReduceT;

  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
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
    D1LastReduction<BlockReduceT>(blocks_total1, d_total1, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(blocks_max1, d_max1, cub::Max{}, temp_storage2);

    D1LastReduction<BlockReduceT>(blocks_total2, d_total2, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(blocks_max2, d_max2, cub::Max{}, temp_storage2);

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
void coloring2Kernel4Temp(IndexT* row_ptr,  // global mem
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
  const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

  IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
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

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  typedef cub::BlockReduce<CountT, THREADS, RED_ALGO> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage1;
  __shared__ typename BlockReduceT::TempStorage temp_storage2;
  __shared__ typename BlockReduceT::TempStorage temp_storage3;
  __shared__ typename BlockReduceT::TempStorage temp_storage4;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
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
    D1LastReduction<BlockReduceT>(blocks_total1, d_total1, cub::Sum{}, temp_storage1);
    D1LastReduction<BlockReduceT>(blocks_max1, d_max1, cub::Max{}, temp_storage2);

    D1LastReduction<BlockReduceT>(blocks_total2, d_total2, cub::Sum{}, temp_storage3);
    D1LastReduction<BlockReduceT>(blocks_max2, d_max2, cub::Max{}, temp_storage4);

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
void coloring1KernelCustomReduce(IndexT* row_ptr,
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
  
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction<BlockReduceTLast>(blocks_total1, d_total, cub::Sum{}, temp_storage_last);
    D1LastReduction<BlockReduceTLast>(blocks_max1, d_max, cub::Max{}, temp_storage_last);

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
void coloring1KernelCustomReduceLast(IndexT* row_ptr,
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
  
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Accu, THREADS, RED_ALGO> BlockReduceTLast;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  for (int k = 0; k < num_hashes; ++k) {
    // thread local array to count collisions
    Counters total_collisions;
    Counters max_collisions;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction2<BlockReduceTLast>(blocks_total1, d_total, cub::Sum{}, Sum_Last(), temp_storage_last);
    D1LastReduction2<BlockReduceTLast>(blocks_max1, d_max, cub::Max{}, Max_Last(), temp_storage_last);

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
void coloring2KernelCustomReduce(IndexT* row_ptr,  // global mem
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

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements

  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  // n_tileNodes + n_tileEdges + 1 elements
  // Since shMemCols is 32 or 64 bit, allignment for HashT which is smaller
  // is fine.
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<int, THREADS, RED_ALGO> BlockReduceTLast;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
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
          const int elem_p_hash_fn = num_bit_widths * gridDim.x;
          //        hash segment         bit_w segment   block value
          //        __________________   _____________   ___________
          int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
          blocks_total1[idx] = l1_sum1.m[i];
          blocks_max1[idx] = l1_max1.m[i];
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction<BlockReduceTLast>(blocks_total1, d_total1, cub::Sum{}, temp_storage_last);
    D1LastReduction<BlockReduceTLast>(blocks_max1, d_max1, cub::Max{}, temp_storage_last);

    D1LastReduction<BlockReduceTLast>(blocks_total2, d_total2, cub::Sum{}, temp_storage_last);
    D1LastReduction<BlockReduceTLast>(blocks_max2, d_max2, cub::Max{}, temp_storage_last);

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
void coloring2KernelCustomReduceLast(IndexT* row_ptr,  // global mem
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

  // shared mem size: 2 * (n_rows + 1 + n_cols)
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements

  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  // n_tileNodes + n_tileEdges + 1 elements
  // Since shMemCols is 32 or 64 bit, allignment for HashT which is smaller
  // is fine.
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
  typedef cub::BlockReduce<Counters, THREADS, RED_ALGO> BlockReduceT;
  typedef cub::BlockReduce<Accu, THREADS, RED_ALGO> BlockReduceTLast;

  __shared__ typename BlockReduceT::TempStorage temp_storage;


  for (int k = 0; k < num_hashes; ++k) {
    Counters total1, max1;
    Counters total2, max2;
    D1CollisionsLocal(shMemRows, shMemCols, tile_boundaries, start_hash + k,
                      total1, max1);
    D2CollisionsLocal(shMemRows, shMemCols, shMemWorkspace, tile_boundaries,
                      start_hash + k, total2, max2);
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
          const int elem_p_hash_fn = num_bit_widths * gridDim.x;
          //        hash segment         bit_w segment   block value
          //        __________________   _____________   ___________
          int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
          blocks_total1[idx] = l1_sum1.m[i];
          blocks_max1[idx] = l1_max1.m[i];
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

  auto& temp_storage_last = reinterpret_cast<typename BlockReduceTLast::TempStorage&>(temp_storage);
  cg::this_thread_block().sync();

  // The last block reduces the results of all other blocks
  if (amLast) {
    D1LastReduction2<BlockReduceTLast>(blocks_total1, d_total1, cub::Sum{}, Sum_Last(), temp_storage_last);
    D1LastReduction2<BlockReduceTLast>(blocks_max1, d_max1, cub::Max{}, Max_Last(), temp_storage_last);

    D1LastReduction2<BlockReduceTLast>(blocks_total2, d_total2, cub::Sum{}, Sum_Last(), temp_storage_last);
    D1LastReduction2<BlockReduceTLast>(blocks_max2, d_max2, cub::Max{}, Max_Last(), temp_storage_last);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
  }
}
