#include <cub/cub.cuh>

#include "coloring_counters.cuh"

namespace apa22_coloring {

template <typename T>
__device__ __forceinline__ void zero_smem(T* arr, int len) {
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    arr[i] = 0;
  }
}

template<int THREADS,
         int N_HASHES,
         int START_HASH,
         int N_BITW,
         int START_BITW,
         typename SCountT,
         typename IndexT>
__forceinline__ __device__
void D1WarpCollisions(const int n_tileNodes,
                      const int ceil_nodes,
                      const IndexT part_offset,
                      const IndexT* shMemRows,
                      const IndexT* shMemCols,
                      SCountT* shMem_collisions) {
  constexpr int N_THREAD_GROUPS = THREADS / N_HASHES;
  const int Thread_Grp_ID = threadIdx.x / N_HASHES;
  const int Logic_Lane_ID = threadIdx.x % N_HASHES;
  const int Thread_Hash = Logic_Lane_ID + START_HASH;

  // Local Collisions
  for (int i = Thread_Grp_ID; i < n_tileNodes; i += N_THREAD_GROUPS) {
    IndexT glob_row = part_offset + i;
    IndexT row_begin = shMemRows[i];
    IndexT row_end = shMemRows[i + 1];

    const auto row_hash = hash(glob_row, Thread_Hash);
    SCountT reg_collisions[N_BITW]{};

    for (IndexT col_idx = row_begin; col_idx < row_end; ++col_idx) {
      IndexT col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if (__builtin_expect(col != glob_row, true)) {
        const auto col_hash = hash(col, Thread_Hash);

        #pragma unroll N_BITW
        for (int counter_idx = 0; counter_idx < N_BITW; ++counter_idx) {
          int shift_val = START_BITW + counter_idx;

          std::uint32_t mask = (1u << shift_val) - 1u;
          if ((row_hash & mask) == (col_hash & mask)) {
            reg_collisions[counter_idx] += 1;
          }
          // else {
          //   // if hashes differ in lower bits they also differ when
          //   increasing
          //   // the bit_width
          //   break;
          // }
        }
      }
    }

    // Write register collisions to smem
    #pragma unroll N_BITW
    for (int bit_w = 0; bit_w < N_BITW; ++bit_w) {
      int idx = ceil_nodes * N_HASHES * bit_w +
                 N_HASHES * i +
                 Logic_Lane_ID;
      shMem_collisions[idx] = reg_collisions[bit_w];
    }
  }
}

template <int N_HASHES, typename T, typename red_op>
__forceinline__ __device__ void ShflGroupsDown(T& elem, red_op binary_op) {
  constexpr int groups_per_warp = 32 / N_HASHES;
  for (int i = groups_per_warp / 2; i >= 1; i >>= 1) {
    int offset = i * N_HASHES;
    auto peer_elem = cub::ShuffleDown<32>(elem, offset, 31, 0xffffffff);
    elem = binary_op(peer_elem, elem);
  }
}

template <int THREADS,
          int N_HASHES,
          int N_BITW,
          typename CountT,
          typename SCountT>
__forceinline__ __device__
void Smem_Reduction(SCountT* shMem_collisions,
                    int ceil_nodes,
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
      const int elem_p_hash_fn = N_BITW * gridDim.x;
      //        hash segment                     bit_w segment   block value
      //        ______________________________   _____________   ___________
      int idx = cub::LaneId() * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
      blocks_total1[idx] = sum_red;
      blocks_max1[idx] = max_red;
    }
  }
}

template <int THREADS,
          typename CountT>
__forceinline__ __device__
void deviceReduction(CountT* blocks_total1,
                     CountT* blocks_max1,
                     Counters* d_total,
                     Counters* d_max) {
  __shared__ bool amLast;
  // Thread 0 takes a ticket
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
    // If the ticket ID is equal to the number of blocks, we are the last
    // block!
    amLast = (ticket == gridDim.x - 1);
  }
  __syncthreads();

    typedef cub::BlockReduce<CountT, THREADS> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  // The last block reduces the results of all other blocks
  if (amLast) {
    LastReduction<BlockReduceT>(blocks_total1, d_total, gridDim.x, cub::Sum{},
                                temp_storage);
    LastReduction<BlockReduceT>(blocks_max1, d_max, gridDim.x, cub::Max{},
                                temp_storage);

    if (threadIdx.x == 0) {
      // reset retirement count so that next run succeeds
      retirementCount = 0;
    }
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
__global__ __launch_bounds__(THREADS, BLK_SM)
void D1warp(IndexT* row_ptr,
            IndexT* col_ptr,
            IndexT* tile_boundaries,  // gridDim.x + 1 elements
            CountT* blocks_total1,
            CountT* blocks_max1,
            Counters* d_total,
            Counters* d_max) {
  static_assert(cub::PowerOfTwo<N_HASHES>::VALUE, "N_HASHES must be a power of 2");
  static_assert(N_HASHES <= 32, "N_HASHES must be smaller than 32");
  static_assert(THREADS % 32 == 0, "Threads must be a multiple of 32");
  static_assert(std::is_integral_v<CountT>, "Counters must be of integral type");
  static_assert(std::is_integral_v<SCountT>, "Shorter Counters must be of integral type");
  static_assert(alignof(SCountT) <= alignof(IndexT),
            "Smaller Counter is not small enough. Alignment problem in Smem.");



  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr + 1] - tile_boundaries[partNr];
  const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  const int groups_per_warp = 32 / N_HASHES;
  int ceil_nodes = roundUp(n_tileNodes, groups_per_warp);

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;  // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes + 1];
  SCountT* shMem_collisions = reinterpret_cast<SCountT*>(&shMemCols[n_tileEdges]);
  zero_smem(shMem_collisions, ceil_nodes * N_BITW * N_HASHES);
  // syncthreads but we can also sync after Partition2ShMem

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, part_offset,
                  n_tileNodes);

  D1WarpCollisions<THREADS, N_HASHES, START_HASH, N_BITW, START_BITW, SCountT>(
      n_tileNodes, ceil_nodes, part_offset, shMemRows, shMemCols,
      shMem_collisions);
  __syncthreads();

  Smem_Reduction<THREADS, N_HASHES, N_BITW>(shMem_collisions,
                                            ceil_nodes,
                                            blocks_total1,
                                            blocks_max1);

  __threadfence();
  deviceReduction<THREADS>(blocks_total1,
                           blocks_max1,
                           d_total,
                           d_max);
  
}



template<int THREADS,
         int N_HASHES,
         int START_HASH,
         int N_BITW,
         int START_BITW,
         typename SCountT,
         typename IndexT,
         typename HashT>
__forceinline__ __device__
void D2WarpCollisions(const int n_tileNodes,
                      const int ceil_nodes,
                      const int max_node_degree,
                      const IndexT part_offset,
                      const IndexT* shMemRows,
                      const IndexT* shMemCols,
                      HashT* sMem_workspace,
                      SCountT* shMem_collisions) {
  constexpr int N_THREAD_GROUPS = THREADS / N_HASHES;
  const int Thread_Grp_ID = threadIdx.x / N_HASHES;
  const int Logic_Lane_ID = threadIdx.x % N_HASHES;
  const int Thread_Hash = Logic_Lane_ID + START_HASH;

  const int thread_ws_elems = roundUp(max_node_degree, 32) + 1;
  const int group_ws_size = thread_ws_elems * N_HASHES;

  // Local Collisions
  for (int i = Thread_Grp_ID; i < n_tileNodes; i += N_THREAD_GROUPS) {
    IndexT glob_row = part_offset + i;
    IndexT row_begin = shMemRows[i];
    IndexT row_end = shMemRows[i + 1];

    const int group_ws_begin = i * group_ws_size;
    HashT* thread_ws = sMem_workspace + group_ws_begin +
                       Logic_Lane_ID * thread_ws_elems;

    thread_ws[0] = __brev(hash(glob_row, Thread_Hash));
    
    int j = 1;
    for (IndexT col_idx = row_begin; col_idx < row_end; ++col_idx) {
      IndexT col = shMemCols[col_idx - shMemRows[0]];
      // col_idx - shMemRows[0] transforms global col_idx to shMem index
      if (__builtin_expect(col != glob_row, true)) {
        thread_ws[j] = __brev(hash(col, Thread_Hash));
        ++j;
      }
    }
    for (int k = j; k < thread_ws_elems; ++k) {
      thread_ws[k] = cub::Traits<HashT>::MAX_KEY;
      // Max value of uint32 so that padding elements are at the very end of 
      // the sorted bitreversed array
    }

    // sorting network to avoid bank conflicts
    odd_even_merge_sort(thread_ws, thread_ws_elems);
    // odd_even_merge_sort(thread_ws, j);
    // insertionSort(thread_ws, j);


    #pragma unroll N_BITW
    for(int counter_idx = 0; counter_idx < N_BITW; ++counter_idx){
      auto shift_val = START_BITW + counter_idx;

      uint32_t mask = (1u << shift_val) - 1u;
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
      

      const int idx = ceil_nodes * N_HASHES * counter_idx +
                      N_HASHES * i +
                      Logic_Lane_ID;
      shMem_collisions[idx] = max_so_far - 1;
    }
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
void D2warp(IndexT* row_ptr,  // global mem
                         IndexT* col_ptr,  // global mem
                         IndexT* tile_boundaries,
                         int max_node_degree,
                         Counters::value_type* blocks_total2,
                         Counters::value_type* blocks_max2,
                         Counters* d_total2,
                         Counters* d_max2) {
  using HashT = std::uint32_t;
  static_assert(cub::PowerOfTwo<N_HASHES>::VALUE, "N_HASHES must be a power of 2");
  static_assert(N_HASHES <= 32, "N_HASHES must be smaller than 32");
  static_assert(THREADS % 32 == 0, "Threads must be a multiple of 32");
  static_assert(std::is_integral_v<CountT>, "Counters must be of integral type");
  static_assert(std::is_integral_v<SCountT>, "Shorter Counters must be of integral type");
  static_assert(alignof(SCountT) <= alignof(CountT), "Smaller Counter is not actually smaller");


  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr + 1] - tile_boundaries[partNr];
  const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  const int groups_per_warp = 32 / N_HASHES;
  int ceil_nodes = roundUp(n_tileNodes, groups_per_warp); // tiling knows this

  // tiling knows this as well
  const int single_node_list_elems = (roundUp(max_node_degree, 32) + 1) * N_HASHES;
  const int list_space = single_node_list_elems * n_tileNodes;

  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;  // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes + 1];
  HashT* shMemWorkspace = reinterpret_cast<HashT*>(&shMemCols[n_tileEdges]);
  SCountT* shMem_collisions = reinterpret_cast<SCountT*>(&shMemWorkspace[list_space]);
  zero_smem(shMem_collisions, ceil_nodes * N_BITW * N_HASHES);
  // syncthreads but we can also sync after Partition2ShMem

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, part_offset,
                  n_tileNodes);

  D2WarpCollisions<THREADS, N_HASHES, START_HASH, N_BITW, START_BITW>(
      n_tileNodes, ceil_nodes, max_node_degree, part_offset, shMemRows,
      shMemCols, shMemWorkspace, shMem_collisions);

  __syncthreads();

  Smem_Reduction<THREADS, N_HASHES, N_BITW>(shMem_collisions,
                                            ceil_nodes,
                                            blocks_total2,
                                            blocks_max2);

  __threadfence();
  deviceReduction<THREADS>(blocks_total2,
                           blocks_max2,
                           d_total2,
                           d_max2);
  
}

}  // namespace apa22_coloring