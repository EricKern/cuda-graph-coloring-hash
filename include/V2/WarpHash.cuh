#include <cub/cub.cuh>

#include "coloring_counters.cuh"

namespace apa22_coloring {

template <typename T>
__device__ __forceinline__ void zero_smem(T* arr, int len) {
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    arr[i] = 0;
  }
}

// template <typename CubReduceT,
//           typename CountT,
//           typename ReductionOp>
// __forceinline__ __device__
// void LastReduction(CountT* blocks_res_ptr, Counters* out_counters, int
// n_tiles,
//                        ReductionOp op,
//                        typename CubReduceT::TempStorage& temp_storage) {
//   const int elem_p_hash_fn = num_bit_widths * n_tiles;
//   // Initial reduction during load in BlockLoadThreadReduce limits thread
//   // local results to at most blockDim.x

//   // Last block reduction
//   #pragma unroll num_hashes
//   for (int k = 0; k < num_hashes; ++k) {
//     int hash_offset = k * elem_p_hash_fn;
//     #pragma unroll num_bit_widths
//     for (int i = 0; i < num_bit_widths; ++i) {
//       CountT* start_addr = blocks_res_ptr + hash_offset + i * n_tiles;

//       CountT accu = BlockLoadThreadReduce(start_addr, n_tiles, op);
//       accu = CubReduceT(temp_storage).Reduce(accu, op);
//       if (threadIdx.x == 0) {
//         out_counters[k].m[i] = accu;
//       }
//       cg::this_thread_block().sync();
//     }
//   }
// }

// template <typename IndexT>
// __forceinline__ __device__
// void Partition2ShMem(IndexT* shMemRows,
//                      IndexT* shMemCols,
//                      IndexT* row_ptr,     // global mem
//                      IndexT* col_ptr,     // global mem
//                      IndexT part_offset,  // tile_boundaries[part_nr]
//                      int n_tileNodes) {
//   int part_size = n_tileNodes + 1;
//   // +1 to load one additional row_ptr value to determine size of last row

//   // put row_ptr partition in shMem
//   for (int i = threadIdx.x; i < part_size; i += blockDim.x) {
//     shMemRows[i] = row_ptr[i + part_offset];
//   }

//   __syncthreads();

//   int num_cols = shMemRows[part_size-1] - shMemRows[0];
//   IndexT col_offset = shMemRows[0];
//   for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
//     shMemCols[i] = col_ptr[i + col_offset];
//   }

//   __syncthreads();
//   // If partition contains n nodes then we now have
//   // n+1 elements of row_ptr in shMem and the col_ptr values for
//   // n nodes
// }

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
          int BLK_SM,
          int N_HASHES = 16,
          int START_HASH = 3,
          typename CountT = int,
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

  constexpr int N_THREAD_GROUPS = THREADS / N_HASHES;
  const int Thread_Grp_ID = threadIdx.x / N_HASHES;
  const int Logic_Lane_ID = threadIdx.x % N_HASHES;
  const int Thread_Hash = Logic_Lane_ID + START_HASH;

  const int partNr = blockIdx.x;
  const IndexT part_offset = tile_boundaries[partNr];
  const int n_tileNodes = tile_boundaries[partNr + 1] - tile_boundaries[partNr];
  const int n_tileEdges = row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

  const int groups_per_warp = 32 / N_HASHES;
  int ceil_nodes = roundUp(n_tileNodes, groups_per_warp);

  typedef cub::BlockReduce<CountT, THREADS> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  extern __shared__ IndexT shMem[];
  IndexT* shMemRows = shMem;  // n_tileNodes +1 elements
  IndexT* shMemCols = &shMemRows[n_tileNodes + 1];
  CountT* shMem_collisions = &shMemCols[n_tileEdges];
  zero_smem(shMem_collisions, ceil_nodes * N_BITW * N_HASHES);
  // syncthreads but we can also sync after Partition2ShMem

  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, part_offset,
                  n_tileNodes);

  // Local Collisions
  for (int i = Thread_Grp_ID; i < n_tileNodes; i += N_THREAD_GROUPS) {
    IndexT glob_row = part_offset + i;
    IndexT row_begin = shMemRows[i];
    IndexT row_end = shMemRows[i + 1];

    const auto row_hash = hash(glob_row, Thread_Hash);
    CountT reg_collisions[N_BITW]{};

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

    #pragma unroll N_BITW
    for (int bit_w = 0; bit_w < N_BITW; ++bit_w) {
      int idx = ceil_nodes * N_HASHES * bit_w +
                 N_HASHES * i +
                 Logic_Lane_ID;
      shMem_collisions[idx] = reg_collisions[bit_w];
    }
  }
  __syncthreads();

  // Reduction
  // Must pad shared memory per bit width to full multiple of groups_per_warp
  const int warp_id = threadIdx.x / 32;
  const int ceil = ceil_nodes * N_HASHES;
  for (int i = warp_id; i < N_BITW; i += THREADS / 32) {
    CountT sum_red = 0;
    CountT max_red = 0;
    CountT* start = &shMem_collisions[ceil_nodes * N_HASHES * i];

    for (int warp_chunk = 0; warp_chunk < ceil;
         warp_chunk += 32) {
      sum_red = cub::Sum{}(sum_red, start[warp_chunk + cub::LaneId()]);
      max_red = cub::Max{}(max_red, start[warp_chunk + cub::LaneId()]);
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

  __threadfence();

  __shared__ bool amLast;
  // Thread 0 takes a ticket
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
    // If the ticket ID is equal to the number of blocks, we are the last
    // block!
    amLast = (ticket == gridDim.x - 1);
  }
  __syncthreads();

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

}  // namespace apa22_coloring