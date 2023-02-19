#pragma once

#include "tiling.hpp"       //! header file for tiling
#include "coloring_counters.cuh"
#include "hash.cuh"
#include "util.cuh"

namespace apa22_coloring {

enum Distance {
    D1,
    D2,
    D2_SortNet,
    D1_Warp,
    D2_Warp,
};

class Tiling {
 public:
  Tiling(const Tiling&) = delete;
  Tiling& operator=(const Tiling&) = delete;

  Distance dist;
  std::unique_ptr<int[]> tile_boundaries; // array with indices of each tile in all slices
  int n_tiles;
  int tile_target_mem;  // returns how much shmem must be dynamically allocated (in bytes)

  // postprocessing
  int max_node_degree;
  int max_nodes;    // maximum in any node
  int max_edges;    // maximum in any node
  int biggest_tile_nodes;     // maximum in the biggest node
  int biggest_tile_edges;     // maximum in the biggest node

  Tiling(Distance dist,
         int BLK_SM,
         int* row_ptr,
         int m_rows,
         void* kernel,
         int max_smem_SM = -1,        // If not all ShMem shall be used
         bool print = false);
};

Tiling::Tiling(Distance dist,
               int BLK_SM,
               int* row_ptr,
               int m_rows,
               void* kernel,
               int max_smem_SM,
               bool print) : dist(dist) {
  int devId, MaxShmemSizeSM, BlockReserved;
  cudaGetDevice(&devId);
  cudaDeviceGetAttribute(&MaxShmemSizeSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, devId);
  cudaDeviceGetAttribute(&BlockReserved, cudaDevAttrReservedSharedMemoryPerBlock, devId);
  // On Turing 64 kB Shmem hard Cap with 32kB L1.
  // There are other predefined "carveout" levels that might also be an option
  // if more L1 cache seems usefull (on Turing just 32kB).

  // override MaxShmemSizeSM
  if (max_smem_SM != -1) {
    MaxShmemSizeSM = max_smem_SM;
  }

  // to extract the static shared memory allocation of a kernel
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, kernel);
  // Starting with CC 8.0, cuda runtime needs 1KB shMem per block
  int const static_shMem_SM = a.sharedSizeBytes * BLK_SM;
  int const reserved_shMem_SM = BlockReserved * BLK_SM;
  int const max_dyn_SM = MaxShmemSizeSM - static_shMem_SM - reserved_shMem_SM;
  
  // cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn_SM);

  if (print) {
    std::printf("BlockReserved: %d\n", BlockReserved);
    std::printf("MaxShmemSizeSM: %d\n", MaxShmemSizeSM);
    std::printf("Static sharedSizeBytes Block: %ld\n", a.sharedSizeBytes);
    std::printf("Static sharedSizeBytes SM: %ld\n", a.sharedSizeBytes * BLK_SM);
    std::printf("max_dyn_SM: %d\n", max_dyn_SM);
  }

  tile_target_mem = max_dyn_SM/BLK_SM;
  if (dist == D1) {
    auto calc_tile_size = [](int tile_rows, int tile_cols,
                             int max_node_degree) -> int {
      return (tile_cols + tile_rows + 1) * sizeof(int);
    };
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      calc_tile_size,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else if (dist == D2) {
    // We need double the amount of shmem because we need memory for workspace
    auto calc_tile_size = [](int tile_rows, int tile_cols,
                             int max_node_degree) -> int {
      return (tile_cols + tile_rows + 1) * sizeof(int) * 2;
    };
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      calc_tile_size,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else if (dist == D2_SortNet) {
    // we can use only half of the shmem to store the tile because we need
    // the other half to find dist2 collisions and since the
    auto calc_tile_size = [](int tile_rows, int tile_cols,
                             int max_node_degree) -> int {

      int single_node_mem = roundUp(max_node_degree + 1, 32) + 1;
      int additional = single_node_mem * tile_rows;
      return (tile_cols + tile_rows + 1 + additional) * sizeof(int);
    };
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      calc_tile_size,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else if (dist == D1_Warp) {
    const int groups_per_warp = 32 / num_hashes;
    auto calc_tile_size = [=](int tile_rows, int tile_cols,
                             int max_node_degree) -> int {

      int row_counters = roundUp(tile_rows, groups_per_warp);
      int smem_collison_counters = row_counters * num_hashes * num_bit_widths;
      return (tile_cols + tile_rows + 1) * sizeof(int) +
              smem_collison_counters * sizeof(char);
    };
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      calc_tile_size,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else if (dist == D2_Warp) {
    // we can use only half of the shmem to store the tile because we need
    // the other half to find dist2 collisions
    const int groups_per_warp = 32 / num_hashes;

    auto calc_tile_size = [=](int tile_rows, int tile_cols,
                             int max_node_degree) -> int {
      int row_counters = roundUp(tile_rows, groups_per_warp);
      int smem_collison_counters = row_counters * num_hashes * num_bit_widths;

      int single_node_mem = roundUp(max_node_degree, 32) + 1;
      // max_node_degree for column indices and +1 for bank conflict avoidance
      // and at the same time use for row_ptr value (Is part of sorted list).

      // because computed for each hash simultaneously
      single_node_mem *= num_hashes;
      
      int list_space = single_node_mem * tile_rows;

      return (tile_cols + tile_rows + 1 + list_space) * sizeof(int) + 
              smem_collison_counters * sizeof(char);
    };
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      calc_tile_size,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else {
    std::printf("Please pass a valid enum Distance");
  }
  // post processing
  get_MaxTileSize(n_tiles, tile_boundaries.get(), row_ptr,
                  &biggest_tile_nodes, &biggest_tile_edges, &max_nodes, &max_edges);
}

class GPUSetupD1 {
 protected:
  size_t m_rows;
  size_t row_ptr_len;
  size_t col_ptr_len;
  size_t tile_bound_len;
  int n_tiles;
  size_t len_blocks_arr;

 public:
  int* d_row_ptr;
  int* d_col_ptr;
  int* d_tile_boundaries;

  int* blocks_total1;    // data structure for block reduction results
  int* blocks_max1;      // data structure for block reduction results
  Counters* d_total1;
  Counters* d_max1;

  GPUSetupD1(GPUSetupD1 const&) = delete;
  void operator=(GPUSetupD1 const&) = delete;

  GPUSetupD1(int* row_ptr, int* col_ptr, int* tile_boundaries, int n_tiles);
  ~GPUSetupD1();
};

// only constructor
GPUSetupD1::GPUSetupD1(int* row_ptr,
                       int* col_ptr,
                       int* tile_boundaries,
                       int n_tiles) {

  this->n_tiles = n_tiles;
  this->m_rows = tile_boundaries[n_tiles];
  this->row_ptr_len = m_rows + 1;
  this->col_ptr_len = row_ptr[m_rows];
  this->tile_bound_len = n_tiles + 1;

  //==================================================
  // Allocate memory for partitioned matrix on device
  //==================================================
  cudaMalloc((void**)&d_row_ptr, row_ptr_len * sizeof(int));
  cudaMalloc((void**)&d_col_ptr, col_ptr_len * sizeof(int));
  cudaMalloc((void**)&d_tile_boundaries, tile_bound_len * sizeof(int));


  cudaMemcpy(d_row_ptr, row_ptr, row_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ptr, col_ptr, col_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tile_boundaries, tile_boundaries, tile_bound_len * sizeof(int),
             cudaMemcpyHostToDevice);

  //==========================================================
  // Allocate memory for intermediate block reduction results
  //==========================================================
  // For each bit_width we allocate an int counter for each block
  len_blocks_arr = num_bit_widths * sizeof(Counters::value_type) * n_tiles;
  // And we do this for each hash function
  len_blocks_arr *= num_hashes;

  cudaMalloc((void**)&(blocks_total1), len_blocks_arr);
  cudaMalloc((void**)&(blocks_max1), len_blocks_arr);

  //==================================
  // Allocate memory for return values
  //==================================
  cudaMalloc((void**)&d_total1, num_hashes * sizeof(Counters));
  cudaMalloc((void**)&d_max1, num_hashes * sizeof(Counters));
}

GPUSetupD1::~GPUSetupD1(){
  //=================
  // Free device mem
  //=================
  cudaFree(this->d_total1);
  cudaFree(this->d_max1);

  cudaFree(this->blocks_total1);
  cudaFree(this->blocks_max1);

  cudaFree(this->d_row_ptr);
  cudaFree(this->d_col_ptr);
  cudaFree(this->d_tile_boundaries);
}

class GPUSetupD2 : public GPUSetupD1{
 public:
  // additionally for dist2
  int* blocks_total2;
  int* blocks_max2;
  Counters* d_total2;
  Counters* d_max2;

  GPUSetupD2(int* row_ptr, int* col_ptr, int* tile_boundaries, int n_tiles);
  ~GPUSetupD2();
  
  GPUSetupD2(GPUSetupD2 const&) = delete;
  void operator=(GPUSetupD2 const&) = delete;
};
  
GPUSetupD2::GPUSetupD2(int* row_ptr,
                       int* col_ptr,
                       int* tile_boundaries,
                       int n_tiles) :
                       GPUSetupD1(row_ptr, col_ptr, tile_boundaries, n_tiles) {

  cudaMalloc((void**)&(blocks_total2), len_blocks_arr);
  cudaMalloc((void**)&(blocks_max2), len_blocks_arr);

  cudaMalloc((void**)&d_total2, num_hashes * sizeof(Counters));
  cudaMalloc((void**)&d_max2, num_hashes * sizeof(Counters));
}

GPUSetupD2::~GPUSetupD2() {
  //=================
  // Free device mem
  //=================
  cudaFree(this->d_total2);
  cudaFree(this->d_max2);

  cudaFree(this->blocks_total2);
  cudaFree(this->blocks_max2);

  // automatic call to base class destructor
}

} // end apa22_coloring