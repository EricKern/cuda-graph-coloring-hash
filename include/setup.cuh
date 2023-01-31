#pragma once

#include <coloringCounters.cuh>
#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling

#include <coloring.cuh>
#include <defines.hpp>

namespace apa22_coloring {

enum Distance {
    D1,
    D2,
    D2_SortNet
};

// Singleton class so we load matrix only once
// consequentially we can only have only one matrix loaded at a time
class MatLoader {
 public:
  static MatLoader& getInstance(const char* path = def::Mat2) {
    static MatLoader Instance(path);

    return Instance;
  }

  MatLoader(const MatLoader&) = delete;
  MatLoader& operator=(const MatLoader&) = delete;
  
  const char* path;
  int* row_ptr;
  int* col_ptr;
  double* val_ptr;
  int m_rows;

 private:
  MatLoader(const char* path) : path(path) {
    m_rows = cpumultiplyDloadMTX(path, &row_ptr, &col_ptr, &val_ptr);
  }
  ~MatLoader() {
    delete[] row_ptr;
    delete[] col_ptr;
    delete[] val_ptr;
  }
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
         bool bank_conflict_free = false,
         bool print = false);
};

Tiling::Tiling(Distance dist,
               int BLK_SM,
               int* row_ptr,
               int m_rows,
               void* kernel,
               int max_smem_SM,
               bool bank_conflict_free,
               bool print) : dist(dist) {
  int devId, MaxShmemSizeSM;
  cudaGetDevice(&devId);
  cudaDeviceGetAttribute(&MaxShmemSizeSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, devId);
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
  int const max_dyn_SM = MaxShmemSizeSM - static_shMem_SM;
  
  // cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn_SM);

  if (print) {
    std::printf("MaxShmemSizeSM: %d\n", MaxShmemSizeSM);
    std::printf("Static sharedSizeBytes Block: %ld\n", a.sharedSizeBytes);
    std::printf("Static sharedSizeBytes SM: %ld\n", a.sharedSizeBytes * BLK_SM);
    std::printf("max_dyn_SM: %d\n", max_dyn_SM);
  }

  tile_target_mem = max_dyn_SM/BLK_SM;
  if (dist == D1) {
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else if (dist == D2) {
    // we can use only half of the shmem to store the tile because we need
    // the other half to find dist2 collisions
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem/2,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree);
  } else if (dist == D2_SortNet) {
    // we can use only half of the shmem to store the tile because we need
    // the other half to find dist2 collisions and since the 
    very_simple_tiling(row_ptr,
                      m_rows,
                      tile_target_mem,
                      &tile_boundaries,
                      &n_tiles,
                      &max_node_degree,
                      true);
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

 public:

  //
  int* d_row_ptr;
  int* d_col_ptr;
  int* d_tile_boundaries;
  SOACounters* d_soa_total1;
  SOACounters* d_soa_max1;
  SOACounters h_soa_total1;
  SOACounters h_soa_max1;
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
  // For each bit_width we allocate a counter for each block and for each hash function

  int len_blocks_arr = num_bit_widths * n_tiles * sizeof(int);

  for (int i = 0; i < hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_total1.m[i]), len_blocks_arr);
  }
  for (int i = 0; i < hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_max1.m[i]), len_blocks_arr);
  }
  //==========================================================================
  // Allocate memory for structs holding pointers to intermediate result alloc
  //==========================================================================
  cudaMalloc((void**)&d_soa_total1, sizeof(SOACounters));
  cudaMalloc((void**)&d_soa_max1, sizeof(SOACounters));

  cudaMemcpy(d_soa_total1, &h_soa_total1, sizeof(SOACounters),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_soa_max1, &h_soa_max1, sizeof(SOACounters),
             cudaMemcpyHostToDevice);

  //==================================
  // Allocate memory for return values
  //==================================
  cudaMalloc((void**)&d_total1, hash_params.len * sizeof(Counters));
  cudaMalloc((void**)&d_max1, hash_params.len * sizeof(Counters));
}

GPUSetupD1::~GPUSetupD1(){
  //=================
  // Free device mem
  //=================
  cudaFree(this->d_total1);
  cudaFree(this->d_max1);

  cudaFree(this->d_soa_total1);
  cudaFree(this->d_soa_max1);

  for (int i = 0; i < hash_params.len; ++i) {
    cudaFree(this->h_soa_total1.m[i]);
    cudaFree(this->h_soa_max1.m[i]);
  }

  cudaFree(this->d_row_ptr);
  cudaFree(this->d_col_ptr);
  cudaFree(this->d_tile_boundaries);
}

class GPUSetupD2 : public GPUSetupD1{
 public:
  // additionally for dist2
  SOACounters* d_soa_total2;
  SOACounters* d_soa_max2;
  SOACounters h_soa_total2;
  SOACounters h_soa_max2;
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

  int len_blocks_arr = num_bit_widths * n_tiles * sizeof(int);

  for (int i = 0; i < hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_total2.m[i]), len_blocks_arr);
  }
  for (int i = 0; i < hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_max2.m[i]), len_blocks_arr);
  }

  cudaMalloc((void**)&d_soa_total2, sizeof(SOACounters));
  cudaMalloc((void**)&d_soa_max2, sizeof(SOACounters));

  cudaMemcpy(d_soa_total2, &h_soa_total2, sizeof(SOACounters),
            cudaMemcpyHostToDevice);
  cudaMemcpy(d_soa_max2, &h_soa_max2, sizeof(SOACounters),
            cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_total2, hash_params.len * sizeof(Counters));
  cudaMalloc((void**)&d_max2, hash_params.len * sizeof(Counters));
}

GPUSetupD2::~GPUSetupD2() {
  //=================
  // Free device mem
  //=================
  cudaFree(this->d_total2);
  cudaFree(this->d_max2);

  cudaFree(this->d_soa_total2);
  cudaFree(this->d_soa_max2);

  for (int i = 0; i < hash_params.len; ++i) {
    cudaFree(this->h_soa_total2.m[i]);
    cudaFree(this->h_soa_max2.m[i]);
  }

  // automatic call to base class destructor
}

} // end apa22_coloring