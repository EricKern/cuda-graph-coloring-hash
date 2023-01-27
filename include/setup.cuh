

#include <coloringCounters.cuh>
#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling

namespace apa = apa22_coloring;

class GPUSetupD1 {
 protected:
  size_t m_rows;
  size_t row_ptr_len;
  size_t col_ptr_len;
  size_t tile_bound_len;

 public:
  // preprocessing
  int n_tiles;
  int max_nodes;
  int max_edges;
  int biggest_tile_nodes;
  int biggest_tile_edges;

  //
  int* d_row_ptr;
  int* d_col_ptr;
  int* d_tile_boundaries;
  apa::SOACounters* d_soa_total1;
  apa::SOACounters* d_soa_max1;
  apa::SOACounters h_soa_total1;
  apa::SOACounters h_soa_max1;
  apa::Counters* d_total1;
  apa::Counters* d_max1;

  GPUSetupD1(GPUSetupD1 const&) = delete;
  void operator=(GPUSetupD1 const&) = delete;

  virtual int calc_shMem();
  GPUSetupD1(int* row_ptr, int* col_ptr, int* tile_boundaries, int n_tiles);
  ~GPUSetupD1();
};

int GPUSetupD1::calc_shMem(){
  return (biggest_tile_nodes + 1 + biggest_tile_edges) * sizeof(int);
}

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

  get_MaxTileSize(n_tiles, tile_boundaries, row_ptr,
                  &biggest_tile_nodes, &biggest_tile_edges, &max_nodes, &max_edges);

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

  int len_blocks_arr = apa::num_bit_widths * n_tiles * sizeof(int);

  for (int i = 0; i < apa::hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_total1.m[i]), len_blocks_arr);
  }
  for (int i = 0; i < apa::hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_max1.m[i]), len_blocks_arr);
  }
  //==========================================================================
  // Allocate memory for structs holding pointers to intermediate result alloc
  //==========================================================================
  cudaMalloc((void**)&d_soa_total1, sizeof(apa::SOACounters));
  cudaMalloc((void**)&d_soa_max1, sizeof(apa::SOACounters));

  cudaMemcpy(d_soa_total1, &h_soa_total1, sizeof(apa::SOACounters),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_soa_max1, &h_soa_max1, sizeof(apa::SOACounters),
             cudaMemcpyHostToDevice);

  //==================================
  // Allocate memory for return values
  //==================================
  cudaMalloc((void**)&d_total1, apa::hash_params.len * sizeof(apa::Counters));
  cudaMalloc((void**)&d_max1, apa::hash_params.len * sizeof(apa::Counters));
}

GPUSetupD1::~GPUSetupD1(){
  //=================
  // Free device mem
  //=================
  cudaFree(this->d_total1);
  cudaFree(this->d_max1);

  cudaFree(this->d_soa_total1);
  cudaFree(this->d_soa_max1);

  for (int i = 0; i < apa::hash_params.len; ++i) {
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
  apa::SOACounters* d_soa_total2;
  apa::SOACounters* d_soa_max2;
  apa::SOACounters h_soa_total2;
  apa::SOACounters h_soa_max2;
  apa::Counters* d_total2;
  apa::Counters* d_max2;

  int calc_shMem() override;
  GPUSetupD2(int* row_ptr, int* col_ptr, int* tile_boundaries, int n_tiles);
  ~GPUSetupD2();
  
  GPUSetupD2(GPUSetupD2 const&) = delete;
  void operator=(GPUSetupD2 const&) = delete;
};


int GPUSetupD2::calc_shMem() {
  return (biggest_tile_nodes + 1 + biggest_tile_edges) * sizeof(int) * 2;
}
  
GPUSetupD2::GPUSetupD2(int* row_ptr,
                       int* col_ptr,
                       int* tile_boundaries,
                       int n_tiles) :
                       GPUSetupD1(row_ptr, col_ptr, tile_boundaries, n_tiles) {

  int len_blocks_arr = apa::num_bit_widths * n_tiles * sizeof(int);

  for (int i = 0; i < apa::hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_total2.m[i]), len_blocks_arr);
  }
  for (int i = 0; i < apa::hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_max2.m[i]), len_blocks_arr);
  }

  cudaMalloc((void**)&d_soa_total2, sizeof(apa::SOACounters));
  cudaMalloc((void**)&d_soa_max2, sizeof(apa::SOACounters));

  cudaMemcpy(d_soa_total2, &h_soa_total2, sizeof(apa::SOACounters),
            cudaMemcpyHostToDevice);
  cudaMemcpy(d_soa_max2, &h_soa_max2, sizeof(apa::SOACounters),
            cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_total2, apa::hash_params.len * sizeof(apa::Counters));
  cudaMalloc((void**)&d_max2, apa::hash_params.len * sizeof(apa::Counters));
}

GPUSetupD2::~GPUSetupD2() {
  //=================
  // Free device mem
  //=================
  cudaFree(this->d_total2);
  cudaFree(this->d_max2);

  cudaFree(this->d_soa_total2);
  cudaFree(this->d_soa_max2);

  for (int i = 0; i < apa::hash_params.len; ++i) {
    cudaFree(this->h_soa_total2.m[i]);
    cudaFree(this->h_soa_max2.m[i]);
  }

  // automatic call to base class destructor
}
