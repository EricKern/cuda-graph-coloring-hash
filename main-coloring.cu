#include <cli_parser.hpp>
#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <coloring.cuh>

#include <stdio.h>
#include <iostream>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <asc.cuh>

#include <defines.hpp>
#include <kernel_setup.hpp>
#include <coloringCounters.cuh>

#include <cpu_coloring.hpp>

void printResult(const apa22_coloring::Counters& sum,
                 const apa22_coloring::Counters& max) {
    printf("Total Collisions\n");
    const auto start_bw = apa22_coloring::start_bit_width;
    for (uint i = 0; i < apa22_coloring::num_bit_widths; ++i) {
      printf("Mask width: %d, Collisions: %d\n", i+start_bw, sum.m[i]);
    }

    printf("Max Collisions per Node\n");
    for (uint i = 0; i < apa22_coloring::num_bit_widths; ++i) {
      printf("Mask width: %d, Collisions: %d\n", i+start_bw, max.m[i]);
    }
}

int main(int argc, char const *argv[]) {
  using namespace apa22_coloring;

  int mat_nr = 2;          //Default value
  chCommandLineGet<int>(&mat_nr, "m", argc, argv);

  const char* inputMat = def::choseMat(mat_nr);

  int* row_ptr;
  int* col_ptr;
  double* val_ptr;  // create pointers for matrix in csr format
  int m_rows;
  // const int m_rows =
  //     cpumultiplyDloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);

  int* ndc_;     // array with indices of each tile in all slices
  // int* slices_;  // array with nodes grouped in slices
  // int* offsets_;
  // simple_tiling(m_rows, number_of_tiles, row_ptr, col_ptr, &slices_, &ndc_,
  //               &offsets_);
  // cpumultiplyDpermuteMatrix(number_of_tiles, 1, ndc_, slices_, row_ptr, col_ptr,
  //                           val_ptr, &row_ptr, &col_ptr, &val_ptr, true);
  int number_of_tiles;
  int shMem_size_bytes;
  kernel_setup(inputMat, row_ptr, col_ptr, val_ptr, ndc_, m_rows, number_of_tiles, shMem_size_bytes);
  printf("Nr_tiles: %d\n", number_of_tiles);
  printf("shMem: %d\n", shMem_size_bytes);
  printf("M-row %d", m_rows);
  std::cout << std::endl;

  int* d_row_ptr;
  int* d_col_ptr;
  int* d_tile_boundaries;
  
  size_t row_ptr_len = m_rows + 1;
  size_t col_ptr_len = row_ptr[m_rows];
  size_t tile_bound_len = number_of_tiles + 1;

      printf("Post cudaMalloc");
  std::cout << std::endl;

  cudaMalloc((void**)&d_row_ptr, row_ptr_len * sizeof(int));
  cudaMalloc((void**)&d_col_ptr, col_ptr_len * sizeof(int));
  cudaMalloc((void**)&d_tile_boundaries, tile_bound_len * sizeof(int));


  cudaMemcpy(d_row_ptr, row_ptr, row_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ptr, col_ptr, col_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tile_boundaries, ndc_, tile_bound_len * sizeof(int),
             cudaMemcpyHostToDevice);

  // For each bit_width we allocate a counter for each block and for each hash function
  SOACounters h_soa_total;
  for (int i = 0; i < hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_total.m[i]), num_bit_widths * number_of_tiles * sizeof(int));
  }
  SOACounters h_soa_max;
  for (int i = 0; i < hash_params.len; ++i) {
    cudaMalloc((void**)&(h_soa_max.m[i]), num_bit_widths * number_of_tiles * sizeof(int));
  }
  SOACounters* d_soa_total;
  cudaMalloc((void**)&d_soa_total, sizeof(SOACounters));
  SOACounters* d_soa_max;
  cudaMalloc((void**)&d_soa_max, sizeof(SOACounters));

  cudaMemcpy(d_soa_total, &h_soa_total, sizeof(SOACounters),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_soa_max, &h_soa_max, sizeof(SOACounters),
             cudaMemcpyHostToDevice);

  Counters* d_total;
  cudaMalloc((void**)&d_total, hash_params.len * sizeof(Counters));
  Counters* d_max;
  cudaMalloc((void**)&d_max, hash_params.len * sizeof(Counters));
  
  // calc shMem
  size_t shMem_bytes = shMem_size_bytes;
  dim3 gridSize(number_of_tiles);
  dim3 blockSize(THREADS);

  uint max_nodes, max_edges;
  get_MaxTileSize(number_of_tiles, ndc_, row_ptr, &max_nodes, &max_edges);

  printf("Pre Kernel");
  std::cout << std::endl;

  // cudaFuncSetAttribute(coloring1Kernel<int>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  coloring1Kernel<<<gridSize, blockSize, shMem_bytes>>>(
      d_row_ptr, d_col_ptr, d_tile_boundaries,
      max_nodes, max_edges, d_soa_total, d_soa_max, d_total, d_max);
  cudaDeviceSynchronize();

  printf("Post Kernel");
  std::cout << std::endl;

  Counters* total = new Counters[hash_params.len];
  cudaMemcpy(total, d_total, hash_params.len * sizeof(Counters),
             cudaMemcpyDeviceToHost);
  Counters* max = new Counters[hash_params.len];
  cudaMemcpy(max, d_max, hash_params.len * sizeof(Counters),
             cudaMemcpyDeviceToHost);

  printResult(total[0], max[0]);

  Counters cpu_max, cpu_total;
  cpu_dist1(row_ptr, col_ptr, m_rows, &cpu_total, &cpu_max);
  
  printf("CPU results");
  printResult(cpu_total, cpu_max);

  cudaFree(d_row_ptr);
  cudaFree(d_col_ptr);
  cudaFree(d_tile_boundaries);

  printf("Post cudaFree");
  std::cout << std::endl;

  // thrust::host_vector<int> row(row_ptr, row_ptr + m_rows + 1);
  // thrust::host_vector<int> col(col_ptr, col_ptr + row_ptr[m_rows]);
  // thrust::host_vector<double> nnz(val_ptr, val_ptr + row_ptr[m_rows]);
  
  // thrust::device_vector<int> d_row = row;
  // thrust::device_vector<int> d_col = col;
  // thrust::device_vector<double> d_nnz = nnz;

  // namespace asc18 = asc_hash_graph_coloring;
  // asc18::cusparse_distance1(d_nnz, d_row, d_col, 1);

  // delete total;
  // delete max;
	delete[] row_ptr;
	delete[] col_ptr;
	delete[] val_ptr;
	delete[] ndc_;
  return 0;
}