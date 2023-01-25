#include <cli_parser.hpp>
#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <coloring.cuh>
#include <setup.cuh>

#include <cstdio>
#include <numeric>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <asc.cuh>

#include <defines.hpp>
#include <kernel_setup.hpp>
#include <coloringCounters.cuh>

#include <cpu_coloring.hpp>

#define DIST2 0

void printResult(const apa22_coloring::Counters& sum,
                 const apa22_coloring::Counters& max) {
    std::printf("Total Collisions\n");
    const auto start_bw = apa22_coloring::start_bit_width;
    for (uint i = 0; i < apa22_coloring::num_bit_widths; ++i) {
      std::printf("Mask width: %d, Collisions: %d\n", i+start_bw, sum.m[i]);
    }

    std::printf("Max Collisions per Node\n");
    for (uint i = 0; i < apa22_coloring::num_bit_widths; ++i) {
      std::printf("Mask width: %d, Collisions: %d\n", i+start_bw, max.m[i]);
    }
}

int main(int argc, char const *argv[]) {
  using namespace apa22_coloring;

  int mat_nr = 2;          //Default value
  chCommandLineGet<int>(&mat_nr, "m", argc, argv);

  const char* inputMat = def::Mat3_Cluster;

  int* row_ptr;
  int* col_ptr;
  double* val_ptr;  // create pointers for matrix in csr format
  int m_rows;

  int* ndc_;     // array with indices of each tile in all slices

  int number_of_tiles;
  int shMem_size_bytes;

  #if DIST2
  kernel_setup<true>(inputMat, row_ptr, col_ptr, val_ptr, ndc_, m_rows, number_of_tiles, shMem_size_bytes, 300);
  #else
  kernel_setup(inputMat, row_ptr, col_ptr, val_ptr, ndc_, m_rows, number_of_tiles, shMem_size_bytes, 300);
  #endif

  #if DIST2
  GPUSetupD2 gpu_setup(row_ptr, col_ptr, ndc_, number_of_tiles);
  #else
  GPUSetupD1 gpu_setup(row_ptr, col_ptr, ndc_, number_of_tiles);
  #endif

  std::printf("Nr_tiles: %d\n", gpu_setup.n_tiles);
  std::printf("shMem: %d\n", gpu_setup.calc_shMem());
  std::printf("M-row %d\n", m_rows);

  std::printf("node: %d\n", gpu_setup.max_nodes);
  std::printf("edges: %d\n", gpu_setup.max_edges);
  
  // calc shMem
  size_t shMem_bytes = gpu_setup.calc_shMem();
  dim3 gridSize(gpu_setup.n_tiles);
  dim3 blockSize(THREADS);

#if DIST2
  coloring2Kernel<<<gridSize, blockSize, shMem_bytes>>>(gpu_setup.d_row_ptr,
                                                        gpu_setup.d_col_ptr,
                                                        gpu_setup.d_tile_boundaries,
                                                        gpu_setup.max_nodes,
                                                        gpu_setup.max_edges,
                                                        gpu_setup.d_soa_total1,
                                                        gpu_setup.d_soa_max1,
                                                        gpu_setup.d_soa_total2,
                                                        gpu_setup.d_soa_max2,
                                                        gpu_setup.d_total1,
                                                        gpu_setup.d_max1,
                                                        gpu_setup.d_total2,
                                                        gpu_setup.d_max2);
#else
  coloring1Kernel<<<gridSize, blockSize, shMem_bytes>>>(gpu_setup.d_row_ptr,
                                                        gpu_setup.d_col_ptr,
                                                        gpu_setup.d_tile_boundaries,
                                                        gpu_setup.max_nodes,
                                                        gpu_setup.max_edges,
                                                        gpu_setup.d_soa_total1,
                                                        gpu_setup.d_soa_max1,
                                                        gpu_setup.d_total1,
                                                        gpu_setup.d_max1);
#endif
  cudaDeviceSynchronize();

  std::printf("Post Kernel\n");

  //====================================
  // Allocate memory for results on host
  //====================================
  std::unique_ptr<Counters[]> total1(new Counters[hash_params.len]);
  std::unique_ptr<Counters[]> max1(new Counters[hash_params.len]);
  cudaMemcpy(total1.get(), gpu_setup.d_total1, hash_params.len * sizeof(Counters),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(max1.get(), gpu_setup.d_max1, hash_params.len * sizeof(Counters),
             cudaMemcpyDeviceToHost);
  std::printf("Dist 1\n");
  printResult(total1[0], max1[0]);

#if DIST2
  std::unique_ptr<Counters[]> total2(new Counters[hash_params.len]);
  std::unique_ptr<Counters[]> max2(new Counters[hash_params.len]);
  cudaMemcpy(total2.get(), gpu_setup.d_total2, hash_params.len * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  cudaMemcpy(max2.get(), gpu_setup.d_max2, hash_params.len * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  std::printf("Dist 2\n");
  printResult(total2[0], max2[0]);
#endif

  Counters cpu_max1, cpu_total1;
  cpu_dist1(row_ptr, col_ptr, m_rows, &cpu_total1, &cpu_max1);
  std::printf("CPU dist 1 results\n");
  printResult(cpu_total1, cpu_max1);

#if DIST2
    Counters cpu_max2, cpu_total2;
    auto redBinaryOp = [](auto lhs, auto rhs){return rhs > lhs ? rhs : lhs;};
    auto transBinaryOp = [](auto lhs, auto rhs){return rhs - lhs;};
    int max_node_degree = std::transform_reduce(row_ptr,
                                                row_ptr + m_rows,
                                                row_ptr + 1,
                                                0,
                                                redBinaryOp,
                                                transBinaryOp);
    cpuDist2(row_ptr, col_ptr, m_rows, max_node_degree, &cpu_total2, &cpu_max2);
    
    std::printf("CPU dist 2 results\n");
    printResult(cpu_total2, cpu_max2);
#endif

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