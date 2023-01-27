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
// #include <kernel_setup.hpp>
#include <coloringCounters.cuh>

#include <cpu_coloring.hpp>

#define DIST2 1

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

  constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
  constexpr int BLK_SM = 4;
  constexpr int THREADS = MAX_THREADS_SM/BLK_SM;
  #if DIST2
  auto kernel = coloring2Kernel<int, THREADS, BLK_SM>;
  #else
  auto kernel = coloring1Kernel<int, THREADS, BLK_SM>;
  #endif

  int devId, MaxShmemSizeSM;
  cudaGetDevice(&devId);
  cudaDeviceGetAttribute(&MaxShmemSizeSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, devId);
  // On Turing 64 kB Shmem hard Cap with 32kB L1.
  // There are other predefined "carveout" levels that might also be an option
  // if more L1 cache seems usefull (on Turing just 32kB).
  // MaxShmemSizeSM /= 2;
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, kernel);

  // Starting with CC 8.0, cuda runtime needs 1KB shMem per block
  int const static_shMem_SM = a.sharedSizeBytes * BLK_SM; 
  int const max_dyn_SM = MaxShmemSizeSM - static_shMem_SM;
  
  // cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn_SM);

  std::printf("MaxShmemSizeSM: %d\n", MaxShmemSizeSM);
  std::printf("a.sharedSizeBytes: %d\n", a.sharedSizeBytes);
  std::printf("a.sharedSizeBytes * BLK_SM: %d\n", a.sharedSizeBytes * BLK_SM);
  std::printf("max_dyn_SM: %d\n", max_dyn_SM);

  int mat_nr = 2;          //Default value
  chCommandLineGet<int>(&mat_nr, "m", argc, argv);
  auto Mat = def::choseMat(mat_nr);

  int* row_ptr;
  int* col_ptr;
  double* val_ptr;
  int m_rows = cpumultiplyDloadMTX(Mat, &row_ptr, &col_ptr, &val_ptr);

  std::unique_ptr<int[]> tile_boundaries; // array with indices of each tile in all slices
  int n_tiles;
  int max_node_degree;
  #if DIST2
  int tile_mem = (max_dyn_SM/BLK_SM)/2;
  #else
  int tile_mem = max_dyn_SM/BLK_SM;
  #endif
  very_simple_tiling(row_ptr, m_rows, tile_mem, &tile_boundaries, &n_tiles, &max_node_degree);

  #if DIST2
  GPUSetupD2 gpu_setup(row_ptr, col_ptr, tile_boundaries.get(), n_tiles);
  #else
  GPUSetupD1 gpu_setup(row_ptr, col_ptr, tile_boundaries.get(), n_tiles);
  #endif

  std::printf("Nr_tiles: %d\n", gpu_setup.n_tiles);
  std::printf("Smem target tiling: %d\n", tile_mem);
  std::printf("Actual Dyn shMem: %d\n", gpu_setup.calc_shMem());
  std::printf("M-row %d\n", m_rows);

  std::printf("biggest_tile_nodes: %d\n", gpu_setup.biggest_tile_nodes);
  std::printf("biggest_tile_edges: %d\n", gpu_setup.biggest_tile_edges);
  
  // calc shMem
  size_t shMem_bytes = gpu_setup.calc_shMem();
  dim3 gridSize(gpu_setup.n_tiles);
  dim3 blockSize(THREADS);

#if DIST2
  coloring2Kernel<int, THREADS, BLK_SM>
  <<<gridSize, blockSize, shMem_bytes>>>(gpu_setup.d_row_ptr,
                                         gpu_setup.d_col_ptr,
                                         gpu_setup.d_tile_boundaries,
                                         tile_mem,
                                         gpu_setup.d_soa_total1,
                                         gpu_setup.d_soa_max1,
                                         gpu_setup.d_soa_total2,
                                         gpu_setup.d_soa_max2,
                                         gpu_setup.d_total1,
                                         gpu_setup.d_max1,
                                         gpu_setup.d_total2,
                                         gpu_setup.d_max2);
#else
  coloring1Kernel<int, THREADS, BLK_SM>
  <<<gridSize, blockSize, shMem_bytes>>>(gpu_setup.d_row_ptr,
                                         gpu_setup.d_col_ptr,
                                         gpu_setup.d_tile_boundaries,
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
  return 0;
}