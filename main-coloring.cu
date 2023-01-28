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

  constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
  constexpr int BLK_SM = 2;
  constexpr int THREADS = MAX_THREADS_SM/BLK_SM;

  int mat_nr = 2;          //Default value
  chCommandLineGet<int>(&mat_nr, "m", argc, argv);
  auto Mat = def::choseMat(mat_nr);

  #if DIST2
  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel<int, THREADS, BLK_SM>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);
  #else
  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1Kernel<int, THREADS, BLK_SM>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);
  #endif

  std::printf("M-row %d\n", mat_loader.m_rows);
  std::printf("Nr_tiles: %d\n", tiling.n_tiles);
  std::printf("Shmem target tiling mem: %d\n", tiling.tile_target_mem);
  std::printf("Actual Dyn shMem: %d\n", tiling.calc_shMem());

  std::printf("biggest_tile_nodes: %d\n", tiling.biggest_tile_nodes);
  std::printf("biggest_tile_edges: %d\n", tiling.biggest_tile_edges);
  std::printf("max nodes in any tile: %d\n", tiling.max_nodes);
  std::printf("max edges in any tile: %d\n", tiling.max_edges);
  
  // calc shMem
  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

#if DIST2
  coloring2Kernel<int, THREADS, BLK_SM>
  <<<gridSize, blockSize, shMem_bytes>>>(gpu_setup.d_row_ptr,
                                         gpu_setup.d_col_ptr,
                                         gpu_setup.d_tile_boundaries,
                                         tiling.tile_target_mem,
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
  cpu_dist1(mat_loader.row_ptr, mat_loader.col_ptr, mat_loader.m_rows,
            hash_params.val[0], &cpu_total1, &cpu_max1);
  std::printf("CPU dist 1 results\n");
  printResult(cpu_total1, cpu_max1);

#if DIST2
    Counters cpu_max2, cpu_total2;
    cpuDist2(mat_loader.row_ptr, mat_loader.col_ptr, mat_loader.m_rows,
             tiling.max_node_degree, hash_params.val[0], &cpu_total2, &cpu_max2);

    std::printf("CPU dist 2 results\n");
    printResult(cpu_total2, cpu_max2);
#endif

  return 0;
}