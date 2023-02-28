#include <cstdio>
#include <memory>
#include <numeric>
#include <string>

#include "cli_parser.hpp"
#include "mat_loader.hpp"
#include "cpu_coloring.hpp"
#include "setup.cuh"
#include "defines.hpp"

#include "coloring_counters.cuh"

#include "V2/warp_hash_coop.cuh"


#define DIST2 1

void printResult(const apa22_coloring::Counters& sum,
                 const apa22_coloring::Counters& max) {
  std::printf("Total Collisions\n");
  const auto start_bw = apa22_coloring::start_bit_width;
  for (uint i = 0; i < apa22_coloring::num_bit_widths; ++i) {
    std::printf("Mask width: %d, Collisions: %d\n", i + start_bw, sum.m[i]);
  }

  std::printf("Max Collisions per Node\n");
  for (uint i = 0; i < apa22_coloring::num_bit_widths; ++i) {
    std::printf("Mask width: %d, Collisions: %d\n", i + start_bw, max.m[i]);
  }
}

int main(int argc, char const *argv[]) {
  using namespace apa22_coloring;

  constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
  constexpr int BLK_SM = 1;
  constexpr int THREADS = MAX_THREADS_SM/BLK_SM;

  const char* mat_input = "CurlCurl_4";          //Default value
  chCommandLineGet<const char*>(&mat_input, "mat", argc, argv);
  // auto Mat = def::CurlCurl_4;
  std::string mat_in_str(mat_input);
  const char* Mat;
  if (auto search = def::map.find(mat_in_str); search != def::map.end()){
    std::cout << "Found " << search->first << " " << search->second << '\n';
    Mat = search->second;
  }
  else {
      std::cout << "Not found\n";
      Mat = mat_input;
  }

  #if DIST2
  auto kernel = coloring2coopWarp<THREADS, BLK_SM, num_hashes, start_hash, int,
                                  char, num_bit_widths, start_bit_width, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2_Warp, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel,
                64*1024, true);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);
  #else
  auto kernel = coloring1coopWarp<THREADS, BLK_SM, num_hashes, start_hash, int,
                                  char, num_bit_widths, start_bit_width, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1_Warp, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel,
                64*1024, true);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);
  #endif

  std::printf("M-row %d\n", mat_loader.m_rows);
  std::printf("Nr_tiles: %d\n", tiling.n_tiles);
  std::printf("Shmem target tiling mem: %d\n", tiling.tile_target_mem);
  std::printf("Actual Dyn shMem: %d\n", tiling.tile_target_mem);

  std::printf("biggest_tile_nodes: %d\n", tiling.biggest_tile_nodes);
  std::printf("biggest_tile_edges: %d\n", tiling.biggest_tile_edges);
  std::printf("max nodes in any tile: %d\n", tiling.max_nodes);
  std::printf("max edges in any tile: %d\n", tiling.max_edges);
  size_t smem_size_part = (tiling.biggest_tile_nodes + tiling.biggest_tile_edges
        + 1) * sizeof(int);
  std::printf("Smem Size for Partition: %ld\n", smem_size_part);
  std::printf("max node degree: %d\n", tiling.max_node_degree);
  
  // calc shMem
  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

#if DIST2
  gridSize.x = get_coop_grid_size((void*)kernel,
                                  THREADS, shMem_bytes);

  void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                        (void *)&gpu_setup.d_col_ptr,
                        (void *)&gpu_setup.d_tile_boundaries,
                        (void *)&tiling.n_tiles,
                        (void *)&tiling.max_node_degree,
                        (void *)&gpu_setup.blocks_total1,
                        (void *)&gpu_setup.blocks_total2,
                        (void *)&gpu_setup.blocks_max1,
                        (void *)&gpu_setup.blocks_max2,
                        (void *)&gpu_setup.d_total1,
                        (void *)&gpu_setup.d_max1,
                        (void *)&gpu_setup.d_total2,
                        (void *)&gpu_setup.d_max2
                        };

  cudaLaunchCooperativeKernel((void *)kernel, gridSize, blockSize, kernelArgs,
                              shMem_bytes, cudaStreamDefault);
#else
  gridSize.x = get_coop_grid_size((void*)kernel,
                                  THREADS, shMem_bytes);

  void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                        (void *)&gpu_setup.d_col_ptr,
                        (void *)&gpu_setup.d_tile_boundaries,
                        (void *)&tiling.n_tiles,
                        (void *)&gpu_setup.blocks_total1,
                        (void *)&gpu_setup.blocks_max1,
                        (void *)&gpu_setup.d_total1,
                        (void *)&gpu_setup.d_max1
                        };

  cudaLaunchCooperativeKernel((void *)kernel, gridSize, blockSize, kernelArgs,
                              shMem_bytes, cudaStreamDefault);
#endif
  cudaDeviceSynchronize();

  std::printf("Post Kernel\n");

  //====================================
  // Allocate memory for results on host
  //====================================
  auto total1 = std::make_unique<Counters[]>(num_hashes);
  auto max1 = std::make_unique<Counters[]>(num_hashes);
  cudaMemcpy(total1.get(), gpu_setup.d_total1, num_hashes * sizeof(Counters),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(max1.get(), gpu_setup.d_max1, num_hashes * sizeof(Counters),
             cudaMemcpyDeviceToHost);
  std::printf("Dist 1\n");
  printResult(total1[0], max1[0]);

#if DIST2
  auto total2 = std::make_unique<Counters[]>(num_hashes);
  auto max2 = std::make_unique<Counters[]>(num_hashes);
  cudaMemcpy(total2.get(), gpu_setup.d_total2, num_hashes * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  cudaMemcpy(max2.get(), gpu_setup.d_max2, num_hashes * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  std::printf("Dist 2\n");
  printResult(total2[0], max2[0]);
#endif

  Counters cpu_max1, cpu_total1;
  cpu_dist1(mat_loader.row_ptr, mat_loader.col_ptr, mat_loader.m_rows,
            start_hash, &cpu_total1, &cpu_max1);
  std::printf("CPU dist 1 results\n");
  printResult(cpu_total1, cpu_max1);

#if DIST2
    Counters cpu_max2, cpu_total2;
    cpuDist2(mat_loader.row_ptr, mat_loader.col_ptr, mat_loader.m_rows,
             tiling.max_node_degree, start_hash, &cpu_total2, &cpu_max2);

    std::printf("CPU dist 2 results\n");
    printResult(cpu_total2, cpu_max2);
#endif

  return 0;
}