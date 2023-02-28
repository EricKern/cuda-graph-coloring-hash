#include "gtest/gtest.h"

#include "cpu_coloring.hpp"
#include "mat_loader.hpp"
#include "setup.cuh"
#include "d1_opt_kernel.cuh"
#include "coop_launch.cuh"
#include "V2/warp_hash.cuh"

namespace {
using namespace apa22_coloring;


static constexpr int MAX_THREADS_SM = 512;
static constexpr int BLK_SM = 1;
static constexpr int THREADS = MAX_THREADS_SM/BLK_SM;

MatLoader& mat_loader = MatLoader::getInstance();

void D1kernelTest(Distance tag, void* kernel_fn, bool coop=false){
  Tiling tiling(tag, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                kernel_fn);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  if(!coop) {
    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                          (void *)&gpu_setup.d_col_ptr,
                          (void *)&gpu_setup.d_tile_boundaries,
                          (void *)&gpu_setup.blocks_total1,
                          (void *)&gpu_setup.blocks_max1,
                          (void *)&gpu_setup.d_total1,
                          (void *)&gpu_setup.d_max1
                          };

    cudaLaunchKernel(kernel_fn, gridSize, blockSize, kernelArgs, shMem_bytes, cudaStreamDefault);
  } else {
    gridSize.x = get_coop_grid_size(kernel_fn,
                                  blockSize.x, shMem_bytes);
    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                          (void *)&gpu_setup.d_col_ptr,
                          (void *)&gpu_setup.d_tile_boundaries,
                          (void *)&tiling.n_tiles,
                          (void *)&gpu_setup.blocks_total1,
                          (void *)&gpu_setup.blocks_max1,
                          (void *)&gpu_setup.d_total1,
                          (void *)&gpu_setup.d_max1
                          };

    cudaLaunchCooperativeKernel(kernel_fn, gridSize,
                                blockSize, kernelArgs, shMem_bytes, NULL);
  }
  cudaError_t kernel_succes = cudaDeviceSynchronize();
  ASSERT_EQ(kernel_succes, cudaSuccess);

  // Allocate memory for results on host
  auto total1 = std::make_unique<Counters[]>(num_hashes);
  auto max1 = std::make_unique<Counters[]>(num_hashes);

  cudaMemcpy(total1.get(), gpu_setup.d_total1, num_hashes * sizeof(Counters),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(max1.get(), gpu_setup.d_max1, num_hashes * sizeof(Counters),
             cudaMemcpyDeviceToHost);

  for (int h = 0; h < num_hashes; ++h) {
    Counters cpu_max1, cpu_total1;
    cpu_dist1(mat_loader.row_ptr, mat_loader.col_ptr, mat_loader.m_rows,
              start_hash + h, &cpu_total1, &cpu_max1);

    for (int i = 0; i < num_bit_widths; ++i) {
      ASSERT_EQ(total1[h].m[i], cpu_total1.m[i]) << "hash: " << h+start_hash
                                                 << " bitw: " << i+start_bit_width;
      ASSERT_EQ(max1[h].m[i], cpu_max1.m[i]) << "hash: " << h+start_hash
                                             << " bitw: " << i+start_bit_width;
    }
  }
}



TEST(Distance1Kernels, D1Initial) {
  D1kernelTest(D1,
               (void*)coloring1Kernel<THREADS, BLK_SM, int>);
}

TEST(Distance1Kernels, D1StructReduce) {
  D1kernelTest(D1,
               (void*)d1KernelStructRed<THREADS, BLK_SM, int>);
}

TEST(Distance1Kernels, D1BreakStatement) {
  D1kernelTest(D1,
               (void*)d1KernelStructRedBreak<THREADS, BLK_SM, int>);
}

TEST(Distance1Kernels, D1Warp) {
  D1kernelTest(D1_Warp,
               (void*)D1warp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>);
}

TEST(Distance1Kernels, D1CoopInitial) {
  D1kernelTest(D1,
               (void*)coloring1coopBase<THREADS, BLK_SM, int>,
               true);
}

TEST(Distance1Kernels, D1Coop) {
  D1kernelTest(D1,
               (void*)coloring1coop<THREADS, BLK_SM, int>,
               true);
}

}