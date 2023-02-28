#include "gtest/gtest.h"

#include "cpu_coloring.hpp"
#include "mat_loader.hpp"
#include "setup.cuh"
#include "d2_opt_kernel.cuh"
#include "V2/WarpHash.cuh"

namespace {
using namespace apa22_coloring;


static constexpr int MAX_THREADS_SM = 512;
static constexpr int BLK_SM = 1;
static constexpr int THREADS = MAX_THREADS_SM/BLK_SM;

MatLoader& mat_loader = MatLoader::getInstance();

enum LaunchType {
    without_maxNodeDegree,
    with_maxNodeDegree,
    reduced_Interface,
    reduced_Interface_withoutMaxDegree
};

void D2kernelTest(Distance tag, void* kernel_fn, LaunchType launch_type){
  Tiling tiling(tag, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                kernel_fn);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  switch (launch_type)
  {
  case without_maxNodeDegree:
    {
    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                      (void *)&gpu_setup.d_col_ptr,
                      (void *)&gpu_setup.d_tile_boundaries,
                      (void *)&gpu_setup.blocks_total1,
                      (void *)&gpu_setup.blocks_total2,
                      (void *)&gpu_setup.blocks_max1,
                      (void *)&gpu_setup.blocks_max2,
                      (void *)&gpu_setup.d_total1,
                      (void *)&gpu_setup.d_max1,
                      (void *)&gpu_setup.d_total2,
                      (void *)&gpu_setup.d_max2
                      };
    cudaLaunchKernel(kernel_fn, gridSize, blockSize, kernelArgs, shMem_bytes, cudaStreamDefault);
    }
    break;
  case with_maxNodeDegree:
    {
    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                          (void *)&gpu_setup.d_col_ptr,
                          (void *)&gpu_setup.d_tile_boundaries,
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
    cudaLaunchKernel(kernel_fn, gridSize, blockSize, kernelArgs, shMem_bytes, cudaStreamDefault);
    }
    break;

  case reduced_Interface:
    {
    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                          (void *)&gpu_setup.d_col_ptr,
                          (void *)&gpu_setup.d_tile_boundaries,
                          (void *)&tiling.max_node_degree,
                          (void *)&gpu_setup.blocks_total2,
                          (void *)&gpu_setup.blocks_max2,
                          (void *)&gpu_setup.d_total2,
                          (void *)&gpu_setup.d_max2
                          };
    cudaLaunchKernel(kernel_fn, gridSize, blockSize, kernelArgs, shMem_bytes, cudaStreamDefault);
    }
    break;

  case reduced_Interface_withoutMaxDegree:
    {
    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                          (void *)&gpu_setup.d_col_ptr,
                          (void *)&gpu_setup.d_tile_boundaries,
                          (void *)&gpu_setup.blocks_total2,
                          (void *)&gpu_setup.blocks_max2,
                          (void *)&gpu_setup.d_total2,
                          (void *)&gpu_setup.d_max2
                          };
    cudaLaunchKernel(kernel_fn, gridSize, blockSize, kernelArgs, shMem_bytes, cudaStreamDefault);
    }
    break;
  
  default:
    break;
  }
  cudaError_t kernel_succes = cudaDeviceSynchronize();
  ASSERT_EQ(kernel_succes, cudaSuccess);

  // Allocate memory for results on host
  auto total2 = std::make_unique<Counters[]>(num_hashes);
  auto max2 = std::make_unique<Counters[]>(num_hashes);
  cudaMemcpy(total2.get(), gpu_setup.d_total2, num_hashes * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  cudaMemcpy(max2.get(), gpu_setup.d_max2, num_hashes * sizeof(Counters),
            cudaMemcpyDeviceToHost);

  for (int h = 0; h < num_hashes; ++h) {
    Counters cpu_max2, cpu_total2;
    cpuDist2(mat_loader.row_ptr, mat_loader.col_ptr, mat_loader.m_rows,
             tiling.max_node_degree, start_hash + h, &cpu_total2, &cpu_max2);

    for (int i = 0; i < num_bit_widths; ++i) {
      ASSERT_EQ(total2[h].m[i], cpu_total2.m[i]) << "hash: " << h+start_hash
                                                 << " bitw: " << i+start_bit_width;
      ASSERT_EQ(max2[h].m[i], cpu_max2.m[i]) << "hash: " << h+start_hash
                                             << " bitw: " << i+start_bit_width;
    }
  }
}



// TEST(Distance2Kernels, D2Warp) {
//   D2kernelTest(D2_Warp,
//                (void*)D2warp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>,
//                true);
// }

TEST(Distance2Kernels, D2SortNetSmall) {
  D2kernelTest(D2,
               (void*)coloring2SortNetSmall<THREADS, BLK_SM, int>,
               without_maxNodeDegree);
}

TEST(Distance2Kernels, D2InsertionSort) {
  D2kernelTest(D2,
               (void*)coloring2Kernel<THREADS, BLK_SM, int>,
               without_maxNodeDegree);
}

TEST(Distance2Kernels, D2SortNetBankConflictFree) {
  D2kernelTest(D2_SortNet,
               (void*)coloring2KernelBank<THREADS, BLK_SM, int>,
               with_maxNodeDegree);
}

TEST(Distance2Kernels, D2Warp) {
  D2kernelTest(D2_Warp,
               (void*)D2warp<THREADS, BLK_SM, num_hashes, start_hash,
                    int, char, num_bit_widths, start_bit_width, int>,
               reduced_Interface);
}
TEST(Distance2Kernels, D2warp_Conflicts) {
  D2kernelTest(D2_Warp_small,
               (void*)D2warp_Conflicts<THREADS, BLK_SM, num_hashes, start_hash,
                    int, char, num_bit_widths, start_bit_width, int>,
               reduced_Interface_withoutMaxDegree);
}

}