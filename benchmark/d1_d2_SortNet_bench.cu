#include <nvbench/nvbench.cuh>

#include "defines.hpp"
#include "setup.cuh"
#include "coloring.cuh"
#include "coop_launch.cuh"
#include "cli_bench.cu"
#include "mat_loader.hpp"
#include "bench_util.cuh"

using namespace apa22_coloring;
static constexpr int MAX_THREADS_SM = 832;  // Turing (2080ti)

using BlocksP_SM = nvbench::enum_type_list<2>;
using MAX_THREADS = nvbench::enum_type_list<576, 640, 704, 1024>;
static constexpr const char* SM_ShMem_key = "SM_ShMem";
std::vector<nvbench::int64_t> SM_ShMem_range = {-1};

template <int BLK_SM, int MAX_THREADS>
void Dist1(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>, nvbench::enum_type<MAX_THREADS>>) {
  constexpr int THREADS = MAX_THREADS / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  add_MatInfo(state);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel,
                state.get_int64(SM_ShMem_key));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr,
        gpu_setup.d_col_ptr,
        gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1,
        gpu_setup.blocks_max1,
        gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist2(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  add_MatInfo(state);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel,
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.blocks_total1,
            gpu_setup.blocks_total2,
            gpu_setup.blocks_max1,
            gpu_setup.blocks_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Banks(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2KernelBank<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  add_MatInfo(state);
  Tiling tiling(D2_SortNet, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel,
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.max_node_degree,
            gpu_setup.blocks_total1,
            gpu_setup.blocks_total2,
            gpu_setup.blocks_max1,
            gpu_setup.blocks_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Coop(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2coop<THREADS,BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  add_MatInfo(state);
  Tiling tiling(D2_SortNet, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel,
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

    cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)coloring2coop<THREADS, BLK_SM, int>, THREADS, shMem_bytes);
  gridSize.x = deviceProp.multiProcessorCount*numBlocksPerSm;
  std::printf("numBlocksPerSm: %d\n", numBlocksPerSm);
  std::printf("gridSize.x: %d\n", gridSize.x);

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

  

  state.exec([&](nvbench::launch& launch) {
    cudaLaunchCooperativeKernel((void *)kernel, gridSize,
                              blockSize, kernelArgs, shMem_bytes, NULL);
  });
}



// NVBENCH_BENCH_TYPES(Dist1, NVBENCH_TYPE_AXES(BlocksP_SM, MAX_THREADS))
//     .set_type_axes_names({"BLK_SM", "THREADS_SM"})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
// NVBENCH_BENCH_TYPES(Dist2, NVBENCH_TYPE_AXES(BlocksP_SM))
//     .set_type_axes_names({"BLK_SM"})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(Dist2Coop, NVBENCH_TYPE_AXES(BlocksP_SM))
    .set_type_axes_names({"BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(Dist2Banks, NVBENCH_TYPE_AXES(BlocksP_SM))
    .set_type_axes_names({"BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
