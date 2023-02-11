#include <nvbench/nvbench.cuh>

#include "defines.hpp"
#include "setup.cuh"
#include "coloring.cuh"
#include "coop_launch.cuh"
#include "cli_bench.cu"
#include "mat_loader.hpp"
#include "bench_util.cuh"

#include "d1_opt_kernel.cuh"  // for dist1 struct reduction
#include "d2_opt_kernel.cuh"

using namespace apa22_coloring;

using BLOCKS_SM = nvbench::enum_type_list<1, 2, 4>;
using THREADS_SM = nvbench::enum_type_list<256, 384, 512, 640, 1024>;
// using BLOCKS_SM = nvbench::enum_type_list<1>;
// using THREADS_SM = nvbench::enum_type_list<1024>;
static constexpr const char* SM_ShMem_key = "Smem_SM";
std::vector<nvbench::int64_t> SM_ShMem_range = {64*1024};
static constexpr const char* Blocks_SM_key = "BLK_SM";
static constexpr const char* Threads_SM_key = "THREADS_SM";

template <int MAX_THREADS_SM, int BLK_SM>
void Dist1(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = d1KernelStructRed<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD1 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

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

template <int MAX_THREADS_SM, int BLK_SM>
void Dist2(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM,
                mat_l.row_ptr,
                mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr,
                       mat_l.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

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

template <int MAX_THREADS_SM, int BLK_SM>
void Dist2SmallSnet(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2SortNetSmall<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM,
                mat_l.row_ptr,
                mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr,
                       mat_l.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

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

template <int MAX_THREADS_SM, int BLK_SM>
void Dist2ThrustSrt(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2KernelThrust<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM,
                mat_l.row_ptr,
                mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr,
                       mat_l.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

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


template <int MAX_THREADS_SM, int BLK_SM>
void Dist2Banks(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2KernelBank<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2_SortNet, BLK_SM,
                mat_l.row_ptr,
                mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr,
                       mat_l.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

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

template <int MAX_THREADS_SM, int BLK_SM>
void Dist2Coop(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2coop<THREADS,BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2_SortNet, BLK_SM,
                mat_l.row_ptr,
                mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr,
                       mat_l.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 blockSize(THREADS);
  int coop_blocks = get_coop_grid_size(reinterpret_cast<void*>(kernel),
                                       THREADS,
                                       shMem_bytes);
  dim3 gridSize(coop_blocks);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

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


NVBENCH_BENCH_TYPES(Dist1, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(Dist2ThrustSrt, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
    
NVBENCH_BENCH_TYPES(Dist2Banks, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(Dist2SmallSnet, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(Dist2, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
    

// NVBENCH_BENCH_TYPES(Dist2Coop, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
    