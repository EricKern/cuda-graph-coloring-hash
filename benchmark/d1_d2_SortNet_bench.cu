#include <nvbench/nvbench.cuh>

#include "defines.hpp"
#include "setup.cuh"
#include "coloring.cuh"
#include "cli_bench.cu"
#include "mat_loader.hpp"
#include "bench_util.cuh"

#include "d1_opt_kernel.cuh"  // for dist1 struct reduction
#include "d2_opt_kernel.cuh"
#include "V2/warp_hash.cuh"

using namespace apa22_coloring;

using BLOCKS_SM = nvbench::enum_type_list<1, 2>;
using THREADS_SM = nvbench::enum_type_list<384, 512, 640, 768, 896, 1024>;
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
void D2WarpMapping(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2warp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2_Warp, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        tiling.max_node_degree,
        gpu_setup.blocks_total2, gpu_setup.blocks_max2, gpu_setup.d_total2,
        gpu_setup.d_max2);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2Warp_Small_Mapping(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2warp_Conflicts<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D2_Warp_small, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, tiling.n_tiles);

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total2, gpu_setup.blocks_max2, gpu_setup.d_total2,
        gpu_setup.d_max2);
  });
}

// NVBENCH_BENCH_TYPES(Dist1, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Dist2ThrustSrt, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
    
// NVBENCH_BENCH_TYPES(Dist2Banks, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(Dist2SmallSnet, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Dist2, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
    
NVBENCH_BENCH_TYPES(D2WarpMapping, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D2Warp_Small_Mapping, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);