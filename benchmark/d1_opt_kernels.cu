#include <nvbench/nvbench.cuh>
#include "cli_bench.cu"
#include "bench_util.cuh"

#include "setup.cuh"
#include "mat_loader.hpp"
#include "d1_opt_kernel.cuh"
#include "V2/warp_hash.cuh"

using BLOCKS_SM = nvbench::enum_type_list<1, 2>;
using THREADS_SM = nvbench::enum_type_list<256, 384, 512, 640, 768, 1024>;
// using BLOCKS_SM = nvbench::enum_type_list<1>;
// using THREADS_SM = nvbench::enum_type_list<1024>;

static constexpr const char* SM_ShMem_key = "Smem_SM";
std::vector<nvbench::int64_t> SM_ShMem_range = {64*1024};
static constexpr const char* Blocks_SM_key = "BLK_SM";
static constexpr const char* Threads_SM_key = "THREADS_SM";

using namespace apa22_coloring;


template <int MAX_THREADS_SM, int BLK_SM>
void D1Naive(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int>;

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
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}


template <int MAX_THREADS_SM, int BLK_SM>
void D1NaiveRanking(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int, cub::BLOCK_REDUCE_RAKING>;

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
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}


template <int MAX_THREADS_SM, int BLK_SM>
void D1DblTmp(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = d1KernelDblTmp<THREADS, BLK_SM, int>;

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
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1Struct(nvbench::state& state,
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
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1StructBreak(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = d1KernelStructRedBreak<THREADS, BLK_SM, int>;

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
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1WarpMapping(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D1warp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1_Warp, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
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
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

    
NVBENCH_BENCH_TYPES(D1Naive, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1NaiveRanking, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1DblTmp, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1Struct, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1StructBreak, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1WarpMapping, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);