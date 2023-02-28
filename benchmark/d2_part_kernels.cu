#include <nvbench/nvbench.cuh>

#include "coloring.cuh"
#include "defines.hpp"
#include "coloring_counters.cuh"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "cli_bench.cu"
#include "bench_util.cuh"
#include "d2_part_kernels.cuh"

using BLOCKS_SM = nvbench::enum_type_list<1>;
using THREADS_SM = nvbench::enum_type_list<1024>;
using THREADS_SM_Fine = nvbench::enum_type_list<1024>;
static constexpr const char* Blocks_SM_key = "BLK_SM";
static constexpr const char* Threads_SM_key = "THREADS_SM";
static constexpr const char* SM_ShMem_key = "Smem_SM";
std::vector<nvbench::int64_t> SM_ShMem_range = {64*1024};


template <int MAX_THREADS_SM, int BLK_SM>
void Distance2(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2OnlyLoad(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2OnlyLoad<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2LoadWrite(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2LoadWrite<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2LoadHash(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2LoadHash<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2LoadHashSort(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2LoadHashSort<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2LoadCollisions(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2LoadCollisions<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2LoadCollisionsReduce(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = D2LoadHashReduce<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec(
      [&](nvbench::launch& launch) {
        kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
                gpu_setup.d_row_ptr,
                gpu_setup.d_col_ptr,
                gpu_setup.d_tile_boundaries,
                gpu_setup.blocks_total1,
                gpu_setup.blocks_max1,
                gpu_setup.blocks_total2,
                gpu_setup.blocks_max2,
                gpu_setup.d_total1,
                gpu_setup.d_max1,
                gpu_setup.d_total2,
                gpu_setup.d_max2);
      });
}

NVBENCH_BENCH_TYPES(D2OnlyLoad, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D2LoadWrite, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D2LoadHash, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D2LoadHashSort, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D2LoadCollisions, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D2LoadCollisionsReduce, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(Distance2, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);