#include <nvbench/nvbench.cuh>
#include "cli_bench.cu"
#include "bench_util.cuh"

#include "setup.cuh"
#include "mat_loader.hpp"
#include "d1_part_kernels.cuh"

using BLOCKS_SM = nvbench::enum_type_list<1, 2>;
using THREADS_SM = nvbench::enum_type_list<768, 1024>;
static constexpr const char* SM_ShMem_key = "SM_ShMem";
std::vector<nvbench::int64_t> SM_ShMem_range = {-1};

using namespace apa22_coloring;

template <int MAX_THREADS_SM, int BLK_SM>
void D1OnlyLoad(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1OnlyLoad<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  // matrix
  size_t in_elem = mat_l.m_rows + mat_l.row_ptr[mat_l.m_rows];
  state.add_global_memory_reads<int>(in_elem, "Mat Row+Col");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1OnlyHash(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1OnlyHash<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  // matrix
  size_t in_elem = mat_l.m_rows + mat_l.row_ptr[mat_l.m_rows];
  state.add_global_memory_reads<int>(in_elem, "Mat Row+Col");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1LoadWrite(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1LoadWrite<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  // matrix
  size_t in_elem = mat_l.m_rows + mat_l.row_ptr[mat_l.m_rows];
  state.add_global_memory_reads<int>(in_elem, "Mat Row+Col");
  // block reduction results
  state.add_global_memory_writes<Counters>(tiling.n_tiles * num_hashes, "R1 Counters");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1HashWrite(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1HashWrite<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  // matrix
  size_t in_elem = mat_l.m_rows + mat_l.row_ptr[mat_l.m_rows];
  state.add_global_memory_reads<int>(in_elem, "Mat Row+Col");
  // block reduction results
  state.add_global_memory_writes<Counters>(tiling.n_tiles * num_hashes, "R1 Counters");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D1FirstReduce(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1FirstReduce<THREADS, BLK_SM, int>;

  MatLoader& mat_l = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_l.row_ptr, mat_l.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_l.row_ptr, mat_l.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  // matrix
  size_t in_elem = mat_l.m_rows + mat_l.row_ptr[mat_l.m_rows];
  state.add_global_memory_reads<int>(in_elem, "Mat Row+Col");
  // block reduction results
  state.add_global_memory_writes<Counters>(tiling.n_tiles * num_hashes, "R1 Counters");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

NVBENCH_BENCH_TYPES(D1OnlyLoad, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({"THREADS_SM", "BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1OnlyHash, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({"THREADS_SM", "BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1LoadWrite, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({"THREADS_SM", "BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1HashWrite, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({"THREADS_SM", "BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1FirstReduce, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({"THREADS_SM", "BLK_SM"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);