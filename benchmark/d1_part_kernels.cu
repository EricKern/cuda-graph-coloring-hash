#include <nvbench/nvbench.cuh>
#include "cli_bench.cu"
#include "bench_util.cuh"

#include "setup.cuh"
#include "mat_loader.hpp"
#include "d1_part_kernels.cuh"

using BLOCKS_SM = nvbench::enum_type_list<1, 2, 4>;
using THREADS_SM = nvbench::enum_type_list<512, 640, 768, 896, 1024>;
// using BLOCKS_SM = nvbench::enum_type_list<1>;
// using THREADS_SM = nvbench::enum_type_list<1024>;
using N_Hashes = nvbench::enum_type_list<15>;
// using N_Hashes = nvbench::enum_type_list<1, 3, 5, 7, 9, 11, 13, 15>;
static constexpr const char* SM_ShMem_key = "Smem_SM";
std::vector<nvbench::int64_t> SM_ShMem_range = {64*1024};
static constexpr const char* Blocks_SM_key = "BLK_SM";
static constexpr const char* Threads_SM_key = "THREADS_SM";

using namespace apa22_coloring;

template <int MAX_THREADS_SM, int BLK_SM>
void D1OnlyLoad(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1OnlyLoad<THREADS, BLK_SM, int>;

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

template <int MAX_THREADS_SM, int BLK_SM, int N_Hashes>
void D1OnlyHash(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>,
                                   nvbench::enum_type<N_Hashes>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1OnlyHash<THREADS, BLK_SM, int, N_Hashes>;

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
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
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
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
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
void D1FirstReduceNoHash(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1ReduceNoHash<THREADS, BLK_SM, int>;

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
void D1FirstReduceNoHashNoBarrier(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1ReduceNoHashNoBarrier<THREADS, BLK_SM, int>;

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
void D1FirstStructReduceNoHash(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                   nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1StructReduceNoHash<THREADS, BLK_SM, int>;

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
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
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
void D1Full(nvbench::state& state,
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

NVBENCH_BENCH_TYPES(D1OnlyLoad, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1OnlyHash, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM, N_Hashes))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key, "N_Hashes"})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1LoadWrite, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
NVBENCH_BENCH_TYPES(D1HashWrite, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1FirstStructReduceNoHash, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1FirstReduceNoHashNoBarrier, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1FirstReduceNoHash, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D1FirstReduce, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
    
NVBENCH_BENCH_TYPES(D1Full, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);


    