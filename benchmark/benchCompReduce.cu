#include <nvbench/nvbench.cuh>
#include <cub/cub.cuh>

#include <defines.hpp>
#include <setup.cuh>
#include <coloring.cuh>

#include "reduceCompKernels.cuh"

using namespace apa22_coloring;
static constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
static constexpr const char* Mat = def::Mat3_Cluster;

template <int BLK_SM>
void Dist1Ranking_Commutative(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Ranking(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Warp(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1Kernel<int, THREADS, BLK_SM>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_WARP_REDUCTIONS>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Ranking_Commutative2(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1KernelDoubleTemp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1KernelDoubleTemp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}


template <int BLK_SM>
void Dist1Warp2(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1KernelDoubleTemp<int, THREADS, BLK_SM>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1KernelDoubleTemp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_WARP_REDUCTIONS>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1CustomReduceWarp(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1KernelCustomReduce<int, THREADS, BLK_SM>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1KernelCustomReduce<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1CustomReduceLast(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring1KernelCustomReduceLast<int, THREADS, BLK_SM>);
  GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring1KernelCustomReduceLast<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_total1,
            gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist2Ranking_Commutative(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2Kernel<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Warp(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel<int, THREADS, BLK_SM>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2Kernel<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}


template <int BLK_SM>
void Dist2Ranking_Commutative2(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel2Temp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2Kernel2Temp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.tile_target_mem,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Ranking_Commutative4(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel4Temp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2Kernel4Temp<int, THREADS, BLK_SM, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.tile_target_mem,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Warp2(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel2Temp<int, THREADS, BLK_SM>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2Kernel2Temp<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.tile_target_mem,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Warp4(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2Kernel4Temp<int, THREADS, BLK_SM>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2Kernel4Temp<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.tile_target_mem,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2CustomReduce(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2KernelCustomReduce<int, THREADS, BLK_SM>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2KernelCustomReduce<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.tile_target_mem,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2CustomReduceLast(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)coloring2KernelCustomReduceLast<int, THREADS, BLK_SM>);
  GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                       mat_loader.col_ptr,
                       tiling.tile_boundaries.get(),
                       tiling.n_tiles);

  size_t shMem_bytes = tiling.calc_shMem();
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.exec([&](nvbench::launch& launch) {
    coloring2KernelCustomReduceLast<int, THREADS, BLK_SM>
        <<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
            gpu_setup.d_row_ptr,
            gpu_setup.d_col_ptr,
            gpu_setup.d_tile_boundaries,
            tiling.tile_target_mem,
            gpu_setup.d_soa_total1,
            gpu_setup.d_soa_max1,
            gpu_setup.d_soa_total2,
            gpu_setup.d_soa_max2,
            gpu_setup.d_total1,
            gpu_setup.d_max1,
            gpu_setup.d_total2,
            gpu_setup.d_max2);
  });
}

using MyEnumList = nvbench::enum_type_list<1, 2, 4>;

NVBENCH_BENCH_TYPES(Dist1Ranking_Commutative, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist1Ranking, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist1Warp, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist1Ranking_Commutative2, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist1Warp2, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist1CustomReduceWarp, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist1CustomReduceLast, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Ranking_Commutative, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Ranking_Commutative2, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Ranking_Commutative4, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Warp, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Warp2, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Warp4, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2CustomReduce, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2CustomReduceLast, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});