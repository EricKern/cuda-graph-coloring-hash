#include <coloring.cuh>
#include <cub/cub.cuh>
#include <defines.hpp>
#include <nvbench/nvbench.cuh>
#include <setup.cuh>

#include "reduceCompKernels.cuh"

using namespace apa22_coloring;
static constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
static constexpr const char* Mat = def::CurlCurl_4;
static constexpr const char* MAT_NAME = "CurlCurl_4";

template <int BLK_SM>
void Dist1Ranking_Commutative(nvbench::state& state,
                              nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int,
                                cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Ranking(nvbench::state& state,
                  nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int, cub::BLOCK_REDUCE_RAKING>;
  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Warp(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Ranking_Commutative2(nvbench::state& state,
                               nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1KernelDoubleTemp<THREADS, BLK_SM, int,
                                cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1Warp2(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1KernelDoubleTemp<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1CustomReduceWarp(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1KernelCustomReduce<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist1CustomReduceLast(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1KernelCustomReduceLast<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void Dist2Ranking_Commutative(nvbench::state& state,
                              nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel<THREADS, BLK_SM, int,
                                cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Warp(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Ranking_Commutative2(nvbench::state& state,
                               nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel2Temp<THREADS, BLK_SM, int,
                                     cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Ranking_Commutative4(nvbench::state& state,
                               nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel4Temp<THREADS, BLK_SM, int,
                                     cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Warp2(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel2Temp<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2Warp4(nvbench::state& state,
                nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel4Temp<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2CustomReduce(nvbench::state& state,
                       nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2KernelCustomReduce<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int BLK_SM>
void Dist2CustomReduceLast(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2KernelCustomReduceLast<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  state.add_element_count(0, MAT_NAME);
  state.add_element_count(mat_loader.m_rows, "Rows");
  state.add_element_count(mat_loader.row_ptr[mat_loader.m_rows], "Non-zeroes");

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        gpu_setup.blocks_total1, gpu_setup.blocks_max1, gpu_setup.blocks_total2,
        gpu_setup.blocks_max2, gpu_setup.d_total1, gpu_setup.d_max1,
        gpu_setup.d_total2, gpu_setup.d_max2);
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