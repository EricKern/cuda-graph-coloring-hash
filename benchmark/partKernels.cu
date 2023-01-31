#include <nvbench/nvbench.cuh>

#include <defines.hpp>
#include <setup.cuh>
#include <partKernels.cuh>

using namespace apa22_coloring;
static constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
static constexpr const char* Mat = def::Mat3;

template <int BLK_SM>
void D1OnlyLoad(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1OnlyLoad<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
        gpu_setup.d_soa_total1,
        gpu_setup.d_soa_max1,
        gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void D1OnlyHash(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1OnlyHash<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
        gpu_setup.d_soa_total1,
        gpu_setup.d_soa_max1,
        gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void D1HashWrite(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1HashWrite<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
        gpu_setup.d_soa_total1,
        gpu_setup.d_soa_max1,
        gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void D1FirstRed(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1FirstReduce<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
        gpu_setup.d_soa_total1,
        gpu_setup.d_soa_max1,
        gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void D1Normal(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
        gpu_setup.d_soa_total1,
        gpu_setup.d_soa_max1,
        gpu_setup.d_total1,
        gpu_setup.d_max1);
  });
}

template <int BLK_SM>
void D2OnlyLoad(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2OnlyLoad<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
void D2OnlyHashD1(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2OnlyHashD1<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
void D2OnlyHashD2(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2OnlyHashD2<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
void D2OnlyHash(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2OnlyHash<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
void D2HashWrite(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2HashWrite<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
void D2FirstRed(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2FirstRed<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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
void D2Normal(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2Kernel<int, THREADS, BLK_SM>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
              mat_loader.row_ptr,
              mat_loader.m_rows,
              reinterpret_cast<void*>(kernel));
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

NVBENCH_BENCH_TYPES(D1OnlyLoad, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D1OnlyHash, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D1HashWrite, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D1FirstRed, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D1Normal, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2OnlyLoad, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2OnlyHashD1, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2OnlyHashD2, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2OnlyHash, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2HashWrite, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2FirstRed, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(D2Normal, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});