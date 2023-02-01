#include <nvbench/nvbench.cuh>

#include <defines.hpp>
#include <setup.cuh>
#include <coloring.cuh>

using namespace apa22_coloring;
static constexpr int MAX_THREADS_SM = 1024;  // Turing (2080ti)
static constexpr const char* Mat = def::Mat3;

template <int BLK_SM>
void Dist1(nvbench::state &state, nvbench::type_list<nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D1, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel);
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

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel);
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

  MatLoader& mat_loader = MatLoader::getInstance(Mat);
  Tiling tiling(D2_SortNet, BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                (void*)kernel);
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

using MyEnumList = nvbench::enum_type_list<1, 2, 4>;

NVBENCH_BENCH_TYPES(Dist1, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});
NVBENCH_BENCH_TYPES(Dist2Banks, NVBENCH_TYPE_AXES(MyEnumList)).set_type_axes_names({"BLK_SM"});