#include <nvbench/nvbench.cuh>
#include <cusparse.h>
#include <thrust/device_vector.h>

#include "coloring.cuh"
#include "defines.hpp"
#include "coloring_counters.cuh"
#include "copy_kernel.cuh"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "cli_bench.cu"
#include "bench_util.cuh"

using BLOCKS_SM = nvbench::enum_type_list<1, 2, 4>;
using THREADS_SM = nvbench::enum_type_list<512, 640, 768, 896, 1024>;
using THREADS_SM_Fine = nvbench::enum_type_list<256, 384, 512, 640, 768, 896, 1024>;
static constexpr const char* Blocks_SM_key = "BLK_SM";
static constexpr const char* Threads_SM_key = "THREADS_SM";
static constexpr const char* SM_ShMem_key = "Smem_SM";
std::vector<nvbench::int64_t> SM_ShMem_range = {32*1024, 64*1024};

using namespace apa22_coloring;

template <int MAX_THREADS_SM, int BLK_SM>
void Distance1(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring1Kernel<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
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
            gpu_setup.d_total1,
            gpu_setup.d_max1);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void Distance1Copy(nvbench::state& state,
                   nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                      nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = copyKernelDist1<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
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
            gpu_setup.d_total1,
            gpu_setup.d_max1);
      });
}

template <int MAX_THREADS_SM, int BLK_SM>
void Distance1CopyFence(nvbench::state& state,
                        nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                           nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = copyKernelD1TFence<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D1, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD1 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
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
            gpu_setup.d_total1,
            gpu_setup.d_max1);
      });
}

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
void Distance2Bank(nvbench::state& state,
                   nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                      nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = coloring2KernelBank<THREADS, BLK_SM, int>;

  MatLoader& mat_loader = MatLoader::getInstance();
  Tiling tiling(D2_SortNet, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                state.get_int64(SM_ShMem_key));
  GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                       tiling.tile_boundaries.get(), tiling.n_tiles);

  size_t shMem_bytes = tiling.tile_target_mem;
  dim3 gridSize(tiling.n_tiles);
  dim3 blockSize(THREADS);

  add_MatInfo(state);
  add_IOInfo(state, gridSize.x);

  state.exec([&](nvbench::launch& launch) {
    kernel<<<gridSize, blockSize, shMem_bytes, launch.get_stream()>>>(
        gpu_setup.d_row_ptr, gpu_setup.d_col_ptr, gpu_setup.d_tile_boundaries,
        tiling.max_node_degree, gpu_setup.blocks_total1, gpu_setup.blocks_max1,
        gpu_setup.blocks_total2, gpu_setup.blocks_max2, gpu_setup.d_total1,
        gpu_setup.d_max1, gpu_setup.d_total2, gpu_setup.d_max2);
  });
}

template <int MAX_THREADS_SM, int BLK_SM>
void Distance2Copy(nvbench::state& state,
                   nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                      nvbench::enum_type<BLK_SM>>) {
  constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
  auto kernel = copyKernelDist2<THREADS, BLK_SM, int>;

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

void Distance1cusparse(nvbench::state &state) {
    cusparseHandle_t handle;
	cusparseCreate(&handle);
	cusparseMatDescr_t descG;
	// creates descriptor for 0-based indexing and general matrix by default
	cusparseCreateMatDescr(&descG);

	cusparseColorInfo_t info;
	cusparseCreateColorInfo(&info);

	// fraction of vertices that has to be colored iteratively before falling back
	// to giving every leftover node an unique color
	constexpr double fraction = 1.0;

	int num_colors; // will be updated by cusparse

    MatLoader& mat_loader = MatLoader::getInstance();

    double* d_val_ptr;
    int* d_col_ptr;
    int* d_row_ptr; 
    cudaMalloc((void**)&d_val_ptr, mat_loader.row_ptr[mat_loader.m_rows] * sizeof(double));
    cudaMalloc((void**)&d_col_ptr, mat_loader.row_ptr[mat_loader.m_rows] * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (mat_loader.m_rows + 1) * sizeof(int));
    
    cudaMemcpy(d_val_ptr, mat_loader.val_ptr, mat_loader.row_ptr[mat_loader.m_rows] * sizeof(double),
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, mat_loader.col_ptr, mat_loader.row_ptr[mat_loader.m_rows] * sizeof(int),
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, mat_loader.row_ptr, mat_loader.m_rows * sizeof(int),
                cudaMemcpyHostToDevice);

    thrust::device_vector<int> coloring(mat_loader.row_ptr[mat_loader.m_rows]);

    add_MatInfo(state);
    size_t in_elem = mat_loader.m_rows + mat_loader.row_ptr[mat_loader.m_rows];
    state.add_global_memory_reads<int>(in_elem);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {		
		    cusparseDcsrcolor(handle,
		                      mat_loader.m_rows,
		                      mat_loader.row_ptr[mat_loader.m_rows],
		                      descG,
		                      d_val_ptr,
		                      d_row_ptr,
		                      d_col_ptr,
		                      &fraction,
		                      &num_colors,
		                      thrust::raw_pointer_cast(coloring.data()),
		                      nullptr, // don't need reordering
		                      info);
    });

    state.add_element_count(num_colors, "Colors");
	cusparseDestroyColorInfo(info);
	cusparseDestroyMatDescr(descG);
	cusparseDestroy(handle);
    cudaFree(d_val_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_row_ptr);
}


NVBENCH_BENCH_TYPES(Distance1, NVBENCH_TYPE_AXES(THREADS_SM_Fine, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Distance1Copy, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Distance1CopyFence, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Distance2, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Distance2Bank, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

// NVBENCH_BENCH_TYPES(Distance2Copy, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
//     .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
//     .add_int64_axis(SM_ShMem_key, SM_ShMem_range);
// NVBENCH_BENCH(Distance1cusparse);
