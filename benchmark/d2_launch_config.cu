#include <nvbench/nvbench.cuh>
#include <cusparse.h>
#include <thrust/device_vector.h>

#include "coloring.cuh"
#include "defines.hpp"
#include "coloring_counters.cuh"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "cli_bench.cu"
#include "bench_util.cuh"
#include "V2/warp_hash.cuh"
#include "V2/warp_hash_coop.cuh"

using BLOCKS_SM = nvbench::enum_type_list<1, 2, 4>;
using THREADS_SM = nvbench::enum_type_list<256, 384, 512, 640, 768, 896, 1024>;
static constexpr const char* Blocks_SM_key = "BLK_SM";
static constexpr const char* Threads_SM_key = "THREADS_SM";
static constexpr const char* SM_ShMem_key = "Smem_SM";
std::vector<nvbench::int64_t> SM_ShMem_range = {64*1024};


template <int MAX_THREADS_SM, int BLK_SM>
void D2WarpConflict(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
    constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
    auto kernel = D2warp_Conflicts<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D2_Warp, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
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
                    gpu_setup.blocks_total2,
                    gpu_setup.blocks_max2,
                    gpu_setup.d_total2,
                    gpu_setup.d_max2);
        });
}

template <int MAX_THREADS_SM, int BLK_SM>
void D2Coop(nvbench::state& state,
               nvbench::type_list<nvbench::enum_type<MAX_THREADS_SM>,
                                  nvbench::enum_type<BLK_SM>>) {
    constexpr int THREADS = MAX_THREADS_SM / BLK_SM;
    auto kernel = coloring2coopWarp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D2_Warp, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                    reinterpret_cast<void*>(kernel),
                    state.get_int64(SM_ShMem_key));
    GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                        tiling.tile_boundaries.get(), tiling.n_tiles);

    size_t shMem_bytes = tiling.tile_target_mem;
    dim3 blockSize(THREADS);
    auto grid = get_coop_grid_size((void*)kernel,
                                    blockSize.x, shMem_bytes);
    dim3 gridSize(grid);

    add_MatInfo(state);
    add_IOInfo(state, gridSize.x);

    void *kernelArgs[] = {(void *)&gpu_setup.d_row_ptr,
                            (void *)&gpu_setup.d_col_ptr,
                            (void *)&gpu_setup.d_tile_boundaries,
                            (void *)&tiling.n_tiles,
                            (void *)&tiling.max_node_degree,
                            (void *)&gpu_setup.blocks_total1,
                            (void *)&gpu_setup.blocks_total2,
                            (void *)&gpu_setup.blocks_max1,
                            (void *)&gpu_setup.blocks_max2,
                            (void *)&gpu_setup.d_total1,
                            (void *)&gpu_setup.d_max1,
                            (void *)&gpu_setup.d_total2,
                            (void *)&gpu_setup.d_max2
                            };

    state.exec(
        [&](nvbench::launch& launch) {
            cudaLaunchCooperativeKernel((void*)kernel, gridSize, blockSize, kernelArgs, shMem_bytes, NULL);
        });
}

NVBENCH_BENCH_TYPES(D2WarpConflict, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);

NVBENCH_BENCH_TYPES(D2Coop, NVBENCH_TYPE_AXES(THREADS_SM, BLOCKS_SM))
    .set_type_axes_names({Threads_SM_key, Blocks_SM_key})
    .add_int64_axis(SM_ShMem_key, SM_ShMem_range);


