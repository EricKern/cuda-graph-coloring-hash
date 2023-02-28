#include <nvbench/nvbench.cuh>

#include "V2/WarpHashCoop.cuh"
#include "V2/WarpHash.cuh"
#include "defines.hpp"
#include "coloring_counters.cuh"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "cli_bench.cu"
#include "bench_util.cuh"

void D1Warp(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = D1warp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D1_Warp,
                BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                64 * 1024);

    GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                        mat_loader.col_ptr,
                        tiling.tile_boundaries.get(), 
                        tiling.n_tiles);

    size_t shMem_bytes = tiling.tile_target_mem;
   
    dim3 blockSize(THREADS);
    dim3 gridSize(tiling.n_tiles);

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

void D1WarpCoop(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = coloring1coopWarp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D1_Warp,
                BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                64 * 1024);

    GPUSetupD1 gpu_setup(mat_loader.row_ptr,
                        mat_loader.col_ptr,
                        tiling.tile_boundaries.get(), 
                        tiling.n_tiles);

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
                        (void *)&gpu_setup.blocks_total1,
                        (void *)&gpu_setup.blocks_max1,
                        (void *)&gpu_setup.d_total1,
                        (void *)&gpu_setup.d_max1};

    state.exec(
        [&](nvbench::launch& launch) {
            cudaLaunchCooperativeKernel((void*)kernel, 
                                        gridSize, 
                                        blockSize, 
                                        kernelArgs, 
                                        shMem_bytes, 
                                        NULL);
        });  
}

void D2Warp(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = D2warp_Conflicts<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D2_Warp, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                    reinterpret_cast<void*>(kernel),
                    64 * 1024);
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

void D2WarpCoop(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = coloring2coopWarp<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D2_Warp,
                BLK_SM,
                mat_loader.row_ptr,
                mat_loader.m_rows,
                reinterpret_cast<void*>(kernel),
                64 * 1024);

    GPUSetupD2 gpu_setup(mat_loader.row_ptr,
                        mat_loader.col_ptr,
                        tiling.tile_boundaries.get(), 
                        tiling.n_tiles);

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
                        (void *)&gpu_setup.blocks_max1,
                        (void *)&gpu_setup.blocks_total2,
                        (void *)&gpu_setup.blocks_max2,
                        (void *)&gpu_setup.d_total1,
                        (void *)&gpu_setup.d_max1,
                        (void *)&gpu_setup.d_total2,
                        (void *)&gpu_setup.d_max2};

    state.exec(
        [&](nvbench::launch& launch) {
            cudaLaunchCooperativeKernel((void*)kernel, 
                                        gridSize, 
                                        blockSize, 
                                        kernelArgs, 
                                        shMem_bytes, 
                                        NULL);
        });  
}

NVBENCH_BENCH(D1Warp);
NVBENCH_BENCH(D1WarpCoop);
NVBENCH_BENCH(D2Warp);
NVBENCH_BENCH(D2WarpCoop);