#include <nvbench/nvbench.cuh>

#include "V2/warp_hash_coop.cuh"
#include "defines.hpp"
#include "coloring_counters.cuh"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "cli_bench.cu"
#include "bench_util.cuh"

void D1Coop(nvbench::state& state) {
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
                        (void *)&gpu_setup.d_max1,
                        };

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



void D2Coop(nvbench::state& state) {
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
            cudaLaunchCooperativeKernel((void*)kernel, 
                                        gridSize, 
                                        blockSize, 
                                        kernelArgs, 
                                        shMem_bytes, 
                                        NULL);
        });  
}


NVBENCH_BENCH(D1Coop);
NVBENCH_BENCH(D2Coop);