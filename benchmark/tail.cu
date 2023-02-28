#include <nvbench/nvbench.cuh>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

#include "V2/WarpHashCoop.cuh"
#include "V2/WarpHashCoopTime.cuh"
#include "defines.hpp"
#include "coloring_counters.cuh"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "cli_bench.cu"
#include "bench_util.cuh"

void WarmUp(nvbench::state& state) {
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

void D1WarpCoopTail(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = coloring1coopWarpTail<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

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
    
    std::vector<clock_t> durations(gridSize.x, 0);
    clock_t* d_durations;
    cudaMalloc((void**)&d_durations, gridSize.x * sizeof(clock_t));
    cudaMemcpy(d_durations, durations.data(), durations.size() * sizeof(clock_t), cudaMemcpyHostToDevice);
     

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
                        (void *)&d_durations};

    state.exec(
        [&](nvbench::launch& launch) {
            cudaLaunchCooperativeKernel((void*)kernel, 
                                        gridSize, 
                                        blockSize, 
                                        kernelArgs, 
                                        shMem_bytes, 
                                        NULL);
        });

    cudaMemcpy(durations.data(), d_durations, gridSize.x * sizeof(clock_t), cudaMemcpyDeviceToHost);
    auto min = *std::min_element(durations.begin(), durations.end());
    auto max = *std::max_element(durations.begin(), durations.end());
    double sum = std::accumulate(durations.begin(), durations.end(), clock_t{0});
    auto mean = sum / gridSize.x;

    state.add_element_count(min, "min_counts");
    state.add_element_count(max, "max_counts");
    state.add_element_count(mean, "mean_counts");
}

void D2WarpCoopTail(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = coloring2coopWarpTail<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D2_Warp, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                    reinterpret_cast<void*>(kernel),
                    64 * 1024);
    GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                        tiling.tile_boundaries.get(), tiling.n_tiles);

    size_t shMem_bytes = tiling.tile_target_mem;
    dim3 blockSize(THREADS);

    auto grid = get_coop_grid_size((void*)kernel,
                                  blockSize.x, shMem_bytes);
    dim3 gridSize(grid);

    add_MatInfo(state);
    add_IOInfo(state, gridSize.x);

    std::vector<clock_t> durations(gridSize.x, 0);
    clock_t* d_durations;
    cudaMalloc((void**)&d_durations, gridSize.x * sizeof(clock_t));
    cudaMemcpy(d_durations, durations.data(), durations.size() * sizeof(clock_t), cudaMemcpyHostToDevice);

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
                        (void *)&gpu_setup.d_max2,
                        (void *)&d_durations};

    state.exec(
        [&](nvbench::launch& launch) {
            cudaLaunchCooperativeKernel((void*)kernel, 
                                        gridSize, 
                                        blockSize, 
                                        kernelArgs, 
                                        shMem_bytes, 
                                        NULL);
        });

    cudaMemcpy(durations.data(), d_durations, gridSize.x * sizeof(clock_t), cudaMemcpyDeviceToHost);
    auto min = *std::min_element(durations.begin(), durations.end());
    auto max = *std::max_element(durations.begin(), durations.end());
    double sum = std::accumulate(durations.begin(), durations.end(), clock_t{0});
    auto mean = sum / gridSize.x;

    state.add_element_count(min, "min_counts");
    state.add_element_count(max, "max_counts");
    state.add_element_count(mean, "mean_counts");

}

void D2WarpCoopTailBig(nvbench::state& state) {
    constexpr int THREADS = 1024;
    constexpr int BLK_SM = 1;
    auto kernel = coloring2coopWarpTailBig<THREADS, BLK_SM, 16, 3, int, char, 8, 3, int>;

    MatLoader& mat_loader = MatLoader::getInstance();
    Tiling tiling(D2_Warp, BLK_SM, mat_loader.row_ptr, mat_loader.m_rows,
                    reinterpret_cast<void*>(kernel),
                    64 * 1024);
    GPUSetupD2 gpu_setup(mat_loader.row_ptr, mat_loader.col_ptr,
                        tiling.tile_boundaries.get(), tiling.n_tiles);

    size_t shMem_bytes = tiling.tile_target_mem;
    dim3 blockSize(THREADS);

    auto grid = get_coop_grid_size((void*)kernel,
                                  blockSize.x, shMem_bytes);
    dim3 gridSize(grid);

    add_MatInfo(state);
    add_IOInfo(state, gridSize.x);

    std::vector<clock_t> durations(gridSize.x, 0);
    clock_t* d_durations;
    cudaMalloc((void**)&d_durations, gridSize.x * sizeof(clock_t));
    cudaMemcpy(d_durations, durations.data(), durations.size() * sizeof(clock_t), cudaMemcpyHostToDevice);

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
                        (void *)&gpu_setup.d_max2,
                        (void *)&d_durations};

    state.exec(
        [&](nvbench::launch& launch) {
            cudaLaunchCooperativeKernel((void*)kernel, 
                                        gridSize, 
                                        blockSize, 
                                        kernelArgs, 
                                        shMem_bytes, 
                                        NULL);
        });

    cudaMemcpy(durations.data(), d_durations, gridSize.x * sizeof(clock_t), cudaMemcpyDeviceToHost);
    auto min = *std::min_element(durations.begin(), durations.end());
    auto max = *std::max_element(durations.begin(), durations.end());
    double sum = std::accumulate(durations.begin(), durations.end(), clock_t{0});
    auto mean = sum / gridSize.x;

    state.add_element_count(min, "min_counts");
    state.add_element_count(max, "max_counts");
    state.add_element_count(mean, "mean_counts");

}

NVBENCH_BENCH(WarmUp);
NVBENCH_BENCH(D1WarpCoopTail);
NVBENCH_BENCH(D2WarpCoopTail);
NVBENCH_BENCH(D2WarpCoopTailBig);