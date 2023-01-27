#include <nvbench/nvbench.cuh>

#include "kernel_setup.hpp"
#include "defines.hpp"
#include "coloringCounters.cuh"
#include "copyKernel.cuh"
#include "init.hpp"
#include "setup.cuh"
#include "partKernels.cuh"

// static constexpr const char* Mat = def::Mat2_Cluster;
// // static constexpr const char* Mat = def::Mat3;
// static constexpr int STEPS = 300;

using namespace apa22_coloring;

void Dist1(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist1();   
    GPUSetupD1 d1{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring1Kernel<<<gridSize, blockSize, params.shMemSize, launch.get_stream()>>>(
            d1.d_row_ptr, d1.d_col_ptr, d1.d_tile_boundaries, params.maxNodes, params.maxEdges, d1.d_soa_total1,
            d1.d_soa_max1, d1.d_total1, d1.d_max1);
    });

}

void Dist1OnlyLoad(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist1();   
    GPUSetupD1 d1{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring1OnlyLoad<<<gridSize, blockSize, params.shMemSize, launch.get_stream()>>>(
            d1.d_row_ptr, d1.d_col_ptr, d1.d_tile_boundaries, params.maxNodes, params.maxEdges, d1.d_soa_total1,
            d1.d_soa_max1, d1.d_total1, d1.d_max1);
    });

}

void Dist1NoReduceNoWrite(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist1();   
    GPUSetupD1 d1{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring1NoReduceNoWrite<<<gridSize, blockSize, params.shMemSize, launch.get_stream()>>>(
            d1.d_row_ptr, d1.d_col_ptr, d1.d_tile_boundaries, params.maxNodes, params.maxEdges, d1.d_soa_total1,
            d1.d_soa_max1, d1.d_total1, d1.d_max1);
    });

}

void Dist1NoReduce(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist1();   
    GPUSetupD1 d1{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring1NoReduce<<<gridSize, blockSize, params.shMemSize, launch.get_stream()>>>(
            d1.d_row_ptr, d1.d_col_ptr, d1.d_tile_boundaries, params.maxNodes, params.maxEdges, d1.d_soa_total1,
            d1.d_soa_max1, d1.d_total1, d1.d_max1);
    });

}

void Dist1NoLastReduce(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist1();   
    GPUSetupD1 d1{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring1NoLastReduce<<<gridSize, blockSize, params.shMemSize, launch.get_stream()>>>(
            d1.d_row_ptr, d1.d_col_ptr, d1.d_tile_boundaries, params.maxNodes, params.maxEdges, d1.d_soa_total1,
            d1.d_soa_max1, d1.d_total1, d1.d_max1);
    });

}

void Dist2(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2Kernel<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}

void Dist2OnlyLoad(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2OnlyLoad<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}

void Dist2NoReduceNoWriteOnlyD1(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2NoReduceNoWriteOnlyD1<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}

void Dist2NoReduceNoWriteOnlyD2(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2NoReduceNoWriteOnlyD2<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}

void Dist2NoReduceNoWrite(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2NoReduceNoWrite<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}

void Dist2NoReduce(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2NoReduce<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}

void Dist2NoLastReduce(nvbench::state &state){

    Dist params = Init<>::Instance()->getDist2();   
    GPUSetupD2 d2{params.rowPtr, params.colPtr, params.ndc, params.nTiles};
    dim3 gridSize(params.nTiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2NoLastReduce<<<gridSize, blockSize, params.shMemSize>>>(d2.d_row_ptr,
                                                            d2.d_col_ptr,
                                                            d2.d_tile_boundaries,
                                                            d2.max_nodes,
                                                            d2.max_edges,
                                                            d2.d_soa_total1,
                                                            d2.d_soa_max1,
                                                            d2.d_soa_total2,
                                                            d2.d_soa_max2,
                                                            d2.d_total1,
                                                            d2.d_max1,
                                                            d2.d_total2,
                                                            d2.d_max2);
    });
}



NVBENCH_BENCH(Dist1);
NVBENCH_BENCH(Dist1OnlyLoad);
NVBENCH_BENCH(Dist1NoReduceNoWrite);
NVBENCH_BENCH(Dist1NoReduce);
NVBENCH_BENCH(Dist1NoLastReduce);
NVBENCH_BENCH(Dist2);
NVBENCH_BENCH(Dist2OnlyLoad);
NVBENCH_BENCH(Dist2NoReduceNoWriteOnlyD1);
NVBENCH_BENCH(Dist2NoReduceNoWriteOnlyD2);
NVBENCH_BENCH(Dist2NoReduceNoWrite);
NVBENCH_BENCH(Dist2NoReduce);
NVBENCH_BENCH(Dist2NoLastReduce);