#pragma once
#include "coloring.cuh"
#include "coloringCounters.cuh"

using namespace apa22_coloring;

template <typename IndexT>
__global__
void copyKernel(IndexT* row_ptr,  // global mem
                IndexT* col_ptr,  // global mem
                IndexT* tile_boundaries,
                int tile_max_nodes,
                int tile_max_edges,
                SOACounters* soa_total,
                SOACounters* soa_max,
                Counters* d_total,
                Counters* d_max) {
                    
    const IndexT partNr = blockIdx.x;

    extern __shared__ IndexT shMem[];
    IndexT* shMemRows = shMem;
    IndexT* shMemCols = &shMemRows[tile_max_nodes + 1];

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

    // local collisions
    for (int k = 0; k < hash_params.len; k++){
        for (int i = 0; i < num_bit_widths; i++){
            if (threadIdx.x == 0){
                soa_total->m[k][partNr + i * gridDim.x] = k * i;
                soa_max->m[k][partNr + i * gridDim.x] = i + k;
            }
        }
    }

    // final reduction of last block
    if (blockIdx.x == 0) {
        // total collisions
        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                if (threadIdx.x == 0){
                    d_total[k].m[i] = i + k;
                }
            }
        }

        // max collisions
        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                if (threadIdx.x == 0){
                    d_max[k].m[i] = i + k;
                }
            }
        }
    }
}