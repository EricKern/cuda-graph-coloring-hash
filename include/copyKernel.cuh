#pragma once
#include "coloring.cuh"
#include "coloringCounters.cuh"

using namespace apa22_coloring;

template <typename IndexT>
__global__
void copyKernelDist1(IndexT* row_ptr,  // global mem
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
                int* start_addr = soa_total->m[k] + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_total[k].m[i] = val;
                }
            }
        }

        // max collisions
        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int* start_addr = soa_total->m[k] + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_max[k].m[i] = val;
                }
            }
        }
    }
}

template <typename IndexT>
__global__
void copyKernelDist2(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     int tile_max_nodes,
                     int tile_max_edges,
                     SOACounters* soa_total1,
                     SOACounters* soa_max1,
                     SOACounters* soa_total2,
                     SOACounters* soa_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {
    extern __shared__ IndexT shMem[];
    IndexT* shMemRows = shMem;                        
    IndexT* shMemCols = &shMemRows[tile_max_nodes+1];
  
    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

    const IndexT partNr = blockIdx.x;  

    for (int k = 0; k < hash_params.len; k++){
        for (int i = 0; i < num_bit_widths; i++){
            if (threadIdx.x == 0){
                soa_total1->m[k][partNr + i * gridDim.x] = i + k;
                soa_max1->m[k][partNr + i * gridDim.x] = i - k;
                soa_total2->m[k][partNr + i * gridDim.x] = i + k;
                soa_max2->m[k][partNr + i * gridDim.x] = i - k;
            }
        }
    }

    if (blockIdx.x == 0) {
        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int* start_addr = soa_total1->m[k] + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }              
                if (threadIdx.x == 0){
                    d_total1[k].m[i] = val;
                }
            }
        }

        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int* start_addr = soa_max1->m[k] + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_max1[k].m[i] = val;
                }
            }
        }
        
        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int* start_addr = soa_max2->m[k] + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_max2[k].m[i] = val;
                }
            }
        }

        for (int k = 0; k < hash_params.len; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int* start_addr = soa_total2->m[k] + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_total2[k].m[i] = val;
                }
            }
        }
    }
}