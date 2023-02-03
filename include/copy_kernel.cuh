#pragma once
#include "coloring.cuh"
#include "coloring_counters.cuh"

using namespace apa22_coloring;

template <int THREADS,
          int BLK_SM,
          typename IndexT>
__global__
__launch_bounds__(THREADS, BLK_SM)
void copyKernelDist1(IndexT* row_ptr,  // global mem
                    IndexT* col_ptr,  // global mem
                    IndexT* tile_boundaries,
                    Counters::value_type* blocks_total1,
                    Counters::value_type* blocks_max1,
                    Counters* d_total,
                    Counters* d_max) {
                    
    const int partNr = blockIdx.x;
    const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];

    extern __shared__ IndexT shMem[];
    IndexT* shMemRows = shMem;                      // n_tileNodes +1 elements
    IndexT* shMemCols = &shMemRows[n_tileNodes+1];

    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

    // local collisions
    for (int k = 0; k < num_hashes; k++){
        for (int i = 0; i < num_bit_widths; i++){
            if (threadIdx.x == 0){
                const int elem_p_hash_fn = num_bit_widths * gridDim.x;
                //        hash segment         bit_w segment   block value
                //        __________________   _____________   ___________
                int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
                blocks_total1[idx] = k * i;
                blocks_max1[idx] = k + i;
            }
        }
    }

    const int elem_p_hash_fn = num_bit_widths * gridDim.x;

    // final reduction of last block
    if (blockIdx.x == 0) {
        // total collisions
        for (int k = 0; k < num_hashes; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int hash_offset = k * elem_p_hash_fn;
                int* start_addr = blocks_total1 + hash_offset + i * gridDim.x;
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
        for (int k = 0; k < num_hashes; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int hash_offset = k * elem_p_hash_fn;
                int* start_addr = blocks_max1 + hash_offset + i * gridDim.x;
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

template <int THREADS,
          int BLK_SM,
          typename IndexT>
__global__
__launch_bounds__(THREADS, BLK_SM)
void copyKernelDist2(IndexT* row_ptr,  // global mem
                     IndexT* col_ptr,  // global mem
                     IndexT* tile_boundaries,
                     Counters::value_type* blocks_total1,
                     Counters::value_type* blocks_max1,
                     Counters::value_type* blocks_total2,
                     Counters::value_type* blocks_max2,
                     Counters* d_total1,
                     Counters* d_max1,
                     Counters* d_total2,
                     Counters* d_max2) {
    
    const int partNr = blockIdx.x;
    const int n_tileNodes = tile_boundaries[partNr+1] - tile_boundaries[partNr];
    
    IndexT part_offset = tile_boundaries[partNr];   // offset in row_ptr array
    const int n_tileEdges =
    row_ptr[n_tileNodes + part_offset] - row_ptr[part_offset];

    // shared mem size: 2 * (n_rows + 1 + n_cols)
    extern __shared__ IndexT shMem[];
    IndexT* shMemRows = shMem;                        // n_tileNodes + 1 elements
    IndexT* shMemCols = &shMemRows[n_tileNodes+1];    // n_tileEdges elements
  
    Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries); 

    for (int k = 0; k < num_hashes; k++){
        for (int i = 0; i < num_bit_widths; i++){
            if (threadIdx.x == 0){
                const int elem_p_hash_fn = num_bit_widths * gridDim.x;
                //        hash segment         bit_w segment   block value
                //        __________________   _____________   ___________
                int idx = k * elem_p_hash_fn + i * gridDim.x + blockIdx.x;
                blocks_total1[idx] = k * i;
                blocks_max1[idx] = k + i;
                blocks_total2[idx] = k - i;
                blocks_max2[idx] = i - k;
            }
        }
    }

    const int elem_p_hash_fn = num_bit_widths * gridDim.x;

    if (blockIdx.x == 0) {
        for (int k = 0; k < num_hashes; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int hash_offset = k * elem_p_hash_fn;
                int* start_addr = blocks_total1 + hash_offset + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_total1[k].m[i] = val;
                }
            }
        }

        for (int k = 0; k < num_hashes; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int hash_offset = k * elem_p_hash_fn;
                int* start_addr = blocks_max1 + hash_offset + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_max1[k].m[i] = val;
                }
            }
        }
        
        for (int k = 0; k < num_hashes; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int hash_offset = k * elem_p_hash_fn;
                int* start_addr = blocks_total2 + hash_offset + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_total2[k].m[i] = val;
                }
            }
        }

        for (int k = 0; k < num_hashes; k++){
            for (int i = 0; i < num_bit_widths; i++){
                int hash_offset = k * elem_p_hash_fn;
                int* start_addr = blocks_max2 + hash_offset + i * gridDim.x;
                int val = 0;
                for (int j = threadIdx.x; j < gridDim.x; j+=blockDim.x){
                    val = start_addr[i];
                }
                if (threadIdx.x == 0){
                    d_max2[k].m[i] = val;
                }
            }
        }
    }
}