#include <nvbench/nvbench.cuh>

#include "kernel_setup.hpp"
#include "defines.hpp"
#include "coloringCounters.cuh"
#include "copyKernel.cuh"

static constexpr const char* Mat = def::Mat3_Cluster;
static constexpr int STEPS = 300;

using namespace apa22_coloring;

template <bool dist2 = false>
void initBenchmark(const char* Matrix, int* &d_row_ptr, int* &d_col_ptr, int* &d_tile_boundaries,
                    SOACounters* &d_soa_total, SOACounters* &d_soa_max, SOACounters &h_soa_total,
                    SOACounters &h_soa_max, Counters* &d_total, Counters* &d_max, int &max_nodes,
                    int &max_edges, int &shMem_size_bytes, int &number_of_tiles, size_t &size){
    int* row_ptr;
    int* col_ptr;
    double* val_ptr;
    int m_rows;
    int* ndc_;

    if constexpr (dist2){
        kernel_setup<true>(Matrix, row_ptr, col_ptr, val_ptr, ndc_, m_rows, number_of_tiles, shMem_size_bytes, STEPS);
    }
    else{
        kernel_setup(Matrix, row_ptr, col_ptr, val_ptr, ndc_, m_rows, number_of_tiles, shMem_size_bytes, STEPS);
    }
  
    size_t row_ptr_len = m_rows + 1;
    size_t col_ptr_len = size = row_ptr[m_rows];
    size_t tile_bound_len = number_of_tiles + 1;

    cudaMalloc((void**)&d_row_ptr, row_ptr_len * sizeof(int));
    cudaMalloc((void**)&d_col_ptr, col_ptr_len * sizeof(int));
    cudaMalloc((void**)&d_tile_boundaries, tile_bound_len * sizeof(int));


    cudaMemcpy(d_row_ptr, row_ptr, row_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, col_ptr, col_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_boundaries, ndc_, tile_bound_len * sizeof(int),
             cudaMemcpyHostToDevice);

    // For each bit_width we allocate a counter for each block and for each hash function
    for (int i = 0; i < hash_params.len; ++i) {
        cudaMalloc((void**)&(h_soa_total.m[i]), num_bit_widths * number_of_tiles * sizeof(int));
    }
    for (int i = 0; i < hash_params.len; ++i) {
        cudaMalloc((void**)&(h_soa_max.m[i]), num_bit_widths * number_of_tiles * sizeof(int));
    }
   
    cudaMalloc((void**)&d_soa_total, sizeof(SOACounters));

    cudaMalloc((void**)&d_soa_max, sizeof(SOACounters));

    cudaMemcpy(d_soa_total, &h_soa_total, sizeof(SOACounters),
             cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa_max, &h_soa_max, sizeof(SOACounters),
             cudaMemcpyHostToDevice);

 
    cudaMalloc((void**)&d_total, hash_params.len * sizeof(Counters));
    cudaMalloc((void**)&d_max, hash_params.len * sizeof(Counters));
  
    get_MaxTileSize(number_of_tiles, ndc_, row_ptr, &max_nodes, &max_edges);

    delete[] row_ptr;
    delete[] col_ptr;
	delete[] val_ptr;
	delete[] ndc_;
}

void initBenchmarkD2(SOACounters &h_soa_total2, SOACounters &h_soa_max2, SOACounters* &d_soa_total2,
                    SOACounters* &d_soa_max2, Counters* &d_total2, Counters* &d_max2, int number_of_tiles){
    for (int i = 0; i < hash_params.len; ++i) {
      cudaMalloc((void**)&(h_soa_total2.m[i]), num_bit_widths * number_of_tiles * sizeof(int));
    }
    for (int i = 0; i < hash_params.len; ++i) {
      cudaMalloc((void**)&(h_soa_max2.m[i]), num_bit_widths * number_of_tiles * sizeof(int));
    }

    cudaMalloc((void**)&d_soa_total2, sizeof(SOACounters));
    cudaMalloc((void**)&d_soa_max2, sizeof(SOACounters));

    cudaMemcpy(d_soa_total2, &h_soa_total2, sizeof(SOACounters),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_soa_max2, &h_soa_max2, sizeof(SOACounters),
              cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_total2, hash_params.len * sizeof(Counters));
    cudaMalloc((void**)&d_max2, hash_params.len * sizeof(Counters));
}

void copyBenchDist1(nvbench::state &state){
    int* d_row_ptr;
    int* d_col_ptr;
    int* d_tile_boundaries;
    SOACounters* d_soa_total;
    SOACounters* d_soa_max;
    SOACounters h_soa_total;
    SOACounters h_soa_max;
    Counters* d_total;
    Counters* d_max;
    int shMem_size_bytes;
    int number_of_tiles;
    int max_nodes;
    int max_edges;
    size_t size;

    state.set_timeout(-1);
    initBenchmark(Mat, d_row_ptr, d_col_ptr, d_tile_boundaries, d_soa_total, d_soa_max,
                h_soa_total, h_soa_max, d_total, d_max, max_nodes, max_edges, shMem_size_bytes,
                number_of_tiles, size);
    
    state.add_element_count(size, "Elements");
    state.add_global_memory_reads<int>(size);
    //state.add_global_memory_writes<Counters>(size);
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    state.collect_loads_efficiency();
    state.collect_stores_efficiency();

    dim3 gridSize(number_of_tiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        copyKernelDist1<<<gridSize, blockSize, shMem_size_bytes, launch.get_stream()>>>(
            d_row_ptr, d_col_ptr, d_tile_boundaries, max_nodes, max_edges, d_soa_total,
            d_soa_max, d_total, d_max);
    });

    for (int i = 0; i < hash_params.len; ++i) {
        cudaFree(h_soa_total.m[i]);
        cudaFree(h_soa_max.m[i]);
    }
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_tile_boundaries);
    cudaFree(d_soa_total);
    cudaFree(d_soa_max);
    cudaFree(d_total);
    cudaFree(d_max);
}

void coloring1Bench(nvbench::state &state){
    int* d_row_ptr;
    int* d_col_ptr;
    int* d_tile_boundaries;
    SOACounters* d_soa_total;
    SOACounters* d_soa_max;
    SOACounters h_soa_total;
    SOACounters h_soa_max;
    Counters* d_total;
    Counters* d_max;
    int shMem_size_bytes;
    int number_of_tiles;
    int max_nodes;
    int max_edges;
    size_t size;

    state.set_timeout(-1);
    initBenchmark(Mat, d_row_ptr, d_col_ptr, d_tile_boundaries, d_soa_total, d_soa_max,
                h_soa_total, h_soa_max, d_total, d_max, max_nodes, max_edges, shMem_size_bytes,
                number_of_tiles, size);
    
    state.add_element_count(size, "Elements");
    state.add_global_memory_reads<int>(size);
    //state.add_global_memory_writes<Counters>(size);
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    state.collect_loads_efficiency();
    state.collect_stores_efficiency();

    dim3 gridSize(number_of_tiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring1Kernel<<<gridSize, blockSize, shMem_size_bytes, launch.get_stream()>>>(
            d_row_ptr, d_col_ptr, d_tile_boundaries, max_nodes, max_edges, d_soa_total,
            d_soa_max, d_total, d_max);
    });

    for (int i = 0; i < hash_params.len; ++i) {
        cudaFree(h_soa_total.m[i]);
        cudaFree(h_soa_max.m[i]);
    }
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_tile_boundaries);
    cudaFree(d_soa_total);
    cudaFree(d_soa_max);
    cudaFree(d_total);
    cudaFree(d_max);
}

void copyBenchDist2(nvbench::state &state){
    int* d_row_ptr;
    int* d_col_ptr;
    int* d_tile_boundaries;
    SOACounters* d_soa_total1;
    SOACounters* d_soa_max1;
    SOACounters* d_soa_total2;
    SOACounters* d_soa_max2;
    SOACounters h_soa_total1;
    SOACounters h_soa_max1;
    SOACounters h_soa_total2;
    SOACounters h_soa_max2;
    Counters* d_total1;
    Counters* d_max1;
    Counters* d_total2;
    Counters* d_max2;
    int shMem_size_bytes;
    int number_of_tiles;
    int max_nodes;
    int max_edges;
    size_t size;

    state.set_timeout(-1);
    initBenchmark<true>(Mat, d_row_ptr, d_col_ptr, d_tile_boundaries, d_soa_total1, d_soa_max1,
                h_soa_total1, h_soa_max1, d_total1, d_max1, max_nodes, max_edges, shMem_size_bytes,
                number_of_tiles, size);
    initBenchmarkD2(h_soa_total2, h_soa_max2, d_soa_total2, d_soa_max2, d_total2, d_max2,
                    number_of_tiles);
    
    state.add_element_count(size, "Elements");
    state.add_global_memory_reads<int>(size);
    //state.add_global_memory_writes<Counters>(size);
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    state.collect_loads_efficiency();
    state.collect_stores_efficiency();

    dim3 gridSize(number_of_tiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        copyKernelDist2<<<gridSize, blockSize, shMem_size_bytes, launch.get_stream()>>>(d_row_ptr,
                        d_col_ptr, d_tile_boundaries, max_nodes, max_edges, d_soa_total1, d_soa_max1,
                        d_soa_total2, d_soa_max2, d_total1, d_max1, d_total2, d_max2);
    });

    for (int i = 0; i < hash_params.len; ++i) {
        cudaFree(h_soa_total1.m[i]);
        cudaFree(h_soa_max1.m[i]);
        cudaFree(h_soa_total2.m[i]);
        cudaFree(h_soa_max2.m[i]);
    }
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_tile_boundaries);
    cudaFree(d_soa_total1);
    cudaFree(d_soa_total2);
    cudaFree(d_soa_max1);
    cudaFree(d_soa_max2);
    cudaFree(d_total1);
    cudaFree(d_total2);
    cudaFree(d_max1);
    cudaFree(d_max2);
}

void coloring2Bench(nvbench::state &state){
    int* d_row_ptr;
    int* d_col_ptr;
    int* d_tile_boundaries;
    SOACounters* d_soa_total1;
    SOACounters* d_soa_max1;
    SOACounters* d_soa_total2;
    SOACounters* d_soa_max2;
    SOACounters h_soa_total1;
    SOACounters h_soa_max1;
    SOACounters h_soa_total2;
    SOACounters h_soa_max2;
    Counters* d_total1;
    Counters* d_max1;
    Counters* d_total2;
    Counters* d_max2;
    int shMem_size_bytes;
    int number_of_tiles;
    int max_nodes;
    int max_edges;
    size_t size;

    state.set_timeout(-1);
    initBenchmark<true>(Mat, d_row_ptr, d_col_ptr, d_tile_boundaries, d_soa_total1, d_soa_max1,
                h_soa_total1, h_soa_max1, d_total1, d_max1, max_nodes, max_edges, shMem_size_bytes,
                number_of_tiles, size);
    initBenchmarkD2(h_soa_total2, h_soa_max2, d_soa_total2, d_soa_max2, d_total2, d_max2,
                    number_of_tiles);
    
    state.add_element_count(size, "Elements");
    state.add_global_memory_reads<int>(size);
    //state.add_global_memory_writes<Counters>(size);
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    state.collect_loads_efficiency();
    state.collect_stores_efficiency();

    dim3 gridSize(number_of_tiles);
    dim3 blockSize(THREADS);

    state.exec([&](nvbench::launch &launch){
        coloring2Kernel<<<gridSize, blockSize, shMem_size_bytes, launch.get_stream()>>>(d_row_ptr,
                        d_col_ptr, d_tile_boundaries, max_nodes, max_edges, d_soa_total1, d_soa_max1,
                        d_soa_total2, d_soa_max2, d_total1, d_max1, d_total2, d_max2);
    });

    for (int i = 0; i < hash_params.len; ++i) {
        cudaFree(h_soa_total1.m[i]);
        cudaFree(h_soa_max1.m[i]);
        cudaFree(h_soa_total2.m[i]);
        cudaFree(h_soa_max2.m[i]);
    }
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_tile_boundaries);
    cudaFree(d_soa_total1);
    cudaFree(d_soa_total2);
    cudaFree(d_soa_max1);
    cudaFree(d_soa_max2);
    cudaFree(d_total1);
    cudaFree(d_total2);
    cudaFree(d_max1);
    cudaFree(d_max2);
}


NVBENCH_BENCH(copyBenchDist1);
NVBENCH_BENCH(coloring1Bench);
NVBENCH_BENCH(copyBenchDist2);
NVBENCH_BENCH(coloring2Bench);