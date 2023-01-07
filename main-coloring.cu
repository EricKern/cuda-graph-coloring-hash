#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <coloring.cuh>

#include <stdio.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <asc.cuh>

#include <defines.hpp>

template <bool max = false>
void printResult(const Counters &res){
  if (max == false){
    printf("Total Collisions\n");
    for (uint i = 0; i < max_bitWidth; ++i) {
      printf("Mask: %d, Collisions: %d\n", i+1, res.m[i]);
    }
  } else {
    printf("Max Collisions per Node\n");
    for (uint i = 0; i < max_bitWidth; ++i) {
      printf("Mask: %d, Collisions: %d\n", i+1, res.m[i]);
    }
  }

}

int main() {
  const char* inputMat = def::Mat0;
  const uint number_of_tiles = 2;

  int* row_ptr;
  int* col_ptr;
  double* val_ptr;  // create pointers for matrix in csr format
  const int m_rows =
      cpumultiplyDloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);

  int* ndc_;     // array with indices of each tile in all slices
  int* slices_;  // array with nodes grouped in slices
  int* offsets_;
  simple_tiling(m_rows, number_of_tiles, row_ptr, col_ptr, &slices_, &ndc_,
                &offsets_);
  cpumultiplyDpermuteMatrix(number_of_tiles, 1, ndc_, slices_, row_ptr, col_ptr,
                            val_ptr, &row_ptr, &col_ptr, &val_ptr, true);

  int* d_row_ptr;
  int* d_col_ptr;
  int* d_tile_boundaries;
  int* d_intra_tile_sep;
  size_t row_ptr_len = m_rows + 1;
  size_t col_ptr_len = row_ptr[m_rows];
  size_t tile_bound_len = number_of_tiles + 1;
  size_t intra_tile_sep_len = number_of_tiles;
  cudaMalloc((void**)&d_row_ptr, row_ptr_len * sizeof(int));
  cudaMalloc((void**)&d_col_ptr, col_ptr_len * sizeof(int));
  cudaMalloc((void**)&d_tile_boundaries, tile_bound_len * sizeof(int));
  cudaMalloc((void**)&d_intra_tile_sep, intra_tile_sep_len * sizeof(int));

  cudaMemcpy(d_row_ptr, row_ptr, row_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ptr, col_ptr, col_ptr_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_tile_boundaries, ndc_, tile_bound_len * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_intra_tile_sep, offsets_, intra_tile_sep_len * sizeof(int),
             cudaMemcpyHostToDevice);

  Counters* d_results;
  cudaMalloc((void**)&d_results, number_of_tiles * 2 *sizeof(Counters));

  uint max_nodes, max_edges;
  get_MaxTileSize(number_of_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
  
  // calc shMem
  size_t shMem_bytes = (max_nodes + max_edges) * sizeof(int);
  dim3 gridSize(number_of_tiles);
  dim3 blockSize(512);

  coloring1Kernel<<<gridSize, blockSize, shMem_bytes>>>(
      d_row_ptr, d_col_ptr, d_tile_boundaries, d_intra_tile_sep, m_rows,
      max_nodes, max_edges, d_results);
  cudaDeviceSynchronize();

  Counters total;
  cudaMemcpy(&total, d_results, 1 * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  Counters max;
  cudaMemcpy(&max, d_results + 1, 1 * sizeof(Counters),
            cudaMemcpyDeviceToHost);

  printResult(total);
  printResult<true>(max);

  cudaFree(d_row_ptr);
  cudaFree(d_col_ptr);
  cudaFree(d_tile_boundaries);
  cudaFree(d_intra_tile_sep);

  thrust::host_vector<int> row(row_ptr, row_ptr + m_rows + 1);
  thrust::host_vector<int> col(col_ptr, col_ptr + row_ptr[m_rows]);
  thrust::host_vector<double> nnz(val_ptr, val_ptr + row_ptr[m_rows]);
  
  thrust::device_vector<int> d_row = row;
  thrust::device_vector<int> d_col = col;
  thrust::device_vector<double> d_nnz = nnz;

  namespace asc18 = asc_hash_graph_coloring;
  asc18::cusparse_distance1(d_nnz, d_row, d_col, 1);

	delete[] row_ptr;
	delete[] col_ptr;
	delete[] val_ptr;
	delete[] ndc_;
	delete[] slices_;
	delete[] offsets_;
  return 0;
}