#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <coloring.cuh>
#include <hash.cuh>

/**
 * @brief Main entry point for all CPU versions
 * @param[in] m_rows			: number of rows of matrix
 * @param[in] number_of_tiles	: desired number of tiles each layer is
 * partitioned into
 * @param[in] row_ptr			: integer array that contains the start
 * of every row and the end of the last row plus one
 * @param[in] col_ptr			: integer array of column indices of the
 * nonzero elements of matrix
 * @param[out] slices_dptr		: pointer to integer array of m_rows
 * elements containing row numbers
 * @param[out] ndc_dptr			: pointer to integer array of tn*n_z+1
 * elements that contains the start of every tile and the end of the last tile
 * @param[out] offsets_dptr		: pointer to array with indices for each
 * tile marking line between nodes with and without redundant values
 */
void simple_tiling(const int m_rows, const int number_of_tiles,
                   const int* const row_ptr, const int* const col_ptr,
                   int** const slices_dptr, int** const ndc_dptr,
                   int** const offsets_dptr) {
  int indices_[] = {0, m_rows};
  int number_of_slices = 1;
  int* slices_ = new int[m_rows];
  int* layers_ = new int[m_rows];
  for (int i = 0; i < m_rows; i++) {
    slices_[i] = i;
    layers_[i] = 0;
  }
  *slices_dptr = slices_;
  tiling_partitioning(m_rows, number_of_tiles, number_of_slices, row_ptr,
                      col_ptr, layers_, indices_, slices_, ndc_dptr,
                      slices_dptr, nullptr, offsets_dptr, nullptr,
                      &number_of_slices, 0);
  delete[] layers_;
}

void get_MaxTileSize(const uint number_of_tiles, int* ndc_, int* row_ptr, uint* maxTileSize, uint* maxEdges) {
  uint tile_node_max = 0;
  uint tile_edge_max = 0;
  // go over tiles
  for (uint tile_nr = number_of_tiles; tile_nr > 0; --tile_nr) {
    // calculate tile size
    const uint tile_sz = ndc_[tile_nr] - ndc_[tile_nr - 1];
    if (tile_sz > tile_node_max)  // if tile size is bigger than the maximal one
      tile_node_max = tile_sz;    // overwrite the maximal value

    const uint tile_edges = row_ptr[ndc_[tile_nr]] - row_ptr[ndc_[tile_nr - 1]];
    if (tile_edges > tile_edge_max)
      tile_edge_max = tile_edges;
  }
  // Return:
  *maxTileSize = tile_node_max;
  *maxEdges = tile_edge_max;
}

int main() {
  const char* inputMat = "/home/eric/Documents/graph-coloring/CurlCurl_0.mtx";
  const uint number_of_tiles = 12;

  int* row_ptr;
  int* col_ptr;
  float* val_ptr;  // create pointers for matrix in csr format
  const int m_rows =
      cpumultiplySloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);

  int* ndc_;     // array with indices of each tile in all slices
  int* slices_;  // array with nodes grouped in slices
  int* offsets_;
  simple_tiling(m_rows, number_of_tiles, row_ptr, col_ptr, &slices_, &ndc_,
                &offsets_);
  cpumultiplySpermuteMatrix(number_of_tiles, 1, ndc_, slices_, row_ptr, col_ptr,
                            val_ptr, &row_ptr, &col_ptr, &val_ptr, true);

  uint max_nodes, max_edges;
  get_MaxTileSize(number_of_tiles, ndc_, row_ptr, &max_nodes, &max_edges);

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
  cudaMalloc((void**)&d_results, number_of_tiles * sizeof(Counters));

  // calc shMem
  size_t shMem_bytes = (max_nodes + max_edges) * sizeof(int);
  dim3 gridSize(number_of_tiles);
  dim3 blockSize(512);

  coloring1Kernel<<<gridSize, blockSize, shMem_bytes>>>(
      d_row_ptr, d_col_ptr, d_tile_boundaries, d_intra_tile_sep, m_rows,
      max_nodes, max_edges, d_results);
  cudaDeviceSynchronize();

  Counters result;
  cudaMemcpy(&result, d_results, 1 * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  return 0;
}