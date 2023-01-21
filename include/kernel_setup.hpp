#pragma once

#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <coloringCounters.cuh>

#include <cmath>
#include <cub/cub.cuh>


namespace apa22_coloring {

template<bool dist2 = false>
int internal_bytes_used(int max_nodes, int max_edges) {
  if constexpr (dist2) {
    return (max_nodes + 1 + max_edges) * sizeof(int) * 2;
  } else {
    return (max_nodes + 1 + max_edges) * sizeof(int);
  }
}

/// @brief Calculates how many tiles are necessary to fit in shMem
/// @tparam distance2 
/// @param [in] inputMat 
/// @param [out] row_ptr 
/// @param [out] col_ptr 
/// @param [out] val_ptr 
/// @param [out] ndc_ 
/// @param [out] n_tiles 
template <bool distance2 = false>
void kernel_setup(const char* inputMat,
                  int* &row_ptr,
                  int* &col_ptr,
                  double* &val_ptr,
                  int* &ndc_,
                  int& nr_rows,
                  int& n_tiles,
                  int& shMem_size_bytes,
                  int steps) {

  // Specialize BlockReduce type for our thread block
  using BlockReduceT = cub::BlockReduce<int,
	                                      THREADS,
	                                      RED_ALGO,
	                                      1,
	                                      1,
	                                      750>;
  using TempStorageT = typename BlockReduceT::TempStorage;

  static constexpr int desired_shmem_lim = 19 * 1024;

  std::cout << "Desired shmem usage per block: " << desired_shmem_lim << std::endl;

  const int m_rows =
	  cpumultiplyDloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);
  const int m_cols = row_ptr[m_rows];
  nr_rows = m_rows;

  // if dist1
  // (m_rows/num_tiles + 1 + m_cols/num_tiles) * sizeof(int) + sizeof(TempStorageT) <= desired_shmem_lim;
  // if dist2
  // 2 * (m_rows/num_tiles + 1 + m_cols/num_tiles) * sizeof(int) + sizeof(TempStorageT) <= desired_shmem_lim;

  int elem_p_block = 0;

  if constexpr (distance2) {
    elem_p_block = (desired_shmem_lim - sizeof(TempStorageT)) / (2 * sizeof(int)) - 1;
  } else {
    elem_p_block = (desired_shmem_lim - sizeof(TempStorageT)) / (sizeof(int)) - 1;
  }

  auto elements = static_cast<int64_t>(m_rows) + static_cast<int64_t>(m_cols);

  int num_tiles = std::ceil(elements / elem_p_block);
  num_tiles += steps;
  std::cout << "elem_p_block " << elem_p_block << std::endl;
  std::cout << "num_tiles " << num_tiles << std::endl;
  std::cout << "SizeOfTempstorage: " << sizeof(TempStorageT) << std::endl;
  // num_tiles = 12240;

  // int* ndc_;     // array with indices of each tile in all slices
  int* slices_; // array with nodes grouped in slices
  int* offsets_;
  simple_tiling(
	  m_rows, num_tiles, row_ptr, col_ptr, &slices_, &ndc_, &offsets_);

  int max_nodes, max_edges;
  get_MaxTileSize(num_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
  int total_bytes_shmem = internal_bytes_used<distance2>(max_nodes, max_edges)
                        + sizeof(TempStorageT);

  std::cout << "Kernel setup pre while" << std::endl;
  while (total_bytes_shmem > desired_shmem_lim) {
	  delete[] slices_;
	  delete[] offsets_;
	  num_tiles += steps;
	  simple_tiling(
		  m_rows, num_tiles, row_ptr, col_ptr, &slices_, &ndc_, &offsets_);
	  get_MaxTileSize(num_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
	  std::cout << "while num_tile: " << num_tiles << std::endl;
    total_bytes_shmem = internal_bytes_used<distance2>(max_nodes, max_edges)
                        + sizeof(TempStorageT);
  }
  std::cout << "Left while" << std::endl;
  std::cout << "Pre permute" << std::endl;
  cpumultiplyDpermuteMatrix(num_tiles, 1, ndc_, slices_, row_ptr, col_ptr,
                            val_ptr, &row_ptr, &col_ptr, &val_ptr, true);
  std::cout << "Post permute" << std::endl;

  // return values
  get_MaxTileSize(num_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
  n_tiles = num_tiles;
  shMem_size_bytes = internal_bytes_used<distance2>(max_nodes, max_edges);
  std::cout << "shMem_size_bytes: " << shMem_size_bytes << std::endl;
  std::cout << "num_tiles: " << num_tiles << std::endl;
  delete[] slices_;
	delete[] offsets_;
}

} // namespace apa22_coloring {