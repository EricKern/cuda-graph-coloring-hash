#pragma once

#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <coloringCounters.cuh>

#include <cmath>


namespace apa22_coloring {

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
                  int& shMem_size_bytes) {
  static constexpr auto reduction_shmem = 100 + sizeof(typename cub::BlockReduce<Counters, THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, 750>::TempStorage);

//   static constexpr int reduction_shmem = sizeof(BlockReduce::TempStorage);
  static constexpr int max_shmem = 19 * 1024 - reduction_shmem;
  static constexpr int shmem_lim = distance2 ? max_shmem / 2 : max_shmem;
  std::cout << "SHmem cub" << reduction_shmem << std::endl;
  // if (distance2) constexpr {
  //   shmem_lim = max_shmem / 2;
  // } else {
  //   shmem_lim = max_shmem;
  // }

  const int m_rows =
    cpumultiplyDloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);
  const int m_cols = row_ptr[m_rows];
  nr_rows = m_rows;

  // ((m_rows/num_tiles) + 1 + m_cols/num_tiles) * sizeof(int) <= max_shmem;

  auto elements = static_cast<int64_t>(m_rows) + static_cast<int64_t>(m_cols);
  
  int num_tiles = std::ceil(elements / (shmem_lim/sizeof(int) - 1));
  num_tiles+=100;

  // int* ndc_;     // array with indices of each tile in all slices
  int* slices_;  // array with nodes grouped in slices
  int* offsets_;
  simple_tiling(m_rows, num_tiles, row_ptr, col_ptr, &slices_, &ndc_,
                &offsets_);

  uint max_nodes, max_edges;
  get_MaxTileSize(num_tiles, ndc_, row_ptr, &max_nodes, &max_edges);

  size_t shMem_bytes = (max_nodes+1 + max_edges) * sizeof(int);

  std::cout << "Kernel setup pre while"  << std::endl;
  std::cout << "pre while shMem: " << shMem_bytes << std::endl;
  while (shMem_bytes > shmem_lim) {
    delete[] slices_;
    delete[] offsets_;
    num_tiles += 100;
    simple_tiling(m_rows, num_tiles, row_ptr, col_ptr, &slices_, &ndc_,
              &offsets_);
    get_MaxTileSize(num_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
    shMem_bytes = (max_nodes+1 + max_edges) * sizeof(int);
    std::cout << "while shMem: " << shMem_bytes << std::endl;
    std::cout << "while num_tile: " << num_tiles << std::endl;
  }
  n_tiles = num_tiles;
  shMem_size_bytes = shMem_bytes;

  std::cout << "Pre permute" << std::endl;
  cpumultiplyDpermuteMatrix(num_tiles, 1, ndc_, slices_, row_ptr, col_ptr,
                            val_ptr, &row_ptr, &col_ptr, &val_ptr, true);
  std::cout << "Post permute" << std::endl;
  delete[] slices_;
	delete[] offsets_;
}

} // namespace apa22_coloring {