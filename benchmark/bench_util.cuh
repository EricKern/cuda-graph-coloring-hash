#pragma once
#include <nvbench/nvbench.cuh>
#include "mat_loader.hpp"
#include "coloring_counters.cuh"
#include "hash.cuh"


void add_MatInfo(nvbench::state &state) {
  auto& mat = apa22_coloring::MatLoader::getInstance(nullptr);
  std::string path(mat.path);
  std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
  std::string::size_type const p(base_filename.find_last_of('.'));
  std::string file_without_extension = base_filename.substr(0, p);

  auto &summ_mame = state.add_summary("apa/matID/name");
  summ_mame.set_string("name", "Matrix Name");
  summ_mame.set_string("value", file_without_extension);

  state.add_element_count(mat.m_rows, "Rows");
  state.add_element_count(mat.row_ptr[mat.m_rows], "Non-zeroes");
}

void add_IOInfo(nvbench::state &state, int n_blocks){
  using namespace apa22_coloring;
  auto& mat = MatLoader::getInstance(nullptr);

  // matrix
  size_t in_elem = mat.m_rows + mat.row_ptr[mat.m_rows];
  state.add_global_memory_reads<int>(in_elem, "Mat Row+Col");

  // block reduction results
  state.add_global_memory_writes<Counters>(n_blocks * num_hashes, "R1 Counters");

  // last reduction
  state.add_global_memory_reads<Counters>(n_blocks * num_hashes);
  state.add_global_memory_writes<Counters>(num_hashes, "Results");
}