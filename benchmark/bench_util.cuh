#pragma once
#include <nvbench/nvbench.cuh>
#include "mat_loader.hpp"


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