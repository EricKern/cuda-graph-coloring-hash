#pragma once
#include <coloringCounters.cuh>

template <typename IndexType>
void cpu_dist1(const IndexType* row_ptr,  // global mem
               const IndexType* col_ptr,  // global mem
               const IndexType numNodes,
               Counters* result_total,
               Counters* result_max){
  // Counters for the total number of collisions and the max nr of collisions
  // per node
  Counters sum_collisions;
  Counters max_collisions;

  for (IndexType i = 0; i < numNodes; ++i) {
    // Separate counter for the current node
    Counters node_collisions;
    const IndexType glob_row = i;
    const IndexType row_begin = row_ptr[i];
    const IndexType row_end = row_ptr[i + 1];

    auto const row_hash = hash(glob_row, static_k_param);

    for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
      const IndexType col = col_ptr[col_idx];

      if (col != glob_row) {  // skip connection to node itself
        auto const col_hash = hash(col, static_k_param);

        for (auto bit_w = 1; bit_w <= max_bitWidth; ++bit_w) {
          std::make_unsigned_t<IndexType> mask = (1u << bit_w) - 1u;
          if ((row_hash & mask) == (col_hash & mask)) {
            node_collisions.m[bit_w - 1] += 1;
          } else {
            // if hashes differ in lower bits they also differ when increasing
            // the bit_width
            break;
          }
        }
      }
    }

    Sum_Counters sum_binary_op = Sum_Counters{};
    Max_Counters max_binary_op = Max_Counters{};
    sum_collisions = sum_binary_op(sum_collisions, node_collisions);
    max_collisions = max_binary_op(max_collisions, node_collisions);
  }

  *result_total = sum_collisions;
  *result_max = max_collisions;
}