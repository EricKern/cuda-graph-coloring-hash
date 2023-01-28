#pragma once
#include <coloringCounters.cuh>
#include <cpu_brev.hpp>
#include <hash.cuh>
#include <algorithm>  // for std::sort

namespace apa22_coloring {


template <typename IndexType>
void cpu_dist1(const IndexType* row_ptr,  // global mem
               const IndexType* col_ptr,  // global mem
               const IndexType numNodes,
               const int k_param,
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

    auto const row_hash = hash(glob_row, k_param);

    for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
      const IndexType col = col_ptr[col_idx];

      if (col != glob_row) {  // skip connection to node itself
        auto const col_hash = hash(col, k_param);

        for (auto counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx) {
          auto shift_val = start_bit_width + counter_idx;
          
          std::make_unsigned_t<IndexType> mask = (1u << shift_val) - 1u;
          if ((row_hash & mask) == (col_hash & mask)) {
            node_collisions.m[counter_idx] += 1;
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

template <typename IndexType>
void cpuDist2(const IndexType* row_ptr,  // global mem
               const IndexType* col_ptr,  // global mem
               const IndexType numNodes,
               const IndexType max_node_degree,
               const int k_param,
               Counters* result_total,
               Counters* result_max){
  // Counters for the total number of collisions and the max nr of collisions
  // per node
  Counters sum_collisions;
  Counters max_collisions;
  using hash_type = std::uint32_t;

  // We have to store the hash of all neighbors of a node and the node itself.
  // Node has always edge to itself we don't need this hash but can use
  // space for own hash
  std::vector<hash_type> workspace(max_node_degree);

  for (IndexType i = 0; i < numNodes; ++i) {
    // Separate counter for the current node
    Counters node_collisions;
    const IndexType glob_row = i;
    const IndexType row_begin = row_ptr[i];
    const IndexType row_end = row_ptr[i + 1];

    workspace[0] = hash(glob_row, k_param);

    IndexType j = 1;
    for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
      const IndexType col = col_ptr[col_idx];

      if (col != glob_row) {  // skip connection to node itself
        workspace[j] = hash(col, k_param);
        ++j;
      }
    }

    std::sort(workspace.begin(), workspace.begin() + j, brev_classic_cmp<hash_type>{});

    // We define node collisions at dist 2 as maximum number of equal patterns
    // in a group - 1
    // e.g. for bitWidth 2 we get the list {00, 01, 01, 01, 10, 11}
    // In this case we have 3x "01" as biggest group. So collisions would be 2.
    // We subtract 1 because if each group has 1 element the coloring would be
    // good so there are no collisions

    // Now we actually do reduce_by_key of list with all ones. With bitmask applied to
    // sorted list as keys. Then we find max in reduced array.

    for(IndexType counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx){
      auto shift_val = start_bit_width + counter_idx;

      std::make_unsigned_t<IndexType> mask = (1u << shift_val) - 1;
      int max_so_far = 1;
      int current = 1;

      auto group_start_hash = workspace[0];
      for (auto edge_idx = 1; edge_idx < j; ++edge_idx) {
        auto next_hash = workspace[edge_idx];

        if((group_start_hash & mask) == (next_hash & mask)){
          current += 1;
        } else {
          group_start_hash = next_hash;

          max_so_far = (current > max_so_far) ? current : max_so_far;
          current = 1;
        }
      }
      max_so_far = (current > max_so_far) ? current : max_so_far;
      // Put max_so_far - 1 in counter for current bit width
      node_collisions.m[counter_idx] = max_so_far - 1;
    }

    Sum_Counters sum_binary_op = Sum_Counters{};
    Max_Counters max_binary_op = Max_Counters{};
    sum_collisions = sum_binary_op(sum_collisions, node_collisions);
    max_collisions = max_binary_op(max_collisions, node_collisions);
  }

  *result_total = sum_collisions;
  *result_max = max_collisions;
}

} // end apa22_coloring