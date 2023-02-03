#pragma once
#include <thrust/reduce.h>        // reduce and reduce_by_key
#include <thrust/functional.h>    // thrust::maximum
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include "cpu_brev.hpp"
#include "hash.cuh"
#include "coloring_counters.cuh"

namespace apa22_coloring
{

template <typename T>
struct Mask {
  int mask_width;
	static_assert(std::is_unsigned_v<T>);   // assuming we only get hashes
	__forceinline__ __host__ T operator()(T a) const noexcept {
		T mask = (1u << mask_width) - 1u;
    return a & mask;
	}
};

template <typename IndexType>
void cpuDist2Thrust(const IndexType* row_ptr,  // global mem
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

    std::vector<int> group_counts(max_node_degree);

    for(IndexType counter_idx = 0; counter_idx < num_bit_widths; ++counter_idx){
      auto shift_val = start_bit_width + counter_idx;
      thrust::constant_iterator<IndexType> one_iter(1);
      // mask hashes -> keys for reduce_by_key
      auto masked_hashes = thrust::make_transform_iterator(
                                                workspace.begin(),
                                                Mask<hash_type>{shift_val});

      // new_end is tuple holding end iterators of output iterators
      auto new_end = thrust::reduce_by_key(thrust::host,
                                          masked_hashes,
                                          masked_hashes + j,
                                          one_iter,
                                          thrust::make_discard_iterator(),
                                          group_counts.begin());

      int result = thrust::reduce(thrust::host,
                                  group_counts.begin(),
                                  new_end.second,
                                  1,
                                  thrust::maximum<int>{});

      node_collisions.m[counter_idx] = result - 1;
    }

    Sum_Counters sum_binary_op = Sum_Counters{};
    Max_Counters max_binary_op = Max_Counters{};
    sum_collisions = sum_binary_op(sum_collisions, node_collisions);
    max_collisions = max_binary_op(max_collisions, node_collisions);
  }

  *result_total = sum_collisions;
  *result_max = max_collisions;
}

} // namespace apa22_coloring
