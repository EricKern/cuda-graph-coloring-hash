#pragma once
#include <cstdint>


namespace apa22_coloring {
//=========================================
// How many and which hash functions to use
//=========================================
static constexpr int num_hashes{16};
static constexpr int start_hash{3};

template <typename IndexType>
__forceinline__ __host__ __device__ std::uint32_t hash(IndexType val,
                                              int k_param) noexcept {
  auto const uval = static_cast<std::make_unsigned_t<IndexType>>(val);
  auto const divisor = (1u << k_param) - 1u;
  return (uval / divisor) << k_param | (uval % divisor);
}

template <typename T>
struct brev_cmp {
  static_assert(std::is_unsigned_v<T>);
  __forceinline__ __device__ bool operator()(T a, T b) const noexcept {
    if constexpr (sizeof(T) > 4) {
      return __brevll(a) < __brevll(b);
    } else {
      return __brev(a) < __brev(b);
    }
  }
};

} // end apa22_coloring