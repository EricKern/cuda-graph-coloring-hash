#pragma once
#include <cstdint>

static constexpr int max_bitWidth{14};

struct Counters {
	using value_type = uint32_t;
	value_type m[max_bitWidth]{};
};

struct Sum_Counters {
  __host__ __device__ __forceinline__ Counters
  operator()(const Counters& a, const Counters& b) const {
    Counters tmp;
    for (auto i = 0; i < max_bitWidth; ++i) {
      tmp.m[i] = a.m[i] + b.m[i];
    }
    return tmp;
  }
};

struct Max_Counters {
  __host__ __device__ __forceinline__ Counters
  operator()(const Counters& a, const Counters& b) const {
    Counters tmp;
    for (auto i = 0; i < max_bitWidth; ++i) {
      tmp.m[i] = max(a.m[i], b.m[i]);
    }
    return tmp;
  }
};

template <typename T>
struct brev_cmp {
  static_assert(std::is_integral_v<T>);
	__forceinline__ __host__ __device__ bool operator()(T a,
	                                                    T b) const noexcept {
		using UT = std::make_unsigned_t<T>;
		return __brev(static_cast<UT>(a)) < __brev(static_cast<UT>(b));
	}
};