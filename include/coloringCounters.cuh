#pragma once
#include <cstdint>

static constexpr int max_bitWidth{8};

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