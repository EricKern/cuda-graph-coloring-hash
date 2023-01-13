#pragma once
#include <cstdint>

static constexpr int THREADS = 512;

namespace apa22_coloring {

  static constexpr int max_bit_width{14};
  static constexpr int start_bit_width{3};

  struct Counters {
    using value_type = uint32_t;
    value_type m[max_bit_width]{};
  };

  struct Sum_Counters {
    __host__ __device__ __forceinline__ Counters
    operator()(const Counters& a, const Counters& b) const {
      Counters tmp;
      for (auto i = 0; i < max_bit_width; ++i) {
        tmp.m[i] = a.m[i] + b.m[i];
      }
      return tmp;
    }
  };

  struct Max_Counters {
    __host__ __device__ __forceinline__ Counters
    operator()(const Counters& a, const Counters& b) const {
      Counters tmp;
      for (auto i = 0; i < max_bit_width; ++i) {
        tmp.m[i] = max(a.m[i], b.m[i]);
      }
      return tmp;
    }
  };

} // end apa22_coloring