#pragma once
#include <cstdint>


namespace apa22_coloring {
  //=========================================
  // How many and which bit widths to use
  //=========================================
  // I guess it makes only sense to measure consecutive bit widths.
  static constexpr int num_bit_widths{8};
  static constexpr int start_bit_width{3};

  struct Counters {
    using value_type = int;
    value_type m[num_bit_widths]{};
  };

  struct Sum_Counters {
    __host__ __device__ __forceinline__ Counters
    operator()(const Counters& a, const Counters& b) const {
      Counters tmp;
      #pragma unroll num_bit_widths
      for (auto i = 0; i < num_bit_widths; ++i) {
        tmp.m[i] = a.m[i] + b.m[i];
      }
      return tmp;
    }
  };

  struct Max_Counters {
    __host__ __device__ __forceinline__ Counters
    operator()(const Counters& a, const Counters& b) const {
      Counters tmp;
      #pragma unroll num_bit_widths
      for (auto i = 0; i < num_bit_widths; ++i) {
        tmp.m[i] = max(a.m[i], b.m[i]);
      }
      return tmp;
    }
  };

} // end apa22_coloring