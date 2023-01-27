#pragma once
#include <cstdint>
#include <cub/cub.cuh>


namespace apa22_coloring {
  //=========================================
  // How many and which hash functions to use
  //=========================================
  static constexpr int static_k_param{2};

  template<int N>
  struct HashParams {
      constexpr HashParams() : val() {
          for (auto i = 1; i < N; ++i) {
              val[i] = i + 2; 
          }
          val[0] = static_k_param;
      }
      int val[N];
      static constexpr int len = N;
  };

  __device__
  static constexpr HashParams<5> hash_params{};

  //=========================================
  // How many and which bit widths to use
  //=========================================
  // I guess it makes only sense to measure consecutive bit widths.
  static constexpr int num_bit_widths{14};
  static constexpr int start_bit_width{3};

  struct Counters {
    int m[num_bit_widths]{};
  };

  // for each hash function we need an array of gridDim.x * num_bit_widths ints
  struct SOACounters {
    int* m[hash_params.len]{};
  };


  struct Sum_Counters {
    __host__ __device__ __forceinline__ Counters
    operator()(const Counters& a, const Counters& b) const {
      Counters tmp;
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
      for (auto i = 0; i < num_bit_widths; ++i) {
        tmp.m[i] = max(a.m[i], b.m[i]);
      }
      return tmp;
    }
  };

} // end apa22_coloring