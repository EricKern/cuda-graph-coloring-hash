#pragma once

// https://stackoverflow.com/a/34464919/19107665
#include <cub/cub.cuh>

template <typename T>
__device__
void odd_even_merge_sort(T arr[], int length) {
  int t = ceilf(log2f((float)length));
  int p = 1 << (t - 1); // pow(2, t - 1)

  while (p > 0) {
    int q = 1 << (t - 1); // pow(2, t - 1)
    int r = 0;
    int d = p;

    while (d > 0) {
      for (int i = 0; i < length - d; ++i) {
        if ((i & p) == r) {
          auto& lhs = arr[i];
          auto& rhs = arr[i + d];
          if (lhs > rhs){
            cub::Swap(lhs, rhs);
          }
        }
      }

      d = q - p;
      q /= 2;
      r = p;
    }
    p /= 2;
  }
}

template <typename T>
__device__ void insertionSort(T arr[], int n) {
  int i, j;
  T key;
  for (i = 1; i < n; i++) {
    key = arr[i];
    j = i - 1;

    // Move elements of arr[0..i-1],
    // that are greater than key, to one
    // position ahead of their
    // current position
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j = j - 1;
    }
    arr[j + 1] = key;
  }
}