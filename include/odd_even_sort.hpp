#include <vector>

template <typename IndexT>
__device__
void odd_even_merge_sort(IndexT arr[], int size) {
  for (int i = 1; i < size; i *= 2) {
    for (int j = 0; j < size - i; j += 2 * i) {
      int left = j;
      int mid = j + i - 1;
      int right = min(j + 2 * i - 1, size - 1);

      int left_pointer = left;
      int right_pointer = mid + 1;

      while (left_pointer <= mid && right_pointer <= right) {
        if (arr[left_pointer] < arr[right_pointer]) {
          left_pointer++;
        } else {
          IndexT temp = arr[right_pointer];
          for (int k = right_pointer - 1; k >= left_pointer; k--) {
            arr[k + 1] = arr[k];
          }
          arr[left_pointer] = temp;
          left_pointer++;
          mid++;
          right_pointer++;
        }
      }
    }
  }
}
