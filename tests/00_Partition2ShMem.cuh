#pragma once
#include <cooperative_groups.h>

#include <coloringCounters.cuh>
#include <coloring.cuh>

template <typename IndexType>
__global__
void ShMemLoadTest(IndexType* row_ptr,  // global mem
                   IndexType* col_ptr,  // global mem
                   IndexType* tile_boundaries,
                   uint tile_max_nodes,
                   uint tile_max_edges,
                   bool* error){

  extern __shared__ IndexType shMem[];
  IndexType* shMemRows = shMem;                       // tile_max_nodes +1 elements
  IndexType* shMemCols = &shMemRows[tile_max_nodes+1]; // tile_max_edges elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);

  // Global mem access in tile_boundaries !!!!!
  uint partNr = blockIdx.x;
  IndexType rowsInShMem = tile_boundaries[partNr+1] - tile_boundaries[partNr] + 1;

  bool correct;
  // check row_ptr in ShMem
  for (uint i = threadIdx.x; i < rowsInShMem; i += blockDim.x){
    correct = shMemRows[i] == row_ptr[tile_boundaries[partNr] + i];
    if (!correct){
      // race condition doesn't matter. As long as it is set something is wrong
      *error = true;
    }
  }

  // check col_ptr in ShMem
  for (uint i = threadIdx.x; i < rowsInShMem-1; i += blockDim.x){
    auto row_start = shMemRows[i];
    auto row_end = shMemRows[i+1];

    for (auto idx = row_start; idx < row_end; ++idx){
      auto col_offset = shMemRows[0];
      correct = shMemCols[idx - col_offset] == col_ptr[idx];
      if (!correct){
        *error = true;
      }
    }
  }
  
}