// 1. copy permuted matrix to device
//    (In the following: tile = partition)
//    tile_boundaries: array of indices giving the starting index of each partition.
//    Partitions are internally grouped. Starting with tile-edge nodes
//    followed by intra-tile nodes

//    intra_tile_sep: array of indices giving the starting index of the first
//    intra-tile node in each partition.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename IndexType>
__device__
void Partition2ShMem(IndexType* shMemRows,
                     IndexType* shMemCols,
                     IndexType* row_ptr,  // global mem
                     IndexType* col_ptr,  // global mem
                     IndexType* tile_boundaries){
  // determine which tile to load with blockId
  uint partNr = blockIdx.x;

  uint partSize = tile_boundaries[partNr+1] - tile_boundaries[partNr] + 1;
  // +1 to load one additional row_ptr value to determine size of last row

  uint part_offset = tile_boundaries[partNr];   // offset in row_ptr array
  // put row_ptr partition in shMem
  for (uint i = threadIdx.x; i < partSize; i += blockDim.x){
    shMemRows[i] = row_ptr[i + part_offset];
  }

  cg::this_thread_block().sync();

  uint num_cols = shMemRows[partSize-1] - shMemRows[0];
  uint col_offset = shMemRows[0];
  for (uint i = threadIdx.x; i < num_cols; i += blockDim.x){
    shMemCols[i] = col_ptr[i + col_offset];
  }
  
  cg::this_thread_block().sync();
  // If partition contains n nodes then we now have
  // n+1 elements of row_ptr in shMem and the col_ptr values for
  // n nodes
}

template <typename IndexType>
__global__
void coloring1Kernel(IndexType* row_ptr,  // global mem
                     IndexType* col_ptr,  // global mem
                     IndexType* tile_boundaries,
                     IndexType* intra_tile_sep,
                     uint m_rows,
                     uint tile_max_nodes,
                     uint tile_max_edges){
  //uint number_of_tiles = gridDim.x;

  extern __shared__ IndexType shMem[];
  IndexType* shMemRows = shMem;                      // tile_max_nodes elements
  IndexType* shMemCols = &shMemRows[tile_max_nodes]; // tile_max_edges elements
  
  Partition2ShMem(shMemRows, shMemCols, row_ptr, col_ptr, tile_boundaries);
}