/** 
 * \file tiling.hpp
 * @brief Header file of shared library containing tiling process functions
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/
#include <cstdint>

double // returns time measurement of tiling process (subdivide layers in grouped rows called tiles)
tiling_partitioning(
    const int row_count, // number of rows of matrix
    const int
        tile_count, // desired number of tiles each slice is partitioned into
    const int in_slice_count, // number of slices
    const int* const
        row2index, // array of row_count+1 elements containing first index of every matrix row in index2column array
    const int* const
        index2column, // array of row2index(row_count) elements containing column indices of non-zero entries in matrix
    const int* const
        node2slice, // integer array of row_count elements with corresponding slice number for each matrix row
    const int* const
        slice2index, // integer array of slice_count+1 elements that contains the start of every slice in in_node2node array
    const int* const
        in_node2node, // pointer to input array containing node numbers grouped in slices and subdivided within each slice into tiles
    int** const
        tile2index, // pointer to integer array of tile_count*slice_count+1 elements that contains the start of every tile (not allocated if NULL)
    int** const
        out_node2node, // pointer to integer array containing node numbers grouped in slices and subdivided within each slice into tiles (not allocated if NULL)
    int** const
        node2tile, // pointer to output integer array with corresponding tile number for each node (not allocated if NULL)
    int** const
        tile2offset, // pointer to output integer array with indices for each tile marking line between nodes with and without redundant values (not allocated if NULL)
    char** const
        stats_msg, // pointer to output char array of chars intended to hold printing information if printing flags are set in til_prop (not allocated if NULL)
    int* const
        out_slice_count, // pointer to number of slices (can be set to NULL, if no output required)
    const uint8_t
        til_prop); // hold information about settings for secondary and quaternary flags

double // returns time measurement of equal destribution of nodes among tiles in the only layer
tiling_equipartitioning(
    const int row_count, // number of rows of matrix
    const int
        tile_count, // desired number of tiles each slice is partitioned into
    int**
        tile2index); // pointer to integer array of tile_count*slice_count+1 elements that contains the start of every tile (not allocated if NULL)

double // returns time measurement of tiling process for lower triangular matrix (subdivide matrix in grouped rows called layers and layers in grouped rows called tiles)
tiling_triangular(
    const int row_count, // number of rows of matrix
    const int* const
        row2index, // array of row_count+1 elements containing first index of every matrix row in index2column array
    const int* const
        index2column, // array of row2index(row_count) elements containing column indices of non-zero entries in matrix
    int** const
        node2node, // pointer to array containing node numbers grouped in slices and subdivided within each slice into tiles (not allocated if NULL)
    int** const
        tile2index, // pointer to integer array of tile_count*slice_count+1 elements that contains the start of every tile (not allocated if NULL)
    int* const
        slice_count, // pointer to number of slices (can be set to NULL, if no output required)
    int* const
        tile_count); // desired number of tiles each slice is partitioned into (can be set to NULL, if no output required)


void simple_tiling(const int m_rows, const int number_of_tiles,
                   const int* const row_ptr,
                   const int* const col_ptr,
                   int** const slices_dptr,
                   int** const ndc_dptr,
                   int** const offsets_dptr);

void get_MaxTileSize(const int number_of_tiles, const int* const ndc_, const int* const row_ptr,
                     int* maxTileSize, int* maxEdges);