#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <defines.hpp>
#include <cli_parser.hpp>

#include <memory>
#include <cstdio>

#include <chrono>


// run as "./tiling_test.out -m 3". Number decides which matrix to use.
// Matrices are defined in include/defines.hpp
int main(int argc, char const *argv[]) {
    int mat_nr = 2;          //Default value
    chCommandLineGet<int>(&mat_nr, "m", argc, argv);
    auto Mat = def::choseMat(mat_nr);
    int* rowPtr;
    int* colPtr;
    double* valPtr;

    // time for loading
    auto load_start = std::chrono::high_resolution_clock::now();
    int m_rows = cpumultiplyDloadMTX(Mat, &rowPtr, &colPtr, &valPtr);
    auto load_stop = std::chrono::high_resolution_clock::now();

    std::unique_ptr<int[]> tile_boundaries;
    int n_tiles;
    int max_node_degree;

    // time for tiling
    auto tile_start = std::chrono::high_resolution_clock::now();
    very_simple_tiling(rowPtr, m_rows, 48000, &tile_boundaries, &n_tiles, &max_node_degree);
    auto tile_stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff_load =
        load_stop - load_start;

    std::chrono::duration<double, std::milli> diff_tile =
        tile_stop - tile_start;

    std::printf("num_tile %d\n", n_tiles);
    std::printf("m_rows %d\n", m_rows);
    std::printf("max_node_degree %d\n", max_node_degree);
    std::printf("load time: %f\n", diff_load.count());
    std::printf("tile time: %f\n", diff_tile.count());
}