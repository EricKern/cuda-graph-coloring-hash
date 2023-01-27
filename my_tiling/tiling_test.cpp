#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <defines.hpp>
#include <cli_parser.hpp>

#include <memory>
#include <cstdio>

#include <chrono>

template <typename T>
void my_print(T* ptr, int len, const char* a) {
    std::printf(a);
    for (size_t i = 0; i < len; ++i) {
        std::printf(" %d", ptr[i]);
    }
    std::printf("\n");
}

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
    very_simple_tiling(rowPtr, m_rows, 6000, &tile_boundaries, &n_tiles, &max_node_degree);
    auto tile_stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff_load =
        load_stop - load_start;

    std::chrono::duration<double, std::milli> diff_tile =
        tile_stop - tile_start;

    int max_nodes, max_edges;
    int biggest_tile_nodes, biggest_tile_edges;
    get_MaxTileSize(n_tiles, tile_boundaries.get(), rowPtr, &biggest_tile_nodes,
                    &biggest_tile_edges, &max_nodes, &max_edges);


    std::printf("requ mem: %d\n", (biggest_tile_nodes+biggest_tile_edges+1)*sizeof(int));
    std::printf("biggest_tile_nodes: %d\n", biggest_tile_nodes);
    std::printf("biggest_tile_edges: %d\n", biggest_tile_edges);
    std::printf("max_nodes: %d\n", max_nodes);
    std::printf("max_edges: %d\n", max_edges);
    std::printf("num_tile %d\n", n_tiles);
    std::printf("m_rows %d\n", m_rows);
    std::printf("max_node_degree %d\n", max_node_degree);
    std::printf("load time: %f\n", diff_load.count());
    std::printf("tile time: %f\n", diff_tile.count());
}