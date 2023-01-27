#include <cpumultiply.hpp>
#include <tiling.hpp>
#include <cub/cub.cuh>

#include "defines.hpp"

static constexpr const char* Mat = def::Mat3_Cluster;
static constexpr int STEPS = 300;
static constexpr int SHMEMLIMIT = 19 * 1024;

struct Dist{
    int* rowPtr;
    int* colPtr;
    int* ndc;
    int nTiles;
    int maxNodes;
    int maxEdges;
    int shMemSize;
};

int internal_bytes_used2(int max_nodes, int max_edges, bool dist2) {
  if (dist2) {
    return (max_nodes + 1 + max_edges) * sizeof(int) * 2;
  } else {
    return (max_nodes + 1 + max_edges) * sizeof(int);
  }
}


template <int distance = 0, int D1Tiles = 0, int D2Tiles = 0>
class Init {
    public:
        static Init* Instance() {
            static Init Instance;

            return &Instance;
        }

        Init(const Init&) = delete;
        Init& operator=(const Init&) = delete;

        Dist getDist1() const {
            return m_dist1;
        }

        Dist getDist2() const {
            return m_dist2;
        }

    private:
        Init() {
            bool tilesKnown;
            if constexpr (distance != 2){
                m_rows = cpumultiplyDloadMTX(Mat, &m_dist1.rowPtr, &m_dist1.colPtr, &m_valPtr);
                tilesKnown = true;
                int startTilesD1 = D1Tiles;
                if constexpr (D1Tiles == 0){
                    tilesKnown = false;
                    const int cols = m_dist1.rowPtr[m_rows];
                    int elem_p_block = (SHMEMLIMIT - sizeof(TempStorageT)) / (sizeof(int)) - 1;

                    auto elements = m_rows + cols;
                    startTilesD1 = std::ceil(elements / elem_p_block);
                }
                partition(m_dist1, false, startTilesD1, tilesKnown);
            }
            if constexpr (distance != 1){
                m_rows = cpumultiplyDloadMTX(Mat, &m_dist2.rowPtr, &m_dist2.colPtr, &m_valPtr);
                tilesKnown = true;
                int startTilesD2 = D2Tiles;
                if constexpr (D2Tiles == 0){
                    tilesKnown = false;
                    const int cols = m_dist1.rowPtr[m_rows];
                    int elem_p_block = (SHMEMLIMIT - sizeof(TempStorageT)) / (2 * sizeof(int)) - 1;

                    auto elements = m_rows + cols;
                    startTilesD2 = std::ceil(elements / elem_p_block);
                }
                partition(m_dist2, true, startTilesD2, tilesKnown);
            }
        }
        ~Init() {
            delete[] m_valPtr;
        }

        void partition(Dist &distStruct, bool dist2, int startTiles, bool tilesKnown) {
            Dist tmpDist = distStruct;
            int* slices;
            int* offsets;
            simple_tiling(m_rows, startTiles, tmpDist.rowPtr, tmpDist.colPtr, &slices, &tmpDist.ndc, &offsets);
            if (!tilesKnown) {
                int shMemSize;
                get_MaxTileSize(tmpDist.nTiles, tmpDist.ndc, tmpDist.rowPtr, &tmpDist.maxNodes, &tmpDist.maxEdges);
                shMemSize = internal_bytes_used2(tmpDist.maxNodes, tmpDist.maxEdges, dist2)
                                + sizeof(TempStorageT);
                
                while (shMemSize > SHMEMLIMIT){
                    delete[] slices;
	                delete[] offsets;
	                startTiles += STEPS;
	                simple_tiling(m_rows, startTiles, tmpDist.rowPtr, tmpDist.colPtr, &slices, &tmpDist.ndc, &offsets);
	                get_MaxTileSize(startTiles, tmpDist.ndc, tmpDist.rowPtr, &tmpDist.maxNodes, &tmpDist.maxEdges);
                    shMemSize = internal_bytes_used2(tmpDist.maxNodes, tmpDist.maxEdges, dist2)
                                            + sizeof(TempStorageT);
                }
            }
            tmpDist.nTiles = startTiles;
            cpumultiplyDpermuteMatrix(tmpDist.nTiles, 1, tmpDist.ndc, slices, tmpDist.rowPtr, tmpDist.colPtr,
                                     m_valPtr, &tmpDist.rowPtr, &tmpDist.colPtr, &m_valPtr, true);
            get_MaxTileSize(tmpDist.nTiles, tmpDist.ndc, tmpDist.rowPtr, &tmpDist.maxNodes, &tmpDist.maxEdges);
            tmpDist.shMemSize = internal_bytes_used2(tmpDist.maxNodes, tmpDist.maxEdges, dist2)
                                + sizeof(TempStorageT);
            
            distStruct = tmpDist;
            delete[] slices;
            delete[] offsets;
        }

        int m_rows;
        double* m_valPtr;
        Dist m_dist1;
        Dist m_dist2;

        using BlockReduceT = cub::BlockReduce<int, THREADS, RED_ALGO, 1, 1, 750>;
        using TempStorageT = typename BlockReduceT::TempStorage;
};