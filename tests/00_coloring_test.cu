#include "gtest/gtest.h"


#include <cpumultiply.hpp>  //! header file for tiling
#include <tiling.hpp>       //! header file for tiling
#include <defines.hpp>

#include <00_Reduce_inl.cuh>
#include <numeric>
#include <00_Partition2ShMem.cuh>

#include <cpu_coloring.hpp>

namespace {
// To use a test fixture, derive a class from testing::Test.
class ColoringEnv : public testing::Test {
 protected:  // You should make the members protected s.t. they can be
             // accessed from sub-classes.
  // virtual void SetUp() will be called before each test is run.  You
  // should define it if you need to initialize the variables.
  // Otherwise, this can be skipped.
  void SetUp() override {
    const char* inputMat = def::Mat2;
    number_of_tiles = 24;

    m_rows = cpumultiplyDloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);

    simple_tiling(m_rows, number_of_tiles, row_ptr, col_ptr, &slices_, &ndc_,
                  &offsets_);
    cpumultiplyDpermuteMatrix(number_of_tiles, 1, ndc_, slices_, row_ptr, col_ptr,
                              val_ptr, &row_ptr, &col_ptr, &val_ptr, true);

    row_ptr_len = m_rows + 1;
    col_ptr_len = row_ptr[m_rows];
    tile_bound_len = number_of_tiles + 1;
    intra_tile_sep_len = number_of_tiles;

    cudaMalloc((void**)&d_row_ptr, row_ptr_len * sizeof(int));
    cudaMalloc((void**)&d_col_ptr, col_ptr_len * sizeof(int));
    cudaMalloc((void**)&d_tile_boundaries, tile_bound_len * sizeof(int));
    cudaMalloc((void**)&d_intra_tile_sep, intra_tile_sep_len * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr, row_ptr_len * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, col_ptr, col_ptr_len * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_boundaries, ndc_, tile_bound_len * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_intra_tile_sep, offsets_, intra_tile_sep_len * sizeof(int),
              cudaMemcpyHostToDevice);
  }

  // virtual void TearDown() will be called after each test is run.
  // You should define it if there is cleanup work to do.  Otherwise,
  // you don't have to provide it.
  //
  void TearDown() override {
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_tile_boundaries);
    cudaFree(d_intra_tile_sep);
    delete ndc_;
    delete slices_;
    delete offsets_;

    delete row_ptr;
    delete col_ptr;
    delete val_ptr;
  }

  // Declares the variables your tests want to use.
  uint number_of_tiles;
  int m_rows;

  int* row_ptr;
  int* col_ptr;
  double* val_ptr;  // create pointers for matrix in csr format

  int* ndc_;     // array with indices of each tile in all slices
  int* slices_;  // array with nodes grouped in slices
  int* offsets_;

  int* d_row_ptr;
  int* d_col_ptr;
  int* d_tile_boundaries;
  int* d_intra_tile_sep;
  size_t row_ptr_len;
  size_t col_ptr_len;
  size_t tile_bound_len;
  size_t intra_tile_sep_len;
};
// When you have a test fixture, you define a test using TEST_F
// instead of TEST.

TEST_F(ColoringEnv, FillShmem) {

  uint max_nodes, max_edges;
  get_MaxTileSize(number_of_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
    // calc shMem
  size_t shMem_bytes = (max_nodes+1 + max_edges) * sizeof(int);
  dim3 gridSize(number_of_tiles);
  dim3 blockSize(512);

  bool* errors;
  cudaMallocManaged((void**)&errors, sizeof(bool));
  *errors = false;

  ShMemLoadTest<<<gridSize, blockSize, shMem_bytes>>>(
      d_row_ptr, d_col_ptr, d_tile_boundaries,
      max_nodes, max_edges, errors);
  cudaDeviceSynchronize();

  EXPECT_FALSE(*errors);
}

// Tests the default c'tor.
TEST_F(ColoringEnv, CountersReduction) {
  using namespace red_test;
  // You can access data in the test fixture here.

  uint num_Blocks = 2;
  uint num_Counters = 50;
  std::vector<Counters> a(num_Counters);
  for (size_t i = 0; i < num_Counters; i++){
    for (size_t m_idx = 0; m_idx < max_bit_width; m_idx++)
    {
      a[i].m[m_idx] = i*max_bit_width + m_idx;
    }
  }
  Counters sum_c = std::reduce(a.begin(), a.end(), Counters{}, Sum_Counters());
  Counters max_c = std::reduce(a.begin(), a.end(), Counters{}, Max_Counters());

  Counters* d_counters_in;
  Counters* d_counters_out;
  cudaMalloc((void**)&d_counters_in, num_Counters * sizeof(Counters));
  cudaMalloc((void**)&d_counters_out, 2 * num_Blocks * sizeof(Counters));
  cudaMemcpy(d_counters_in, a.data(), num_Counters * sizeof(Counters),
            cudaMemcpyHostToDevice);

  dim3 gridSize(num_Blocks);
  dim3 blockSize(25);


  SingleReduce<<<gridSize, blockSize>>>(d_counters_in, num_Counters, d_counters_out);
  cudaDeviceSynchronize();

  Counters d_SumResult;
  Counters d_MaxResult;
  cudaMemcpy(&d_SumResult, d_counters_out, sizeof(Counters),
            cudaMemcpyDeviceToHost);
  cudaMemcpy(&d_MaxResult, d_counters_out + 1, sizeof(Counters),
            cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < max_bit_width; i++)
  {
    EXPECT_EQ(sum_c.m[i], d_SumResult.m[i]);
    EXPECT_EQ(max_c.m[i], d_MaxResult.m[i]);
  }
}


TEST_F(ColoringEnv, CPU_Comparison) {
  uint max_nodes, max_edges;
  get_MaxTileSize(number_of_tiles, ndc_, row_ptr, &max_nodes, &max_edges);
  
  // calc shMem
  size_t shMem_bytes = (max_nodes+1 + max_edges) * sizeof(int);
  dim3 gridSize(number_of_tiles);
  dim3 blockSize(512);

  Counters* d_results;
  cudaMalloc((void**)&d_results, number_of_tiles * 2 *sizeof(Counters));

  // run GPU version
  coloring1Kernel<<<gridSize, blockSize, shMem_bytes>>>(
      d_row_ptr, d_col_ptr, d_tile_boundaries, d_intra_tile_sep, m_rows,
      max_nodes, max_edges, d_results);
  cudaDeviceSynchronize();

  Counters gpu_total;
  cudaMemcpy(&gpu_total, d_results, 1 * sizeof(Counters),
            cudaMemcpyDeviceToHost);
  Counters gpu_max;
  cudaMemcpy(&gpu_max, d_results + 1, 1 * sizeof(Counters),
            cudaMemcpyDeviceToHost);


  Counters cpu_max, cpu_total;
  cpu_dist1(row_ptr, col_ptr, m_rows, &cpu_total, &cpu_max);

  for (int i = 0; i < max_bit_width; ++i){
    EXPECT_EQ(cpu_total.m[i], gpu_total.m[i]);
    EXPECT_EQ(cpu_max.m[i], gpu_max.m[i]);
  }
  

  cudaFree(d_results);
}

}  // namespace