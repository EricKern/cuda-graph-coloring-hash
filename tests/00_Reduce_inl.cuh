#pragma once
#include <coloringCounters.cuh>

#include <cub/cub.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace red_test{
__device__ unsigned int red_test_retirementCount = 0;

__global__
void SingleReduce(Counters* in_counters,  // global mem
                  uint num_counters,
                  Counters* d_results){
  // // template BlockDim is bad
  
  typedef cub::BlockReduce<Counters, 512> BlockReduce;
  using TempStorageT = typename BlockReduce::TempStorage;

  __shared__ TempStorageT temp_storage;

  Counters thread_counter = Counters{};
  uint glob_tid = blockIdx.x * blockDim.x + threadIdx.x;
  // if(threadIdx.x == 0){
  //   for (size_t i = 0; i < max_bitWidth; i++)
  //   {
  //     d_results[blockIdx.x].m[i] = 0;
  //   }
    
  // }

  uint div = num_counters / blockDim.x;
  uint rem = num_counters % blockDim.x;
  uint block_valid;

  if(blockIdx.x < div){
    block_valid = blockDim.x;
  } else if (blockIdx.x == div){
    block_valid = rem;
  } else {
    block_valid = 0;
  }

  if(glob_tid < num_counters){
    thread_counter = in_counters[glob_tid];
  }


  Counters sum_accu = BlockReduce(temp_storage).Reduce(thread_counter,
                                                  Sum_Counters(), block_valid);
  cg::this_thread_block().sync();

  Counters max_accu = BlockReduce(temp_storage).Reduce(thread_counter,
                                                  Max_Counters(), block_valid);
  cg::this_thread_block().sync();

  if(threadIdx.x == 0){
    d_results[blockIdx.x] = sum_accu;
    d_results[blockIdx.x + gridDim.x] = max_accu;
  }

  __threadfence();

  __shared__ bool amLast;
  // Thread 0 takes a ticket
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&red_test_retirementCount, gridDim.x);
    // If the ticket ID is equal to the number of blocks, we are the last
    // block!
    amLast = (ticket == gridDim.x - 1);
  }

  cg::this_thread_block().sync();

    // if(threadIdx.x == 0 && blockIdx.x == 1){
    //   printf("glob_tid: %d\n", glob_tid);
    //   printf("num_counters: %d\n", num_counters);
    //   printf("block_valid: %d\n", block_valid);
    //   printf("0: %d, 1: %d, 2:%d", thread_counter.m[0], thread_counter.m[1], thread_counter.m[2]);
    // }
  // The last block sums the results of all other blocks
  if (amLast && (threadIdx.x < gridDim.x)) {
    thread_counter = d_results[threadIdx.x];

    Counters sum_accu = BlockReduce(temp_storage).Reduce(thread_counter,
                                                         Sum_Counters(),
                                                         gridDim.x);
    cg::this_thread_block().sync();

    thread_counter = d_results[threadIdx.x + gridDim.x];
    Counters max_accu = BlockReduce(temp_storage).Reduce(thread_counter,
                                                         Max_Counters(),
                                                         gridDim.x);
    cg::this_thread_block().sync();

    if (threadIdx.x == 0) {
      d_results[0] = sum_accu;
      d_results[1] = max_accu;

      // reset retirement count so that next run succeeds
      red_test_retirementCount = 0;
    }
  }
}
}