// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// A sample program demonstrating using Google C++ testing framework.

// This sample shows how to write a simple unit test for a function,
// using Google C++ testing framework.
//
// Writing a unit test using Google C++ testing framework is easy as 1-2-3:

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.

#include "add.cuh"

#include <limits.h>

#include "gtest/gtest.h"

namespace {

  TEST(CudaCopy, add) {
    float *a;
    float *b;
    float *z;
    int len = 20;
    cudaMallocManaged((void**)&a, len*sizeof(float));
    cudaMallocManaged((void**)&b, len*sizeof(float));
    cudaMallocManaged((void**)&z, len*sizeof(float));

    for (size_t i = 0; i < len; ++i){
        a[i] = 3;
        b[i] = 7;
        z[i] = 0;
    }
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(a, len*sizeof(float), device);
    cudaMemPrefetchAsync(b, len*sizeof(float), device);
    cudaMemPrefetchAsync(z, len*sizeof(float), device);
    add_kernel<<<10, 128>>>(a, b, z, len);
    cudaDeviceSynchronize();
    cudaMemPrefetchAsync(z, len*sizeof(float), cudaCpuDeviceId);

    std::for_each(z, z+len, [](float elem){EXPECT_EQ(elem, 10);});

  }

}  // namespace
