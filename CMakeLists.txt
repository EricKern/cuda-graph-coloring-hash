cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "61;75")

project(graph-coloring VERSION 1.0 LANGUAGES CXX CUDA
    DESCRIPTION "cuda implementation of distance1 and distance2 coloring")

# google test start =======================================
set(BUILD_GMOCK OFF) # currently we don't need gMock
include(FetchContent)
FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/e5e46b09a10cf0f575dbab2702b72f2460e31964.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# enable_testing()
# add_subdirectory(tests)
# google test end =======================================

find_package(nvbench)
# add_subdirectory(nvbench)


add_subdirectory(tiling)
add_subdirectory(cpumultiply)

set(Target1Name main-coloring.out)
add_executable(${Target1Name} main-coloring.cu)
target_compile_options(${Target1Name} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-Xptxas;-v>")
target_compile_options(${Target1Name} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
target_include_directories(${Target1Name} PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(${Target1Name} PRIVATE cuda_std_17)
target_link_libraries(${Target1Name} PRIVATE cusparse tiling cpumultiply)

set(BenchTarget d1_d2_SortNet_bench.out)
add_executable(${BenchTarget} benchmark/d1_d2_SortNet_bench.cu)
target_compile_options(${BenchTarget} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
target_include_directories(${BenchTarget} PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(${BenchTarget} PRIVATE tiling cpumultiply nvbench::main)
set_target_properties(${BenchTarget} PROPERTIES COMPILE_FEATURES cuda_std_17)

# set(BenchTarget numBlocks_bench.out)
# add_executable(${BenchTarget} benchmark/numBlocks_bench.cu)
# target_compile_options(${BenchTarget} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
# target_include_directories(${BenchTarget} PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(${BenchTarget} PRIVATE tiling cpumultiply nvbench::main)
# set_target_properties(${BenchTarget} PROPERTIES COMPILE_FEATURES cuda_std_17)

# set(partKernelsName partKernels.out)
# add_executable(${partKernelsName} benchmark/partKernels.cu)
# target_compile_options(${partKernelsName} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
# target_include_directories(${partKernelsName} PRIVATE "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(${partKernelsName} PRIVATE tiling cpumultiply nvbench::main)
# set_target_properties(${partKernelsName} PROPERTIES COMPILE_FEATURES cuda_std_17)


# set(tiling_test tiling_test.out)
# add_executable(${tiling_test} my_tiling/tiling_test.cpp)
# target_include_directories(${tiling_test} PRIVATE ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(${tiling_test} PRIVATE tiling cpumultiply)
# set_target_properties(${tiling_test} PROPERTIES COMPILE_FEATURES cuda_std_17)