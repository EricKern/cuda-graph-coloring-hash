cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "75")

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

enable_testing()
add_subdirectory(tests)
# google test end =======================================

find_package(nvbench)


add_subdirectory(tiling)
add_subdirectory(cpumultiply)
add_subdirectory(benchmark)

set(example example.out)
add_executable(${example} example.cu)
target_compile_options(${example} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-Xptxas;-v>")
target_compile_options(${example} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_compile_features(${example} PRIVATE cuda_std_17)
target_link_libraries(${example} PRIVATE tiling cpumultiply)
