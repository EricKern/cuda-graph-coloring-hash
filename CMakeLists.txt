cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ppcg VERSION 1.0 LANGUAGES CXX CUDA DESCRIPTION "tiling example")

add_subdirectory(tiling)
add_subdirectory(cpumultiply)

add_executable(example main-coloring.cu)
target_compile_options(example PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;-Xptxas;-v>")
target_compile_options(example PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v>")
target_include_directories(example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(example PRIVATE cuda_std_17)
target_link_libraries(example PRIVATE cusparse tiling cpumultiply)
