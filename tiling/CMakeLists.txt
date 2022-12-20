# Cmake file for the compilation of tiling shared library
# Created by Robert Gutmann as a part of student research project
# In case of problems, contact me: rgutmann@stud.uni-heidelberg.de.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Disable in-source builds to avoid cluttering the source directory.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#set(CMAKE_VERBOSE_MAKEFILE ON)

project(tiling VERSION 1.0 LANGUAGES CXX DESCRIPTION "tiling shared library")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

list(APPEND general_flags -O3 -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -fomit-frame-pointer -mavx2 -mfma)
list(APPEND intel_flags -mavx512f -mavx512cd -mavx512er -mavx512pf)
list(APPEND amd_flags -march=znver1 -mtune=znver1 -msse4a)

set(metis "/opt/asc/pub/epochs/2020/spack/opt/spack/linux-centos7-x86_64/gcc-10.2.0/metis-5.1.0-7w765z55delyfdspz2gadwvl5ajdd3f4")
include_directories(${metis}/include)

add_library(tilingNTL SHARED tiling.cpp)
set_target_properties(tilingNTL PROPERTIES PUBLIC_HEADER tiling.hpp)
target_compile_options(tilingNTL PRIVATE ${general_flags} ${intel_flags})
target_link_libraries(tilingNTL PRIVATE -L${metis}/lib -lmetis)

add_library(tilingAMD SHARED tiling.cpp)
set_target_properties(tilingAMD PROPERTIES PUBLIC_HEADER tiling.hpp)
target_compile_options(tilingAMD PRIVATE ${general_flags} ${amd_flags})
target_link_libraries(tilingAMD PRIVATE -L${metis}/lib -lmetis)
