cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(OpenMP COMPONENTS CXX REQUIRED)

find_library(METIS_LIB NAMES metis REQUIRED)
find_path(METIS_INC NAMES metis.h REQUIRED)

add_library(tiling STATIC tiling.cpp)
target_compile_features(tiling PRIVATE cxx_std_11)
target_include_directories(tiling PRIVATE ${METIS_INC} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tiling PRIVATE ${METIS_LIB} OpenMP::OpenMP_CXX)
