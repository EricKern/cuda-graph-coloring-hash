cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(OpenMP COMPONENTS CXX REQUIRED)

add_library(cpumultiply STATIC cpumultiply.cpp)
target_compile_features(cpumultiply PRIVATE cxx_std_11)
target_include_directories(cpumultiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpumultiply PRIVATE OpenMP::OpenMP_CXX)
