


# Now simply link against gtest or gtest_main as needed. Eg
add_executable(cu_test1.out cu_test1.cu)
target_link_libraries(cu_test1.out gtest_main)
target_include_directories(cu_test1.out PRIVATE "${PROJECT_SOURCE_DIR}")
add_test(NAME cu_test COMMAND cu_test1.out)

add_executable(00_coloring_test.out 00_coloring_test.cu)
target_link_libraries(00_coloring_test.out gtest_main tiling cpumultiply)
target_include_directories(00_coloring_test.out PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tests")
add_test(NAME 00_coloring_test COMMAND 00_coloring_test.out)