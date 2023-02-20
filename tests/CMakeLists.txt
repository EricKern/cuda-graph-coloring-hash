

add_executable(kernelTestsD1.out kernelTestsD1.cu)
target_link_libraries(kernelTestsD1.out gtest_main tiling cpumultiply)
target_include_directories(kernelTestsD1.out PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tests")
add_test(NAME kernelTestsD1 COMMAND kernelTestsD1.out)

add_executable(kernelTestsD2.out kernelTestsD2.cu)
target_link_libraries(kernelTestsD2.out gtest_main tiling cpumultiply)
target_include_directories(kernelTestsD2.out PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tests")
add_test(NAME kernelTestsD2 COMMAND kernelTestsD2.out)
