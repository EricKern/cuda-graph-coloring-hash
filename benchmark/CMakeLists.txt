


set(cpy_bench cpy_bench.out)
add_executable(${cpy_bench} benchmark.cu)
target_compile_options(${cpy_bench} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
target_include_directories(${cpy_bench} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/benchmark)
target_link_libraries(${cpy_bench} PRIVATE tiling cpumultiply nvbench::nvbench cusparse)
set_target_properties(${cpy_bench} PROPERTIES COMPILE_FEATURES cuda_std_17)

set(sortBench d1_d2_SortNet_bench.out)
add_executable(${sortBench} d1_d2_SortNet_bench.cu)
target_compile_options(${sortBench} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
target_include_directories(${sortBench} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/benchmark)
target_link_libraries(${sortBench} PRIVATE nvbench::nvbench tiling cpumultiply)
set_target_properties(${sortBench} PROPERTIES COMPILE_FEATURES cuda_std_17)

# set(numBlocks_bench numBlocks_bench.out)
# add_executable(${numBlocks_bench} num_blocks_bench.cu)
# target_compile_options(${numBlocks_bench} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
# target_include_directories(${numBlocks_bench} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/benchmark)
# target_link_libraries(${numBlocks_bench} PRIVATE tiling cpumultiply nvbench::main)
# set_target_properties(${numBlocks_bench} PROPERTIES COMPILE_FEATURES cuda_std_17)

# set(partKernelsName part_kernels.out)
# add_executable(${partKernelsName} part_kernels.cu)
# target_compile_options(${partKernelsName} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas;-v;-lineinfo;>")
# target_include_directories(${partKernelsName} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/benchmark)
# target_link_libraries(${partKernelsName} PRIVATE tiling cpumultiply nvbench::main)
# set_target_properties(${partKernelsName} PROPERTIES COMPILE_FEATURES cuda_std_17)