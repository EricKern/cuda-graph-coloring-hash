# Project
This project provides parallel graph coloring algorithms implemented in cuda.
It was part of the Advanced Parallel Algorithms Course at Heidelberg University.

The coloring is done with a simple hash function. The algorithms afterward count the
collisions for a given number of allowed colors and differently parametrized hash functions.

## Structure
- in `./include` you find our kernel implementations
- in `./tests` are unit tests for each final version checking correctness against cpu versions
- in `./benchmark` are the nvbench files for our different kernels

- `APA_23_final.pdf` is the final version of the presentation
- `example.cu` is an example with all the boilerplate code.\
You can call it with `example.out -mat path/to/matrixmarket_file.mtx`

Matrices are available at http://sparse.tamu.edu/.
The algorithms assume 100% pattern symmetry

## Build:
This project has been built with the following spack packages

    cuda@11.8.0
    gcc@11.3.0
    cmake@3.23.1
    metis@5.1.0

    and nvbench@main

Please use cmake to build the project in a build directory.

For example:

```
mkdir build && cd build
cmake ..
make
```

## Notes:
The provided tiling algorithm in `./tiling/tiling.cpp` has been extended
by our simple tiling algorithm.

The function signatures of our initial distance 2 versions still assume that our distance 2
kernels count collisions for both, distance 1 and 2.
We dropped that implementation because a separate performance evaluation was more important
for our analysis.
Implementing both should be very easy with the existing device functions.
