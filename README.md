# Project Structure
- in `./include` you find our kernel implementations
- in `./tests` are unit test for each final version checking correctness against cpu versions
- in `./benchmark` are the nvbench files for our different kernels

- `APA_23_initial.pdf` is the initial version of the presentation
- `APA_23_final.pdf` has the discussed feedback included


## Build:
This project has been built with the following spack packages

    cuda@11.8.0
    gcc@11.3.0
    cmake@3.23.1
    metis@5.1.0

    and nvbench@main

Pleas use cmake to build the project in a build directory.

For example do:

```
mkdir build && cd build
cmake ..
make
```

## Notes:
The provided tiling algorithm in `./tiling/tiling.cpp` has been extended
by our simple tiling algorithm.

The function signatures of our initial distance 2 versions still assumes that our distance 2
kernels count collisions for both, distance 1 and 2.
We droped that implementation because seperate performace evaluation was more importand
for our analysis.
Implementing both should be very easy with the existing device functions.
