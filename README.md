# Project Structure
- in `./include` you find our kernel implementations
- in `./tests` are unit test for each final version checking correctness
- in `./benchmark` are the nvbench files for our different kernels




## Notes:
The provided tiling algorithm in `./tiling/tiling.cpp` has been extended
by our simple tiling algorithm.

The function signatures of our initial distance 2 versions still assumes that our distance 2
kernels count collisions for both, distance 1 and 2.
We droped that implementation because seperate performace evaluation was more importand
for our analysis.
Implementing both should be very easy with the existing device functions.