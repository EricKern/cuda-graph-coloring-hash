#!/bin/bash
#SBATCH -o %j-test.txt
#SBATCH -p skylake
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks-per-node=8

#gpu:gtx_1080_ti:1      gpu:rtx_2080_ti:1
../rls-build/tests/kernelTestsD1.out
../rls-build/tests/kernelTestsD2.out

# or use ctest