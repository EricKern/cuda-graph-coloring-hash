#!/bin/bash
#SBATCH -o %j-bench.txt
#SBATCH -p skylake
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --ntasks-per-node=8

#gpu:gtx_1080_ti:1      gpu:rtx_2080_ti:1
../rls-build/numBlocks_bench.out --devices 0 --timeout 30