#!/bin/bash
#SBATCH -o test.txt
#SBATCH -p skylake
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --exclusive

./build/benchmark.out --device 0 --min-samples 50 --timeout 2000

# ncu -f -o profile2 --set full ./build/main-coloring.out 
