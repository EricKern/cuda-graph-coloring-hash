spack load gcc@11.3.0
spack load cuda@11.8.0
spack load cmake@3.24.3%gcc
spack load metis
spack load nvbench

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/asc/pub/spack/opt/spack/linux-centos7-x86_64_v2/gcc-11.3.0/cuda-11.8.0-a2ulnpbrmsu6ncgqspd5zza5ib4lzdpn/lib64/