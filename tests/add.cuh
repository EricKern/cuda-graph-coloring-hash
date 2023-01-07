


__global__
void add_kernel(float * a, float * b, float *c, int len){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x){
        c[i] = a[i] + b[i];
    }
    
}