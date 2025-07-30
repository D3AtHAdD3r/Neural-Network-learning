#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel_12() {
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA!\n");
}

int main_74645() {
    testKernel_12 << <1, 1 >> > ();
    cudaDeviceSynchronize();
    std::cout << "CUDA test completed." << std::endl;
    return 0;
}