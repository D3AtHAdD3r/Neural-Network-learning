#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel() {
    printf("Hello from CUDA!\n");
}

int main_xu() {
    testKernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
    std::cout << "CUDA test completed." << std::endl;
    return 0;
}