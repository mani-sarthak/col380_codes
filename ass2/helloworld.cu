#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello, CUDA World!\n");
}

int main() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the GPU to finish
    return 0;
}
