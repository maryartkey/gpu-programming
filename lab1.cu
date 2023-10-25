#include <stdio.h>
#include <math.h>

// CUDA kernel to initialize the array with sin values
__global__
void initializeArray(float* arr) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numElements = gridDim.x * blockDim.x;
    
    for(int i = idx; i < numElements; i += numElements) 
    {
        arr[i] = sin((i % 360) * M_PI / 180.0);
    }
}

float calculateError(float* arr, int numElements)
{
    float err = 0;
    for(int i = 0; i < numElements; i++)
    {
    	err += (abs(sin((i % 360) * M_PI / 180) - arr[i]))/10^8;
    }
    return err;
}

int main() 
{
    int numElements = 100000000;    
    size_t size = numElements * sizeof(float);

    // Allocate memory on the GPU
    float* d_arr;
    cudaMalloc((void**)&d_arr, size);

    // Define grid and block dimensions for kernel execution
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    initializeArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr);

    // Copy the data back to the CPU
    float* h_arr = new float[size];
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Calculate the error
    print("Error= \n", calculateError(h_arr, numElements));

    // Free GPU memory
    cudaFree(d_arr);

    // Free CPU memory
    free(h_arr);
    printf("All good!\n");
    return 0;
}
