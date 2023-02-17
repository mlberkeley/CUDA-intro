#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)

// Kernel definition
__global__ void VecAdd(float *A, float *B,
                       float *C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    *(C + i) = *(A + i) + *(B + i);
}

int main()
{
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    cudaError_t err = cudaSuccess;
    float *d_A, *d_B, *d_C;
    err = cudaMalloc(&d_A, size);
    err = cudaMalloc(&d_B, size);
    err = cudaMalloc(&d_C, size);

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32);
    dim3 numBlocks(N / threadsPerBlock.x);
    printf("numBlocks: %d\n", numBlocks.x);
    VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool error = false;
    for (int i = 0; i < N; i++)
    {
        // check if correct
        if (h_C[i] != h_A[i] + h_B[i])
        {
            error = true;
            printf("Error: %f + %f != %f\n", h_A[i], h_B[i], h_C[i]);
            break;
        }
    }
    if (!error)
    {
        printf("Success!\n");
    }
}
