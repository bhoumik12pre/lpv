#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define BLOCK_DIM 16

__global__ void matMul(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];

        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM);

    matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %.0f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}