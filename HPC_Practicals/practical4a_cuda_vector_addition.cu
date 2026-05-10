#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000
#define BLOCK_SIZE 256

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("C[0] = %.0f\n", h_C[0]);
    printf("C[N-1] = %.0f\n", h_C[N - 1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}