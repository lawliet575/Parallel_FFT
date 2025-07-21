#include <cuda.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

__device__ cuDoubleComplex complex_exp(double theta) {
    return make_cuDoubleComplex(cos(theta), sin(theta));
}

__global__ void fft_kernel(cuDoubleComplex *X, int N, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * step * 2;
    if (i + step < N) {
        for (int j = 0; j < step; j++) {
            cuDoubleComplex t = cuCmul(complex_exp(-2.0 * PI * j / (2.0 * step)), X[i + j + step]);
            cuDoubleComplex u = X[i + j];
            X[i + j] = cuCadd(u, t);
            X[i + j + step] = cuCsub(u, t);
        }
    }
}

__global__ void bit_reverse(cuDoubleComplex *X, int N, int logN) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    unsigned int rev = 0;
    unsigned int x = tid;
    for (int i = 0; i < logN; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    if (rev > tid) {
        cuDoubleComplex temp = X[tid];
        X[tid] = X[rev];
        X[rev] = temp;
    }
}

void cuda_fft(cuDoubleComplex *h_X, int N) {
    cuDoubleComplex *d_X;
    cudaMalloc(&d_X, sizeof(cuDoubleComplex) * N);
    cudaMemcpy(d_X, h_X, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);
    int logN = log2(N);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    bit_reverse<<<numBlocks, blockSize>>>(d_X, N, logN);
    cudaDeviceSynchronize();
    for (int step = 1; step < N; step *= 2) {
        int numThreads = N / (2 * step);
        int blocks = (numThreads + blockSize - 1) / blockSize;
        fft_kernel<<<blocks, blockSize>>>(d_X, N, step);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_X, d_X, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_X);
}

int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main() {
    for (int n = 1; n <= 20; n++) {
        int N = next_power_of_2(n);
        cuDoubleComplex *x = (cuDoubleComplex *)malloc(N * sizeof(cuDoubleComplex));
        for (int i = 0; i < n; i++) {
            double real = rand() % 10;
            double imag = rand() % 10;
            x[i] = make_cuDoubleComplex(real, imag);
        }
        for (int i = n; i < N; i++) {
            x[i] = make_cuDoubleComplex(0, 0);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cuda_fft(x, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("n = %2d | Time taken: %.6f ms\n", n, ms);
        free(x);
    }
    return 0;
}
