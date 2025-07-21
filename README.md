# 🚀 Fast Fourier Transform (FFT) in Parallel Computing

This project explores the implementation and performance of the Fast Fourier Transform (FFT) algorithm across different parallel computing paradigms: **multithreading using pthreads**, **OpenMP**, and **CUDA using GPUs**. It benchmarks execution times across input sizes and highlights scalability differences between CPU-based and GPU-based approaches.

## 🧵 Multithreaded FFT (pthreads)

This version manually creates threads to compute FFT recursively on even and odd parts of the array. Thread creation is limited via `MAX_THREADS`, and protected using mutexes.

### 🔧 Run Instructions

Make sure you have `gcc` installed.

```bash
# Compile the pthread-based FFT
gcc -o fft_pthreads FFT_Parallel_MT.c -lm -lpthread

# Run the executable
./fft_pthreads
```

## 🔱 OpenMP FFT

This version uses #pragma omp parallel sections to parallelize FFT calls on even and odd indices. It simplifies thread management with compiler directives.

### 🔧 Run Instructions

```bash
# Compile with OpenMP support
gcc -fopenmp -o fft_openmp FFT_Parallel_OPENMP.c -lm

# Run the executable
./fft_openmp
```

## 🚀 CUDA FFT via Jupyter Notebook

Implements FFT kernels using CUDA for GPU acceleration. Includes bit-reversal logic and core butterfly stages optimized for parallel execution.

### 📌 How to Run
Use platforms like Kaggle or Colab with GPU enabled.

1. Upload fft_cuda.ipynb to your workspace.

2. Enable GPU from runtime/environment settings.

3. Execute notebook cells to build and run the FFT kernels.

Timing results for each input size (padded to next power of two) will be displayed.
