#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>
#include <time.h>

#define PI 3.14159265358979323846
#define MAX_THREADS 4

int thread_count = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    complex double *x;
    int n;
} FFTArgs;

void *FFT_thread(void *args);

void FFT(complex double *x, int n) {
    if (n <= 1) return;

    complex double *even = malloc(n/2 * sizeof(complex double));
    complex double *odd = malloc(n/2 * sizeof(complex double));

    for (int i = 0; i < n/2; i++) {
        even[i] = x[i*2];
        odd[i]  = x[i*2 + 1];
    }

    pthread_t thread1, thread2;
    FFTArgs args1 = {even, n/2};
    FFTArgs args2 = {odd, n/2};

    int spawn_threads = 0;

    pthread_mutex_lock(&mutex);
    if (thread_count < MAX_THREADS) {
        thread_count++;
        spawn_threads = 1;
    }
    pthread_mutex_unlock(&mutex);

    if (spawn_threads) {
        pthread_create(&thread1, NULL, FFT_thread, &args1);
        pthread_create(&thread2, NULL, FFT_thread, &args2);
        pthread_join(thread1, NULL);
        pthread_join(thread2, NULL);

        pthread_mutex_lock(&mutex);
        thread_count--;
        pthread_mutex_unlock(&mutex);
    } else {
        FFT(even, n/2);
        FFT(odd, n/2);
    }

    for (int k = 0; k < n/2; k++) {
        complex double t = cexp(-2.0 * I * PI * k / n) * odd[k];
        x[k]       = even[k] + t;
        x[k + n/2] = even[k] - t;
    }

    free(even);
    free(odd);
}

void *FFT_thread(void *args) {
    FFTArgs *fftArgs = (FFTArgs *)args;
    FFT(fftArgs->x, fftArgs->n);
    return NULL;
}

int main() {
    for (int n = 1; n <= 20; n++) {
        complex double *x = malloc(n * sizeof(complex double));

        // Generate random input
        for (int i = 0; i < n; i++) {
            double real = rand() % 10;  // 0 to 9
            double imag = rand() % 10;
            x[i] = real + imag * I;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        FFT(x, n);

        clock_gettime(CLOCK_MONOTONIC, &end);
        long elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
        double elapsed_ms = elapsed_ns / 1e6;

        printf("n = %2d | Time taken: %8ld ns | %.6f ms\n", n, elapsed_ns, elapsed_ms);

        free(x);
    }

    return 0;
}
