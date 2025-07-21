#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <time.h>

#define PI 3.14159265358979323846

void FFT(complex double *x, int n) {
    if (n <= 1) return;

    complex double *even = malloc(n/2 * sizeof(complex double));
    complex double *odd  = malloc(n/2 * sizeof(complex double));

    for (int i = 0; i < n/2; i++) {
        even[i] = x[i*2];
        odd[i]  = x[i*2 + 1];
    }

    // Parallel sections for FFT calls on even and odd
    #pragma omp parallel sections
    {
        #pragma omp section
        FFT(even, n/2);

        #pragma omp section
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

int main() {
    srand((unsigned int)time(NULL));

    printf("n\tTime (ns)\tTime (ms)\n");
    printf("--------------------------------------\n");

    for (int n = 1; n <= 20; n++) {
        complex double *x = malloc(n * sizeof(complex double));

        // Fill with random complex numbers
        for (int i = 0; i < n; i++) {
            double real = rand() % 10 + 1;
            double imag = rand() % 10 + 1;
            x[i] = real + imag * I;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        FFT(x, n);

        clock_gettime(CLOCK_MONOTONIC, &end);

        long elapsed_ns = (end.tv_sec - start.tv_sec) * 1e9 +
                          (end.tv_nsec - start.tv_nsec);
        double elapsed_ms = elapsed_ns / 1e6;

        printf("%2d\t%10ld\t%.6f\n", n, elapsed_ns, elapsed_ms);
        free(x);
    }

    return 0;
}
