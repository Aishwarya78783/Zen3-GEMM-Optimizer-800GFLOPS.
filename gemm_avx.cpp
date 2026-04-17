#include <iostream>
#include <vector>
#include <immintrin.h> // AVX2 Intrinsics
#include <omp.h>       // OpenMP
#include <chrono>

// Optimization: 6x16 Register Blocking using AVX2
// This kernel assumes B is stored in a way that allows contiguous loads
void gemm_micro_kernel_6x16(int K, float* A, int lda, float* B, int ldb, float* C, int ldc) {
    // Registers to hold C block (6 rows x 2 YMM registers for 16 columns)
    __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        // Load two 256-bit chunks from B (8 floats each = 16 floats total)
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        // Broadcast A elements and FMA
        __m256 a;
        #define FMA_STEP(row, reg0, reg1) \
            a = _mm256_set1_ps(A[row * lda + k]); \
            reg0 = _mm256_fmadd_ps(a, b0, reg0); \
            reg1 = _mm256_fmadd_ps(a, b1, reg1);

        FMA_STEP(0, c00, c01);
        FMA_STEP(1, c10, c11);
        FMA_STEP(2, c20, c21);
        FMA_STEP(3, c30, c31);
        FMA_STEP(4, c40, c41);
        FMA_STEP(5, c50, c51);
    }

    // Store results back to C
    _mm256_storeu_ps(&C[0 * ldc + 0], c00); _mm256_storeu_ps(&C[0 * ldc + 8], c01);
    _mm256_storeu_ps(&C[1 * ldc + 0], c10); _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c20); _mm256_storeu_ps(&C[2 * ldc + 8], c21);
    _mm256_storeu_ps(&C[3 * ldc + 0], c30); _mm256_storeu_ps(&C[3 * ldc + 8], c31);
    _mm256_storeu_ps(&C[4 * ldc + 0], c40); _mm256_storeu_ps(&C[4 * ldc + 8], c41);
    _mm256_storeu_ps(&C[5 * ldc + 0], c50); _mm256_storeu_ps(&C[5 * ldc + 8], c51);
}

int main() {
    int N = 2048; // Matrix size
    std::vector<float> A(N * N, 1.0f), B(N * N, 1.0f), C(N * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += 6) {
        for (int j = 0; j < N; j += 16) {
            gemm_micro_kernel_6x16(N, &A[i * N], N, &B[j], N, &C[i * N + j], N);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double flops = 2.0 * N * N * N;
    std::cout << "GFLOPS: " << (flops / diff.count()) / 1e9 << std::endl;

    return 0;
}
