CC=g++
CFLAGS=-O3 -mavx2 -mfma -fopenmp -march=znver3

all: gemm_avx

gemm_avx: gemm_avx.cpp
	$(CC) $(CFLAGS) gemm_avx.cpp -o gemm_avx

clean:
	rm -f gemm_avx
