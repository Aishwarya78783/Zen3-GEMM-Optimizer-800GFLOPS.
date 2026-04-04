# Zen3-GEMM-Optimizer-800GFLOPS.

# Zen3-GEMM-Optimizer-700GFLOPS

## 🚀 Overview
A high-performance implementation of General Matrix Multiplication (GEMM) specifically optimized for the **AMD Zen 3 (Family 19h)** micro-architecture. This project demonstrates how to bypass the "Memory Wall" and maximize the utilization of Fused Multiply-Add (FMA) units.

**Performance Benchmark:**
- **Hardware:** Ryzen 5000 Series (6 Cores / 12 Threads)
- **Theoretical Peak:** ~800 GFLOPS (at 4.2GHz boost)
- **Achieved Performance:** **[INSERT YOUR REAL AVG NUMBER] GFLOPS**
- **Efficiency:** ~[X]% of Theoretical Peak

## 🛠️ Technical Optimizations (First Principles Approach)

### 1. Register Blocking (6x16 Micro-kernel)
I manually managed the 16 YMM registers to store a 6x16 tile of Matrix C. This minimizes register pressure while maximizing arithmetic intensity (192 ops per 3 loads).

### 2. Multi-Level Cache Tiling
Implemented a 3-level blocking strategy to ensure data residency in L1 (32KB) and L2 (512KB) caches, hiding DRAM latency.

### 3. SIMD Vectorization (AVX2 + FMA3)
Utilized `immintrin.h` to invoke hardware-level instructions (`_mm256_fmadd_ps`), processing 8 single-precision floats per cycle per FMA unit.

### 4. Data Packing & Memory Alignment
Implemented custom packing functions to ensure Matrix B is stored contiguously in memory, enabling the CPU prefetcher to operate at full bandwidth with zero cache-line splits.

## 🚀 How to Build & Run

### Prerequisites
- Linux / WSL2
- `g++` (GCC 11+)
- OpenMP

### Compilation
```bash
g++ -O3 -mavx2 -mfma -march=native -fopenmp zen_800_final.cpp -o zen_gemm
