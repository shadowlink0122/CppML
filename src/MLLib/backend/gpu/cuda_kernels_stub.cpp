/**
 * @file cuda_kernels_stub.cpp
 * @brief Stub implementation for CUDA kernels when CUDA is not available
 */

#include "cuda_kernels.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace MLLib {
namespace Backend {
namespace cuda {

// Initialize CUDA context and cuBLAS (stub)
void cuda_init() {
  // Stub implementation - no actual CUDA initialization
  printf("CUDA stub initialized (no GPU hardware available)\n");
}

// Cleanup CUDA context (stub)
void cuda_cleanup() {
  // No-op for stub implementation
}

// Check if CUDA is available (stub)
bool cuda_is_available() {
  // Stub implementation - always return false since no real CUDA
  return false;
}

// Get GPU memory information (stub)
void cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes) {
  // Get memory info for simulation
  if (total_bytes)
    *total_bytes = static_cast<size_t>(2LL * 1024 * 1024 * 1024);  // 2GB
  if (free_bytes)
    *free_bytes = static_cast<size_t>(1LL * 1024 * 1024 * 1024);  // 1GB
}

// GPU matrix multiplication using CPU (stub)
void cuda_matmul(const double* h_a, const double* h_b, double* h_c, int m,
                 int n, int k) {
  // Simple CPU implementation for testing
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int l = 0; l < k; l++) {
        sum += h_a[i * k + l] * h_b[l * n + j];
      }
      h_c[i * n + j] = sum;
    }
  }
}

// GPU element-wise addition using CPU (stub)
void cuda_add(const double* h_a, const double* h_b, double* h_result,
              size_t size) {
  for (size_t i = 0; i < size; i++) {
    h_result[i] = h_a[i] + h_b[i];
  }
}

// GPU element-wise subtraction using CPU (stub)
void cuda_subtract(const double* h_a, const double* h_b, double* h_result,
                   size_t size) {
  for (size_t i = 0; i < size; i++) {
    h_result[i] = h_a[i] - h_b[i];
  }
}

// GPU element-wise multiplication using CPU (stub)
void cuda_multiply(const double* h_a, const double* h_b, double* h_result,
                   size_t size) {
  for (size_t i = 0; i < size; i++) {
    h_result[i] = h_a[i] * h_b[i];
  }
}

// GPU scalar addition using CPU (stub)
void cuda_add_scalar(const double* h_a, double scalar, double* h_result,
                     size_t size) {
  for (size_t i = 0; i < size; i++) {
    h_result[i] = h_a[i] + scalar;
  }
}

// GPU scalar multiplication using CPU (stub)
void cuda_multiply_scalar(const double* h_a, double scalar, double* h_result,
                          size_t size) {
  for (size_t i = 0; i < size; i++) {
    h_result[i] = h_a[i] * scalar;
  }
}

// GPU fill array using CPU (stub)
void cuda_fill(double* h_array, double value, size_t size) {
  std::fill(h_array, h_array + size, value);
}

// GPU copy array using CPU (stub)
void cuda_copy(const double* h_src, double* h_dst, size_t size) {
  std::memcpy(h_dst, h_src, size * sizeof(double));
}

}  // namespace cuda
}  // namespace Backend
}  // namespace MLLib
