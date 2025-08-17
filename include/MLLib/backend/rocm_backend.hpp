/**
 * @file rocm_backend.hpp
 * @brief AMD ROCm backend implementation for MLLib
 */

#pragma once

#ifdef WITH_ROCM

// Check if HIP headers are available
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#include <hipblas.h>
#define HIP_AVAILABLE 1
#else
#define HIP_AVAILABLE 0
// Mock types when HIP is not available
typedef void* hipblasHandle_t;
#endif

#include "../ndarray.hpp"
#include "backend.hpp"

namespace MLLib {
namespace Backend {

/**
 * @class ROCmBackend
 * @brief AMD ROCm implementation of GPU backend
 */
class ROCmBackend {
public:
  static bool isAvailable();
  static void initialize();
  static void cleanup();

  // Memory management
  static void* allocateMemory(size_t size);
  static void deallocateMemory(void* ptr);
  static void copyToDevice(void* dst, const void* src, size_t size);
  static void copyFromDevice(void* dst, const void* src, size_t size);
  static void copyDeviceToDevice(void* dst, const void* src, size_t size);

  // BLAS operations
  static void gemm(bool transposeA, bool transposeB, int m, int n, int k,
                   double alpha, const double* A, int lda, const double* B,
                   int ldb, double beta, double* C, int ldc);

  // Optimized matrix operations
  static void matmul(const double* A, const double* B, double* C, int rows_a,
                     int cols_a, int cols_b);

  // Activation functions
  static void relu(const double* input, double* output, size_t size);
  static void sigmoid(const double* input, double* output, size_t size);
  static void tanh_activation(const double* input, double* output, size_t size);

  // Utility functions
  static void synchronize();
  static int getDeviceCount();
  static void setDevice(int device);
  static std::string getDeviceName(int device);

private:
  static hipblasHandle_t hipblas_handle_;
  static bool initialized_;
};

}  // namespace Backend
}  // namespace MLLib

#endif  // WITH_ROCM
