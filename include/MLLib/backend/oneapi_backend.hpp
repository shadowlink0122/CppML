/**
 * @file oneapi_backend.hpp
 * @brief Intel oneAPI backend implementation for MLLib
 */

#pragma once

#ifdef WITH_ONEAPI

// Check if oneAPI headers are available
#if __has_include(<sycl/sycl.hpp>)
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#define ONEAPI_AVAILABLE 1
#else
#define ONEAPI_AVAILABLE 0
// Mock types when oneAPI is not available
namespace sycl {
class queue {};
class device {};
}  // namespace sycl
#endif

#include "../ndarray.hpp"
#include "backend.hpp"

namespace MLLib {
namespace Backend {

/**
 * @class OneAPIBackend
 * @brief Intel oneAPI implementation of GPU backend
 */
class OneAPIBackend {
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
  static sycl::queue* queue_;
  static bool initialized_;
  static sycl::device current_device_;
};

}  // namespace Backend
}  // namespace MLLib

#endif  // WITH_ONEAPI
