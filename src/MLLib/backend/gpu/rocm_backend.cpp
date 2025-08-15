/**
 * @file rocm_backend.cpp
 * @brief AMD ROCm backend implementation
 */

#ifdef WITH_ROCM

#include "../../../../include/MLLib/backend/rocm_backend.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace MLLib {
namespace Backend {

#if HIP_AVAILABLE
hipblasHandle_t ROCmBackend::hipblas_handle_ = nullptr;
bool ROCmBackend::initialized_ = false;

bool ROCmBackend::isAvailable() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  return (error == hipSuccess && device_count > 0);
}

void ROCmBackend::initialize() {
  if (!initialized_) {
    hipError_t error = hipInit(0);
    if (error != hipSuccess) {
      throw std::runtime_error("Failed to initialize HIP");
    }

    hipblasStatus_t status = hipblasCreate(&hipblas_handle_);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create hipBLAS handle");
    }

    initialized_ = true;
    printf("ROCm backend initialized successfully\n");
  }
}

void ROCmBackend::cleanup() {
  if (initialized_ && hipblas_handle_) {
    hipblasDestroy(hipblas_handle_);
    hipblas_handle_ = nullptr;
    initialized_ = false;
    printf("ROCm backend cleaned up\n");
  }
}

void* ROCmBackend::allocateMemory(size_t size) {
  void* ptr;
  hipError_t error = hipMalloc(&ptr, size);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to allocate HIP memory");
  }
  return ptr;
}

void ROCmBackend::deallocateMemory(void* ptr) {
  if (ptr) {
    hipFree(ptr);
  }
}

void ROCmBackend::copyToDevice(void* dst, const void* src, size_t size) {
  hipError_t error = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to copy to device");
  }
}

void ROCmBackend::copyFromDevice(void* dst, const void* src, size_t size) {
  hipError_t error = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to copy from device");
  }
}

void ROCmBackend::copyDeviceToDevice(void* dst, const void* src, size_t size) {
  hipError_t error = hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to copy device to device");
  }
}

void ROCmBackend::gemm(bool transposeA, bool transposeB, int m, int n, int k,
                       double alpha, const double* A, int lda, const double* B,
                       int ldb, double beta, double* C, int ldc) {
  if (!initialized_) {
    throw std::runtime_error("ROCm backend not initialized");
  }

  hipblasOperation_t transa = transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t transb = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  hipblasStatus_t status =
      hipblasDgemm(hipblas_handle_, transa, transb, m, n, k, &alpha, A, lda, B,
                   ldb, &beta, C, ldc);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("hipBLAS GEMM failed");
  }
}

void ROCmBackend::relu(const double* input, double* output, size_t size) {
  // TODO: Implement HIP kernel for ReLU
  for (size_t i = 0; i < size; ++i) {
    output[i] = std::max(0.0, input[i]);
  }
}

void ROCmBackend::sigmoid(const double* input, double* output, size_t size) {
  // TODO: Implement HIP kernel for Sigmoid
  for (size_t i = 0; i < size; ++i) {
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }
}

void ROCmBackend::tanh_activation(const double* input, double* output,
                                  size_t size) {
  // TODO: Implement HIP kernel for Tanh
  for (size_t i = 0; i < size; ++i) {
    output[i] = std::tanh(input[i]);
  }
}

void ROCmBackend::synchronize() {
  hipError_t error = hipDeviceSynchronize();
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to synchronize device");
  }
}

int ROCmBackend::getDeviceCount() {
  int device_count = 0;
  hipGetDeviceCount(&device_count);
  return device_count;
}

void ROCmBackend::setDevice(int device) {
  hipError_t error = hipSetDevice(device);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to set device");
  }
}

std::string ROCmBackend::getDeviceName(int device) {
  hipDeviceProp_t prop;
  hipError_t error = hipGetDeviceProperties(&prop, device);
  if (error != hipSuccess) {
    return "Unknown AMD GPU";
  }
  return std::string(prop.name);
}

#else
// Stub implementation when HIP is not available
hipblasHandle_t ROCmBackend::hipblas_handle_ = nullptr;
bool ROCmBackend::initialized_ = false;

bool ROCmBackend::isAvailable() {
  printf("ROCm backend: HIP headers not available, ROCm not supported\n");
  return false;
}

void ROCmBackend::initialize() {
  printf("ROCm backend: Stub implementation - initialize called\n");
}

void ROCmBackend::cleanup() {
  printf("ROCm backend: Stub implementation - cleanup called\n");
}

void* ROCmBackend::allocateMemory(size_t size) {
  printf("ROCm backend: Stub implementation - allocateMemory called\n");
  return malloc(size);
}

void ROCmBackend::deallocateMemory(void* ptr) {
  printf("ROCm backend: Stub implementation - deallocateMemory called\n");
  free(ptr);
}

void ROCmBackend::copyToDevice(void* dst, const void* src, size_t size) {
  printf("ROCm backend: Stub implementation - copyToDevice called\n");
  memcpy(dst, src, size);
}

void ROCmBackend::copyFromDevice(void* dst, const void* src, size_t size) {
  printf("ROCm backend: Stub implementation - copyFromDevice called\n");
  memcpy(dst, src, size);
}

void ROCmBackend::copyDeviceToDevice(void* dst, const void* src, size_t size) {
  printf("ROCm backend: Stub implementation - copyDeviceToDevice called\n");
  memcpy(dst, src, size);
}

void ROCmBackend::gemm(bool transposeA, bool transposeB, int m, int n, int k,
                       double alpha, const double* A, int lda, const double* B,
                       int ldb, double beta, double* C, int ldc) {
  printf("ROCm backend: Stub implementation - gemm called\n");
  // Simple CPU fallback implementation would go here
  (void)transposeA;
  (void)transposeB;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
}

void ROCmBackend::relu(const double* input, double* output, size_t size) {
  printf("ROCm backend: Stub implementation - relu called\n");
  for (size_t i = 0; i < size; ++i) {
    output[i] = std::max(0.0, input[i]);
  }
}

void ROCmBackend::sigmoid(const double* input, double* output, size_t size) {
  printf("ROCm backend: Stub implementation - sigmoid called\n");
  for (size_t i = 0; i < size; ++i) {
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }
}

void ROCmBackend::tanh_activation(const double* input, double* output,
                                  size_t size) {
  printf("ROCm backend: Stub implementation - tanh_activation called\n");
  for (size_t i = 0; i < size; ++i) {
    output[i] = std::tanh(input[i]);
  }
}

void ROCmBackend::synchronize() {
  printf("ROCm backend: Stub implementation - synchronize called\n");
}

int ROCmBackend::getDeviceCount() {
  printf("ROCm backend: Stub implementation - getDeviceCount called\n");
  return 0;
}

void ROCmBackend::setDevice(int device) {
  printf("ROCm backend: Stub implementation - setDevice called\n");
  (void)device;
}

std::string ROCmBackend::getDeviceName(int device) {
  printf("ROCm backend: Stub implementation - getDeviceName called\n");
  (void)device;
  return "ROCm Stub Device";
}

#endif  // HIP_AVAILABLE

}  // namespace Backend
}  // namespace MLLib

#endif  // WITH_ROCM
