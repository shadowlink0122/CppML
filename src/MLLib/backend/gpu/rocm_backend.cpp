/**
 * @file rocm_backend.cpp
 * @brief AMD ROCm backend implementation
 */

#ifdef WITH_ROCM

// Check if ROCm headers are available
#ifdef __has_include
#if __has_include(<hip/hip_runtime.h>) && __has_include(<hipblas.h>)
#define ROCM_HEADERS_AVAILABLE
#endif
#endif

#ifdef ROCM_HEADERS_AVAILABLE
#include "../../../../include/MLLib/backend/rocm_backend.hpp"
#include <cstdio>
#include <stdexcept>

namespace MLLib {
namespace Backend {

hipblasHandle_t ROCmBackend::hipblas_handle_ = nullptr;
bool ROCmBackend::initialized_ = false;

bool ROCmBackend::isAvailable() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  return (error == hipSuccess && device_count > 0);
}

void ROCmBackend::initialize() {
  if (initialized_) return;

  hipError_t hip_status = hipInit(0);
  if (hip_status != hipSuccess) {
    throw std::runtime_error("Failed to initialize HIP runtime");
  }

  hipblasStatus_t status = hipblasCreate(&hipblas_handle_);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create hipBLAS handle");
  }

  initialized_ = true;
  printf("ROCm backend initialized successfully\n");
}

void ROCmBackend::cleanup() {
  if (!initialized_) return;

  if (hipblas_handle_) {
    hipblasDestroy(hipblas_handle_);
    hipblas_handle_ = nullptr;
  }

  initialized_ = false;
}

void* ROCmBackend::allocateMemory(size_t size) {
  void* ptr = nullptr;
  hipError_t error = hipMalloc(&ptr, size);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to allocate device memory");
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
    throw std::runtime_error("Failed to copy data to device");
  }
}

void ROCmBackend::copyFromDevice(void* dst, const void* src, size_t size) {
  hipError_t error = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to copy data from device");
  }
}

void ROCmBackend::copyDeviceToDevice(void* dst, const void* src, size_t size) {
  hipError_t error = hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
  if (error != hipSuccess) {
    throw std::runtime_error("Failed to copy data between devices");
  }
}

void ROCmBackend::gemm(bool transposeA, bool transposeB, int m, int n, int k,
                       double alpha, const double* A, int lda, const double* B,
                       int ldb, double beta, double* C, int ldc) {

  hipblasOperation_t opA = transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t opB = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  hipblasStatus_t status = hipblasDgemm(hipblas_handle_, opA, opB, m, n, k,
                                        &alpha, A, lda, B, ldb, &beta, C, ldc);

  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("hipBLAS GEMM operation failed");
  }
}

void ROCmBackend::matmul(const double* A, const double* B, double* C,
                         int rows_a, int cols_a, int cols_b) {
  // Optimized matrix multiplication using rocBLAS
  const double alpha = 1.0, beta = 0.0;
  gemm(false, false, rows_a, cols_b, cols_a, alpha, A, rows_a, B, cols_a, beta,
       C, rows_a);
}

// HIP kernel implementations
__global__ void hip_relu_kernel(const double* input, double* output,
                                size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmax(0.0, input[idx]);
  }
}

__global__ void hip_sigmoid_kernel(const double* input, double* output,
                                   size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
  }
}

__global__ void hip_tanh_kernel(const double* input, double* output,
                                size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanh(input[idx]);
  }
}

void ROCmBackend::relu(const double* input, double* output, size_t size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(hip_relu_kernel, blocks, threads_per_block, 0, 0, input,
                     output, size);

  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    throw std::runtime_error("HIP ReLU kernel launch failed");
  }
}

void ROCmBackend::sigmoid(const double* input, double* output, size_t size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(hip_sigmoid_kernel, blocks, threads_per_block, 0, 0, input,
                     output, size);

  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    throw std::runtime_error("HIP Sigmoid kernel launch failed");
  }
}

void ROCmBackend::tanh_activation(const double* input, double* output,
                                  size_t size) {
  int threads_per_block = 256;
  int blocks = (size + threads_per_block - 1) / threads_per_block;

  hipLaunchKernelGGL(hip_tanh_kernel, blocks, threads_per_block, 0, 0, input,
                     output, size);

  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    throw std::runtime_error("HIP Tanh kernel launch failed");
  }
}

void ROCmBackend::synchronize() {
  hipError_t error = hipDeviceSynchronize();
  if (error != hipSuccess) {
    throw std::runtime_error("Device synchronization failed");
  }
}

int ROCmBackend::getDeviceCount() {
  int count = 0;
  hipGetDeviceCount(&count);
  return count;
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

}  // namespace Backend
}  // namespace MLLib

#else  // !ROCM_HEADERS_AVAILABLE

// Stub implementation when ROCm headers are not available
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Forward declaration for ROCmBackend class
namespace MLLib {
namespace Backend {
class ROCmBackend;
}
}  // namespace MLLib

// Mock hipBLAS handle type
typedef void* hipblasHandle_t;

#include "../../../../include/MLLib/backend/rocm_backend.hpp"

namespace MLLib {
namespace Backend {

hipblasHandle_t ROCmBackend::hipblas_handle_ = nullptr;
bool ROCmBackend::initialized_ = false;

bool ROCmBackend::isAvailable() {
  printf("ROCm backend: Headers not available, ROCm not supported\n");
  return false;
}

void ROCmBackend::initialize() {
  printf("ROCm backend: Headers not available, initialization skipped\n");
}

void ROCmBackend::cleanup() {
  // Nothing to cleanup in stub mode
}

void* ROCmBackend::allocateMemory(size_t size) {
  (void)size;  // Suppress unused parameter warning
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::deallocateMemory(void* ptr) {
  (void)ptr;  // Suppress unused parameter warning
}

void ROCmBackend::copyToDevice(void* dst, const void* src, size_t size) {
  (void)dst;
  (void)src;
  (void)size;
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::copyFromDevice(void* dst, const void* src, size_t size) {
  (void)dst;
  (void)src;
  (void)size;
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::copyDeviceToDevice(void* dst, const void* src, size_t size) {
  (void)dst;
  (void)src;
  (void)size;
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::gemm(bool transposeA, bool transposeB, int m, int n, int k,
                       double alpha, const double* A, int lda, const double* B,
                       int ldb, double beta, double* C, int ldc) {
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
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::relu(const double* input, double* output, size_t size) {
  (void)input;
  (void)output;
  (void)size;
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::sigmoid(const double* input, double* output, size_t size) {
  (void)input;
  (void)output;
  (void)size;
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::tanh_activation(const double* input, double* output,
                                  size_t size) {
  (void)input;
  (void)output;
  (void)size;
  throw std::runtime_error("ROCm backend not available");
}

void ROCmBackend::synchronize() {
  // Nothing to synchronize in stub mode
}

int ROCmBackend::getDeviceCount() {
  return 0;
}

void ROCmBackend::setDevice(int device) {
  (void)device;
  throw std::runtime_error("ROCm backend not available");
}

std::string ROCmBackend::getDeviceName(int device) {
  (void)device;
  return "ROCm not available";
}

}  // namespace Backend
}  // namespace MLLib

#endif  // ROCM_HEADERS_AVAILABLE

#endif  // WITH_ROCM
