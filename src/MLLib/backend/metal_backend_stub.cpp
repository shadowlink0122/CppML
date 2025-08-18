/**
 * @file metal_backend_stub.cpp
 * @brief CPU-based stub implementation for Metal backend
 * Used in CI environments where Metal support is not available
 */

#include "../../../include/MLLib/backend/metal_backend.hpp"
#include "../../../include/MLLib/ndarray.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace MLLib {
namespace Backend {

// Static member initialization for stub
bool MetalBackend::initialized_ = false;
void* MetalBackend::device_ = nullptr;
void* MetalBackend::command_queue_ = nullptr;
void* MetalBackend::library_ = nullptr;

bool MetalBackend::isAvailable() {
  return false;  // Metal is not available in CI/Linux environments
}

void MetalBackend::initialize() {
  if (initialized_) {
    return;
  }

  std::cout << "⚠️  MetalBackend: Using CPU fallback (Metal not available)"
            << std::endl;
  initialized_ = true;
}

void MetalBackend::cleanup() {
  if (!initialized_) {
    return;
  }

  std::cout << "🧹 MetalBackend: Cleaning up CPU fallback" << std::endl;
  initialized_ = false;
}

// Memory management (CPU fallback)
void* MetalBackend::allocateMemory(size_t size) {
  std::cout << "⚠️  MetalBackend: CPU malloc for " << size << " bytes"
            << std::endl;
  return malloc(size);
}

void MetalBackend::deallocateMemory(void* ptr) {
  if (ptr) {
    free(ptr);
  }
}

void MetalBackend::copyToDevice(void* dst, const void* src, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU memcpy (to device fallback)" << std::endl;
  memcpy(dst, src, size);
}

void MetalBackend::copyFromDevice(void* dst, const void* src, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU memcpy (from device fallback)"
            << std::endl;
  memcpy(dst, src, size);
}

void MetalBackend::copyDeviceToDevice(void* dst, const void* src, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU memcpy (device to device fallback)"
            << std::endl;
  memcpy(dst, src, size);
}

// BLAS operations (CPU fallback)
void MetalBackend::gemm(bool transposeA, bool transposeB, int m, int n, int k,
                        double alpha, const double* A, int lda, const double* B,
                        int ldb, double beta, double* C, int ldc) {
  std::cout << "⚠️  MetalBackend: CPU GEMM fallback (" << m << "x" << n << "x"
            << k << ")" << std::endl;

  // Simple CPU GEMM implementation
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int l = 0; l < k; l++) {
        double a_val = transposeA ? A[l * lda + i] : A[i * lda + l];
        double b_val = transposeB ? B[j * ldb + l] : B[l * ldb + j];
        sum += a_val * b_val;
      }
      C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
    }
  }
}

// High-level matrix operations
void MetalBackend::matmul(const double* A, const double* B, double* C, int m,
                          int n, int k) {
  std::cout << "⚠️  MetalBackend: CPU matmul fallback (" << m << "x" << n << "x"
            << k << ")" << std::endl;

  // Simple matrix multiplication: C = A * B
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

// Activation functions (CPU fallback)
void MetalBackend::relu(const double* input, double* output, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU ReLU fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    output[i] = std::max(0.0, input[i]);
  }
}

void MetalBackend::sigmoid(const double* input, double* output, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU Sigmoid fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }
}

void MetalBackend::tanh_activation(const double* input, double* output,
                                   size_t size) {
  std::cout << "⚠️  MetalBackend: CPU Tanh fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    output[i] = std::tanh(input[i]);
  }
}

// Extended activation functions
void MetalBackend::leaky_relu(const double* input, double* output, size_t size,
                              double alpha) {
  std::cout << "⚠️  MetalBackend: CPU Leaky ReLU fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] > 0 ? input[i] : alpha * input[i];
  }
}

void MetalBackend::gelu(const double* input, double* output, size_t size,
                        bool approximate) {
  std::cout << "⚠️  MetalBackend: CPU GELU fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    if (approximate) {
      // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 *
      // x^3)))
      double x = input[i];
      double x3 = x * x * x;
      double inner = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x3);
      output[i] = 0.5 * x * (1.0 + std::tanh(inner));
    } else {
      // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
      output[i] = 0.5 * input[i] * (1.0 + std::erf(input[i] / std::sqrt(2.0)));
    }
  }
}

void MetalBackend::elu(const double* input, double* output, size_t size,
                       double alpha) {
  std::cout << "⚠️  MetalBackend: CPU ELU fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] > 0 ? input[i] : alpha * (std::exp(input[i]) - 1.0);
  }
}

void MetalBackend::swish(const double* input, double* output, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU Swish fallback" << std::endl;
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] / (1.0 + std::exp(-input[i]));
  }
}

void MetalBackend::softmax(const double* input, double* output, size_t size) {
  std::cout << "⚠️  MetalBackend: CPU Softmax fallback" << std::endl;

  // Find maximum for numerical stability
  double max_val = input[0];
  for (size_t i = 1; i < size; i++) {
    max_val = std::max(max_val, input[i]);
  }

  // Compute exponentials and sum
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }

  // Normalize
  for (size_t i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

// Utility functions
void MetalBackend::synchronize() {
  // No-op for CPU fallback
}

int MetalBackend::getDeviceCount() {
  return 0;  // No Metal devices available
}

void MetalBackend::setDevice(int device) {
  if (device != 0) {
    std::cout << "⚠️  MetalBackend: Cannot set device " << device
              << " (CPU fallback)" << std::endl;
  }
}

std::string MetalBackend::getDeviceName(int /* device */) {
  return "CPU Fallback (Metal not available)";
}

void MetalBackend::initializeKernels() {
  // No-op for CPU fallback
}

}  // namespace Backend
}  // namespace MLLib
