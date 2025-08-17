/**
 * @file oneapi_backend.cpp
 * @brief Intel oneAPI backend implementation (stub version)
 */

#include "../../../../include/MLLib/backend/oneapi_backend.hpp"
#include <cstdio>
#include <stdexcept>

namespace MLLib {
namespace Backend {

// Static member definitions
bool OneAPIBackend::initialized_ = false;

bool OneAPIBackend::isAvailable() {
  // oneAPI not available on this platform
  return false;
}

void OneAPIBackend::initialize() {
  // Stub implementation
}

void OneAPIBackend::cleanup() {
  // Stub implementation
}

void* OneAPIBackend::allocateMemory(size_t size) {
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::deallocateMemory(void* ptr) {
  (void)ptr;
}

void OneAPIBackend::copyToDevice(void* dst, const void* src, size_t size) {
  (void)dst;
  (void)src;
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::copyFromDevice(void* dst, const void* src, size_t size) {
  (void)dst;
  (void)src;
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::copyDeviceToDevice(void* dst, const void* src,
                                       size_t size) {
  (void)dst;
  (void)src;
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::gemm(bool transposeA, bool transposeB, int m, int n, int k,
                         double alpha, const double* A, int lda,
                         const double* B, int ldb, double beta, double* C,
                         int ldc) {
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
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::matmul(const double* A, const double* B, double* C,
                           int rows_a, int cols_a, int cols_b) {
  (void)A;
  (void)B;
  (void)C;
  (void)rows_a;
  (void)cols_a;
  (void)cols_b;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::relu(const double* input, double* output, size_t size) {
  (void)input;
  (void)output;
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::sigmoid(const double* input, double* output, size_t size) {
  (void)input;
  (void)output;
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::tanh_activation(const double* input, double* output,
                                    size_t size) {
  (void)input;
  (void)output;
  (void)size;
  throw std::runtime_error("oneAPI backend not available");
}

void OneAPIBackend::synchronize() {
  // Stub implementation
}

int OneAPIBackend::getDeviceCount() {
  return 0;
}

void OneAPIBackend::setDevice(int device) {
  (void)device;
  throw std::runtime_error("oneAPI backend not available");
}

std::string OneAPIBackend::getDeviceName(int device) {
  (void)device;
  return "oneAPI not available";
}

}  // namespace Backend
}  // namespace MLLib
