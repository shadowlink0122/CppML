/**
 * @file metal_backend.hpp
 * @brief Apple Metal backend implementation for MLLib
 */

#pragma once

#ifdef WITH_METAL

#include "../ndarray.hpp"
#include "backend.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#else
// Forward declarations for C++ compilation
typedef void* id;
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLLibrary;
typedef void* MTLComputePipelineState;
#endif

namespace MLLib {
namespace Backend {

// Forward declaration
class ActivationKernelRegistry;

/**
 * @class MetalBackend
 * @brief Apple Metal implementation of GPU backend using generic kernel manager
 */
class MetalBackend {
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

  // High-level matrix operations
  static void matmul(const double* A, const double* B, double* C, int m, int n,
                     int k);

  // Activation functions (now using generic kernel manager)
  static void relu(const double* input, double* output, size_t size);
  static void sigmoid(const double* input, double* output, size_t size);
  static void tanh_activation(const double* input, double* output, size_t size);

  // Extended activation functions
  static void leaky_relu(const double* input, double* output, size_t size,
                         double alpha = 0.01);
  static void gelu(const double* input, double* output, size_t size,
                   bool approximate = false);
  static void elu(const double* input, double* output, size_t size,
                  double alpha = 1.0);
  static void swish(const double* input, double* output, size_t size);
  static void softmax(const double* input, double* output, size_t size);

  // Utility functions
  static void synchronize();
  static int getDeviceCount();
  static void setDevice(int device);
  static std::string getDeviceName(int device);

private:
#ifdef __OBJC__
  static id<MTLDevice> device_;
  static id<MTLCommandQueue> command_queue_;
  static id<MTLLibrary> library_;
#else
  static void* device_;
  static void* command_queue_;
  static void* library_;
#endif
  static bool initialized_;

  static void initializeKernels();
};

}  // namespace Backend
}  // namespace MLLib

#endif  // WITH_METAL
