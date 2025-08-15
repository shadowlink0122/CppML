/**
 * @file metal_backend.hpp
 * @brief Apple Metal backend implementation for MLLib
 */

#pragma once

#ifdef WITH_METAL

#include "../ndarray.hpp"
#include "backend.hpp"
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

namespace MLLib {
namespace Backend {

/**
 * @class MetalBackend
 * @brief Apple Metal implementation of GPU backend
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
  static id<MTLDevice> device_;
  static id<MTLCommandQueue> command_queue_;
  static id<MTLLibrary> library_;
  static bool initialized_;

  // Metal compute kernels
  static id<MTLComputePipelineState> relu_pipeline_;
  static id<MTLComputePipelineState> sigmoid_pipeline_;
  static id<MTLComputePipelineState> tanh_pipeline_;

  static void initializeKernels();
};

}  // namespace Backend
}  // namespace MLLib

#endif  // WITH_METAL
