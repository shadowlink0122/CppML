#include "../../../include/MLLib/backend/backend.hpp"
#include "../../../include/MLLib/device/device.hpp"
#include "backend_internal.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

// Include GPU backend headers conditionally - they are in gpu/ subfolder
#ifdef WITH_ROCM
#include "../../../include/MLLib/backend/rocm_backend.hpp"
#endif
#ifdef WITH_ONEAPI
#include "../../../include/MLLib/backend/oneapi_backend.hpp"
#endif
// Metal backend header should only be included in .mm files
#ifdef WITH_CUDA
// CUDA functions are declared in backend_internal.hpp
#endif

namespace MLLib {
namespace Backend {

// Static variables for GPU backend management
static GPUBackendType current_gpu_backend_ = GPUBackendType::NONE;
static bool gpu_backend_initialized_ = false;

// GPU backend availability check functions
static bool isROCmAvailable() {
#ifdef WITH_ROCM
// Simple check - ROCm is available if headers are available and we're on
// Linux/compatible system
#ifdef __linux__
  return true;  // Assume ROCm can be available on Linux
#else
  return false;  // ROCm typically not available on non-Linux
#endif
#else
  return false;
#endif
}

static bool isOneAPIAvailable() {
#ifdef WITH_ONEAPI
// Simple check - oneAPI can be available on Intel platforms
#if defined(__x86_64__) || defined(__i386__)
  return true;  // Assume oneAPI can be available on Intel platforms
#else
  return false;  // oneAPI typically not available on non-Intel
#endif
#else
  return false;
#endif
}

static bool isMetalAvailable() {
#ifdef WITH_METAL
#ifdef __APPLE__
  // Check if Apple GPU is actually detected
  if (!Device::isGPUAvailable()) {
    return false;
  }

  auto gpus = Device::detectGPUs();
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::APPLE) {
      return true;
    }
  }
  return false;
#else
  return false;
#endif
#else
  return false;
#endif
}

static bool isCUDAAvailable() {
#ifdef WITH_CUDA
  // Check if actual CUDA hardware and drivers are available
  if (!Device::isGPUAvailable()) {
    return false;
  }

  // Check if any NVIDIA GPU is detected
  auto gpus = Device::detectGPUs();
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::NVIDIA) {
      return true;
    }
  }
  return false;
#else
  return false;
#endif
}

void Backend::matmul(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation(
      [&]() {
        cpu_matmul(a, b, result);
      },
      [&]() {
        gpu_matmul(a, b, result);
      });
}

void Backend::add(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation(
      [&]() {
        cpu_add(a, b, result);
      },
      [&]() {
        gpu_add(a, b, result);
      });
}

void Backend::subtract(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation(
      [&]() {
        cpu_subtract(a, b, result);
      },
      [&]() {
        gpu_subtract(a, b, result);
      });
}

void Backend::multiply(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation(
      [&]() {
        cpu_multiply(a, b, result);
      },
      [&]() {
        gpu_multiply(a, b, result);
      });
}

void Backend::add_scalar(const NDArray& a, double scalar, NDArray& result) {
  dispatch_backend_operation(
      [&]() {
        cpu_add_scalar(a, scalar, result);
      },
      [&]() {
        gpu_add_scalar(a, scalar, result);
      });
}

void Backend::multiply_scalar(const NDArray& a, double scalar,
                              NDArray& result) {
  dispatch_backend_operation(
      [&]() {
        cpu_multiply_scalar(a, scalar, result);
      },
      [&]() {
        gpu_multiply_scalar(a, scalar, result);
      });
}

void Backend::fill(NDArray& array, double value) {
  dispatch_backend_operation(
      [&]() {
        cpu_fill(array, value);
      },
      [&]() {
        gpu_fill(array, value);
      });
}

void Backend::copy(const NDArray& src, NDArray& dst) {
  dispatch_backend_operation(
      [&]() {
        cpu_copy(src, dst);
      },
      [&]() {
        gpu_copy(src, dst);
      });
}

GPUBackendType Backend::getCurrentGPUBackend() {
  if (!gpu_backend_initialized_) {
    // Auto-select best available GPU backend based on platform and priority
#ifdef __APPLE__
    // On Apple platforms, prioritize Metal first
    if (isMetalAvailable()) {
      current_gpu_backend_ = GPUBackendType::METAL;
      printf("MLLib: Selected Metal GPU backend\n");
    } else if (isCUDAAvailable()) {
      current_gpu_backend_ = GPUBackendType::CUDA;
      printf("MLLib: Selected CUDA GPU backend\n");
    } else if (isOneAPIAvailable()) {
      current_gpu_backend_ = GPUBackendType::ONEAPI;
      printf("MLLib: Selected oneAPI GPU backend\n");
    } else if (isROCmAvailable()) {
      current_gpu_backend_ = GPUBackendType::ROCM;
      printf("MLLib: Selected ROCm GPU backend\n");
    } else {
      current_gpu_backend_ = GPUBackendType::NONE;
      printf("MLLib: No GPU backend available, using CPU\n");
    }
#else
    // On other platforms, prioritize CUDA first, then others
    if (isCUDAAvailable()) {
      current_gpu_backend_ = GPUBackendType::CUDA;
      printf("MLLib: Selected CUDA GPU backend\n");
    } else if (isROCmAvailable()) {
      current_gpu_backend_ = GPUBackendType::ROCM;
      printf("MLLib: Selected ROCm GPU backend\n");
    } else if (isOneAPIAvailable()) {
      current_gpu_backend_ = GPUBackendType::ONEAPI;
      printf("MLLib: Selected oneAPI GPU backend\n");
    } else if (isMetalAvailable()) {
      current_gpu_backend_ = GPUBackendType::METAL;
      printf("MLLib: Selected Metal GPU backend\n");
    } else {
      current_gpu_backend_ = GPUBackendType::NONE;
      printf("MLLib: No GPU backend available, using CPU\n");
    }
#endif
    gpu_backend_initialized_ = true;
  }
  return current_gpu_backend_;
}

std::vector<GPUBackendType> Backend::getAvailableGPUBackends() {
  std::vector<GPUBackendType> available_backends;

  // Check each backend's availability at runtime - use local checks to avoid
  // linking issues
#ifdef WITH_CUDA
  if (isCUDAAvailable()) {
    available_backends.push_back(GPUBackendType::CUDA);
  }
#endif

#ifdef WITH_ROCM
  if (isROCmAvailable()) {
    available_backends.push_back(GPUBackendType::ROCM);
  }
#endif

#ifdef WITH_METAL
  if (isMetalAvailable()) {
    available_backends.push_back(GPUBackendType::METAL);
  }
#endif

#ifdef WITH_ONEAPI
  if (isOneAPIAvailable()) {
    available_backends.push_back(GPUBackendType::ONEAPI);
  }
#endif

  // OpenCL support would need additional implementation
#ifdef WITH_OPENCL
  // TODO: Implement OpenCL availability check
  // available_backends.push_back(GPUBackendType::OPENCL);
#endif

  return available_backends;
}

bool Backend::setPreferredGPUBackend(GPUBackendType backend) {
  // Simple validation and setting without dynamic checking
  if (backend == GPUBackendType::NONE) {
    current_gpu_backend_ = backend;
    gpu_backend_initialized_ = true;
    printf("MLLib: Set preferred GPU backend to None\n");
    return true;
  }

  // Check if backend is theoretically supported
  bool is_supported = false;
  const char* backend_name = "Unknown";

  switch (backend) {
  case GPUBackendType::CUDA:
#ifdef WITH_CUDA
    is_supported = true;
    backend_name = "CUDA";
#endif
    break;
  case GPUBackendType::ROCM:
#ifdef WITH_ROCM
    is_supported = true;
    backend_name = "ROCm";
#endif
    break;
  case GPUBackendType::METAL:
#ifdef WITH_METAL
    is_supported = true;
    backend_name = "Metal";
#endif
    break;
  case GPUBackendType::ONEAPI:
#ifdef WITH_ONEAPI
    is_supported = true;
    backend_name = "oneAPI";
#endif
    break;
  case GPUBackendType::OPENCL:
#ifdef WITH_OPENCL
    is_supported = true;
    backend_name = "OpenCL";
#endif
    break;
  default: break;
  }

  if (is_supported) {
    current_gpu_backend_ = backend;
    gpu_backend_initialized_ = true;
    printf("MLLib: Set preferred GPU backend to %s\n", backend_name);
    return true;
  }

  printf("MLLib: Requested GPU backend %s is not supported in this build\n",
         backend_name);
  return false;
}

}  // namespace Backend
}  // namespace MLLib
