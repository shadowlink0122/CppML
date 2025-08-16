#include "../../../include/MLLib/backend/backend.hpp"
#include "../../../include/MLLib/device/device.hpp"
#include "backend_internal.hpp"
#include <stdexcept>
#include <vector>

namespace MLLib {
namespace Backend {

void Backend::matmul(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation([&]() { cpu_matmul(a, b, result); },
                             [&]() { gpu_matmul(a, b, result); });
}

void Backend::add(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation([&]() { cpu_add(a, b, result); },
                             [&]() { gpu_add(a, b, result); });
}

void Backend::subtract(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation([&]() { cpu_subtract(a, b, result); },
                             [&]() { gpu_subtract(a, b, result); });
}

void Backend::multiply(const NDArray& a, const NDArray& b, NDArray& result) {
  dispatch_backend_operation([&]() { cpu_multiply(a, b, result); },
                             [&]() { gpu_multiply(a, b, result); });
}

void Backend::add_scalar(const NDArray& a, double scalar, NDArray& result) {
  dispatch_backend_operation([&]() { cpu_add_scalar(a, scalar, result); },
                             [&]() { gpu_add_scalar(a, scalar, result); });
}

void Backend::multiply_scalar(const NDArray& a, double scalar,
                              NDArray& result) {
  dispatch_backend_operation([&]() { cpu_multiply_scalar(a, scalar, result); },
                             [&]() { gpu_multiply_scalar(a, scalar, result); });
}

void Backend::fill(NDArray& array, double value) {
  dispatch_backend_operation([&]() { cpu_fill(array, value); },
                             [&]() { gpu_fill(array, value); });
}

void Backend::copy(const NDArray& src, NDArray& dst) {
  dispatch_backend_operation([&]() { cpu_copy(src, dst); },
                             [&]() { gpu_copy(src, dst); });
}

GPUBackendType Backend::getCurrentGPUBackend() {
  // TODO: Implement proper GPU backend detection
  // This should query the actual GPU backend in use
  return GPUBackendType::NONE;
}

std::vector<GPUBackendType> Backend::getAvailableGPUBackends() {
  // TODO: Implement proper GPU backend enumeration
  // This should detect available GPU backends at runtime
  std::vector<GPUBackendType> available_backends;

#ifdef WITH_CUDA
  available_backends.push_back(GPUBackendType::CUDA);
#endif

#ifdef WITH_ROCM
  available_backends.push_back(GPUBackendType::ROCM);
#endif

#ifdef WITH_OPENCL
  available_backends.push_back(GPUBackendType::OPENCL);
#endif

#ifdef WITH_METAL
  available_backends.push_back(GPUBackendType::METAL);
#endif

#ifdef WITH_ONEAPI
  available_backends.push_back(GPUBackendType::ONEAPI);
#endif

  return available_backends;
}

bool Backend::setPreferredGPUBackend(GPUBackendType backend) {
  // TODO: Implement proper GPU backend selection
  // This should validate and set the preferred GPU backend
  (void)backend;  // Suppress unused parameter warning

  // For now, just return false to indicate unimplemented
  return false;
}

}  // namespace Backend
}  // namespace MLLib
