#include "../../../include/MLLib/backend/backend.hpp"
#include "../../../include/MLLib/device/device.hpp"
#include <vector>

namespace MLLib {
namespace Backend {

void Backend::matmul(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_matmul(a, b, result); break;
  case DeviceType::GPU: gpu_matmul(a, b, result); break;
  default: cpu_matmul(a, b, result); break;
  }
}

void Backend::add(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_add(a, b, result); break;
  case DeviceType::GPU: gpu_add(a, b, result); break;
  default: cpu_add(a, b, result); break;
  }
}

void Backend::subtract(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_subtract(a, b, result); break;
  case DeviceType::GPU: gpu_subtract(a, b, result); break;
  default: cpu_subtract(a, b, result); break;
  }
}

void Backend::multiply(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_multiply(a, b, result); break;
  case DeviceType::GPU: gpu_multiply(a, b, result); break;
  default: cpu_multiply(a, b, result); break;
  }
}

void Backend::add_scalar(const NDArray& a, double scalar, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_add_scalar(a, scalar, result); break;
  case DeviceType::GPU: gpu_add_scalar(a, scalar, result); break;
  default: cpu_add_scalar(a, scalar, result); break;
  }
}

void Backend::multiply_scalar(const NDArray& a, double scalar,
                              NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_multiply_scalar(a, scalar, result); break;
  case DeviceType::GPU: gpu_multiply_scalar(a, scalar, result); break;
  default: cpu_multiply_scalar(a, scalar, result); break;
  }
}

void Backend::fill(NDArray& array, double value) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_fill(array, value); break;
  case DeviceType::GPU: gpu_fill(array, value); break;
  default: cpu_fill(array, value); break;
  }
}

void Backend::copy(const NDArray& src, NDArray& dst) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_copy(src, dst); break;
  case DeviceType::GPU: gpu_copy(src, dst); break;
  default: cpu_copy(src, dst); break;
  }
}

GPUBackendType Backend::getCurrentGPUBackend() {
  // Stub implementation - return NONE for now
  return GPUBackendType::NONE;
}

std::vector<GPUBackendType> Backend::getAvailableGPUBackends() {
  // Stub implementation - return empty vector for now
  return std::vector<GPUBackendType>();
}

bool Backend::setPreferredGPUBackend(GPUBackendType backend) {
  // Stub implementation - return false for now
  (void)backend;  // Suppress unused parameter warning
  return false;
}

}  // namespace Backend
}  // namespace MLLib
