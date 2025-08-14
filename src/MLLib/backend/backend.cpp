#include "../../../include/MLLib/backend/backend.hpp"
#include "../../../include/MLLib/device/device.hpp"

namespace MLLib {
namespace backend {

void Backend::matmul(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_matmul(a, b, result); break;
  case DeviceType::GPU:
    // Future GPU implementation
    cpu_matmul(a, b, result);  // Fallback to CPU for now
    break;
  default: cpu_matmul(a, b, result); break;
  }
}

void Backend::add(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_add(a, b, result); break;
  default: cpu_add(a, b, result); break;
  }
}

void Backend::subtract(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_subtract(a, b, result); break;
  default: cpu_subtract(a, b, result); break;
  }
}

void Backend::multiply(const NDArray& a, const NDArray& b, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_multiply(a, b, result); break;
  default: cpu_multiply(a, b, result); break;
  }
}

void Backend::add_scalar(const NDArray& a, double scalar, NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_add_scalar(a, scalar, result); break;
  default: cpu_add_scalar(a, scalar, result); break;
  }
}

void Backend::multiply_scalar(const NDArray& a, double scalar,
                              NDArray& result) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_multiply_scalar(a, scalar, result); break;
  default: cpu_multiply_scalar(a, scalar, result); break;
  }
}

void Backend::fill(NDArray& array, double value) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_fill(array, value); break;
  default: cpu_fill(array, value); break;
  }
}

void Backend::copy(const NDArray& src, NDArray& dst) {
  switch (Device::getCurrentDevice()) {
  case DeviceType::CPU: cpu_copy(src, dst); break;
  default: cpu_copy(src, dst); break;
  }
}

}  // namespace backend
}  // namespace MLLib
