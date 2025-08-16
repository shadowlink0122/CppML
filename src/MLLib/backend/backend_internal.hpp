#pragma once

/**
 * @file backend_internal.hpp
 * @brief Internal backend utilities and common definitions
 */

#include "../../../include/MLLib/backend/backend.hpp"
#include <functional>
#include <stdexcept>

namespace MLLib {
namespace Backend {

/**
 * @brief Helper function to dispatch operations to appropriate backend with
 * error handling
 * @param cpu_func Function to execute on CPU backend
 * @param gpu_func Function to execute on GPU backend
 */
template <typename CpuFunc, typename GpuFunc>
void dispatch_backend_operation(CpuFunc&& cpu_func, GpuFunc&& gpu_func) {
  try {
    switch (Device::getCurrentDevice()) {
    case DeviceType::CPU: cpu_func(); break;
    case DeviceType::GPU: gpu_func(); break;
    default: cpu_func(); break;
    }
  } catch (const std::exception& e) {
    // If GPU operation fails, fallback to CPU
    if (Device::getCurrentDevice() == DeviceType::GPU) {
      cpu_func();
    } else {
      throw;  // Re-throw if CPU operation fails
    }
  }
}

}  // namespace Backend
}  // namespace MLLib
