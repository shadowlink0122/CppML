#pragma once

/**
 * @file device.hpp
 * @brief Device type definitions and management
 */

#include <string>
#include <vector>

namespace MLLib {

/**
 * @enum DeviceType
 * @brief Supported device types for computation
 */
enum class DeviceType {
  CPU,  ///< CPU computation
  GPU,  ///< GPU computation (future implementation)
  AUTO  ///< Automatic device selection
};

/**
 * @enum GPUVendor
 * @brief Supported GPU vendors
 */
enum class GPUVendor {
  UNKNOWN,    ///< Unknown or unsupported vendor
  NVIDIA,     ///< NVIDIA GPUs (CUDA)
  AMD,        ///< AMD GPUs (ROCm/OpenCL)
  INTEL_GPU,  ///< Intel GPUs (oneAPI/OpenCL)
  APPLE       ///< Apple GPUs (Metal)
};

/**
 * @struct GPUInfo
 * @brief Information about a detected GPU
 */
struct GPUInfo {
  GPUVendor vendor;         ///< GPU vendor
  std::string name;         ///< GPU name/model
  size_t memory_mb;         ///< GPU memory in MB
  bool compute_capable;     ///< Whether GPU supports compute operations
  std::string api_support;  ///< Supported APIs (CUDA, ROCm, Metal, etc.)
};

/**
 * @class Device
 * @brief Device management class
 */
class Device {
public:
  /**
   * @brief Get current device type
   * @return Current device type
   */
  static DeviceType getCurrentDevice() { return current_device_; }

  /**
   * @brief Set current device type
   * @param device Device type to set
   */
  static void setDevice(DeviceType device) { current_device_ = device; }

  /**
   * @brief Check if GPU is available
   * @return true if GPU is available, false otherwise
   */
  static bool isGPUAvailable();

  /**
   * @brief Set current device type with validation
   * @param device Device type to set
   * @param show_warnings Whether to show warning messages
   * @return true if device was set successfully, false if fallback occurred
   */
  static bool setDeviceWithValidation(DeviceType device,
                                      bool show_warnings = true);

  /**
   * @brief Get device type as string
   * @param device Device type
   * @return Device type as string
   */
  static const char* getDeviceTypeString(DeviceType device);

  /**
   * @brief Detect all available GPUs
   * @return Vector of detected GPU information
   */
  static std::vector<GPUInfo> detectGPUs();

  /**
   * @brief Get primary GPU vendor (highest priority available)
   * @return Primary GPU vendor
   */
  static GPUVendor getPrimaryGPUVendor();

  /**
   * @brief Check if specific GPU vendor is available
   * @param vendor GPU vendor to check
   * @return true if vendor's GPU is available
   */
  static bool isGPUVendorAvailable(GPUVendor vendor);

private:
  static DeviceType current_device_;
};

}  // namespace MLLib
