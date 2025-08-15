#pragma once

#include <string>
#include <vector>

/**
 * @file device.hpp
 * @brief Device type definitions and management
 */

namespace MLLib {

/**
 * @enum DeviceType
 * @brief Supported device types for computation
 */
enum class DeviceType {
  CPU,  ///< CPU computation
  GPU,  ///< GPU computation (generic)
  AUTO  ///< Automatic device selection
};

/**
 * @enum GPUVendor
 * @brief Supported GPU vendors
 */
enum class GPUVendor {
  UNKNOWN,    ///< Unknown or no GPU
  NVIDIA,     ///< NVIDIA GPU (CUDA)
  AMD,        ///< AMD GPU (ROCm/OpenCL)
  INTEL_GPU,  ///< Intel GPU (oneAPI/OpenCL)
  APPLE       ///< Apple Silicon GPU (Metal)
};

/**
 * @struct GPUInfo
 * @brief GPU information structure
 */
struct GPUInfo {
  GPUVendor vendor;
  std::string name;
  size_t memory_mb;
  bool compute_capable;
  std::string api_support;  // "CUDA", "ROCm", "OpenCL", "Metal", etc.
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
   * @brief Detect available GPUs and their vendors
   * @return Vector of detected GPU information
   */
  static std::vector<GPUInfo> detectGPUs();

  /**
   * @brief Get primary GPU vendor
   * @return Primary GPU vendor (first detected)
   */
  static GPUVendor getPrimaryGPUVendor();

  /**
   * @brief Check if specific GPU vendor is available
   * @param vendor GPU vendor to check
   * @return true if vendor GPU is available
   */
  static bool isGPUVendorAvailable(GPUVendor vendor);

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

private:
  static DeviceType current_device_;
};

}  // namespace MLLib
