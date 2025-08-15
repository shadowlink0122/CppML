#pragma once

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
  GPU,  ///< GPU computation (future implementation)
  AUTO  ///< Automatic device selection
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
  static bool isGPUAvailable() {
    // Future implementation
    return false;
  }

private:
  static DeviceType current_device_;
};

}  // namespace MLLib
