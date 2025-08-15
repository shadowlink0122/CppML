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

private:
  static DeviceType current_device_;
};

}  // namespace MLLib
