#include "../../../include/MLLib/device/device.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef WITH_CUDA
#include "../backend/gpu/cuda_kernels.hpp"
#endif

namespace MLLib {

// Static member definition
DeviceType Device::current_device_ = DeviceType::CPU;

bool Device::isGPUAvailable() {
#ifdef WITH_CUDA
  try {
    return backend::cuda::cuda_is_available();
  } catch (const std::exception& e) {
    return false;
  }
#else
  // Check for GPU simulation mode even without CUDA support
  const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
  if (sim_mode && std::string(sim_mode) == "1") {
    return true;  // Force GPU availability in simulation mode
  }
  return false;
#endif
}

bool Device::setDeviceWithValidation(DeviceType device, bool show_warnings) {
  if (device == DeviceType::GPU) {
    if (!isGPUAvailable()) {
      if (show_warnings) {
        printf(
            "⚠️  WARNING: GPU device requested but no CUDA-capable GPU found!\n");
        printf("   Falling back to CPU device for computation.\n");

#ifndef WITH_CUDA
        // Check if we're in GPU simulation mode
        const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
        if (sim_mode && std::string(sim_mode) == "1") {
          printf("   Note: GPU simulation mode should be active but failed.\n");
          printf("   This may indicate a configuration issue.\n");
        } else {
          printf("   Note: MLLib was compiled without CUDA support.\n");
          printf(
              "   To enable GPU support, install CUDA and rebuild with: make clean && make all\n");
        }
#else
        // Check if we're in GPU simulation mode
        const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
        if (sim_mode && std::string(sim_mode) == "1") {
          printf("   Note: GPU simulation mode should be active but failed.\n");
        } else {
          printf("   Possible causes:\n");
          printf("   - No NVIDIA GPU installed\n");
          printf("   - CUDA driver not installed or incompatible\n");
          printf("   - GPU is being used by another process\n");
        }
#endif
      }
      current_device_ = DeviceType::CPU;
      return false;
    } else {
      // Check if we're in GPU simulation mode and show appropriate message
      const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
      if (sim_mode && std::string(sim_mode) == "1" && show_warnings) {
        printf("✅ GPU simulation mode activated successfully\n");
      }
    }
  }

  current_device_ = device;
  return true;
}

const char* Device::getDeviceTypeString(DeviceType device) {
  switch (device) {
  case DeviceType::CPU: return "CPU";
  case DeviceType::GPU: return "GPU";
  case DeviceType::AUTO: return "AUTO";
  default: return "UNKNOWN";
  }
}

}  // namespace MLLib
