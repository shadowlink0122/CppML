#include "../../../include/MLLib/device/device.hpp"
#include <iostream>

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
    return false;
#endif
}

bool Device::setDeviceWithValidation(DeviceType device, bool show_warnings) {
    if (device == DeviceType::GPU) {
        if (!isGPUAvailable()) {
            if (show_warnings) {
                std::cout << "⚠️  WARNING: GPU device requested but no CUDA-capable GPU found!" << std::endl;
                std::cout << "   Falling back to CPU device for computation." << std::endl;
                
#ifndef WITH_CUDA
                std::cout << "   Note: MLLib was compiled without CUDA support." << std::endl;
                std::cout << "   To enable GPU support, install CUDA and rebuild with: make clean && make all" << std::endl;
#else
                std::cout << "   Possible causes:" << std::endl;
                std::cout << "   - No NVIDIA GPU installed" << std::endl;
                std::cout << "   - CUDA driver not installed or incompatible" << std::endl;
                std::cout << "   - GPU is being used by another process" << std::endl;
#endif
            }
            current_device_ = DeviceType::CPU;
            return false;
        }
    }
    
    current_device_ = device;
    return true;
}

const char* Device::getDeviceTypeString(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:  return "CPU";
        case DeviceType::GPU:  return "GPU";
        case DeviceType::AUTO: return "AUTO";
        default:               return "UNKNOWN";
    }
}

}  // namespace MLLib
