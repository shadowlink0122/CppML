#include "../../../include/MLLib/device/device.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include "../backend/gpu/cuda_kernels.hpp"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

namespace MLLib {

// Forward declarations for helper functions
namespace {
bool checkAMDGPU();
std::string detectAMDGPUName();
bool checkIntelGPU();
std::string detectIntelGPUName();
#ifdef __APPLE__
bool checkAppleGPU();
std::string detectAppleGPUName();
#endif
}  // namespace

// Static member definition
DeviceType Device::current_device_ = DeviceType::CPU;

bool Device::isGPUAvailable() {
  // Check for forced CPU-only mode first
  const char* force_cpu = std::getenv("FORCE_CPU_ONLY");
  if (force_cpu && std::string(force_cpu) == "1") {
    return false;  // Force CPU-only mode for testing
  }

  // Check for GPU simulation mode
  const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
  if (sim_mode && std::string(sim_mode) == "1") {
    return true;  // Force GPU availability in simulation mode
  }

#ifdef WITH_CUDA
  try {
    if (Backend::cuda::cuda_is_available()) {
      return true;
    }
  } catch (const std::exception& e) {
    // CUDA failed, continue checking other vendors
  }
#endif

  // Check for other GPU vendors
  auto gpus = detectGPUs();
  return !gpus.empty();
}

std::vector<GPUInfo> Device::detectGPUs() {
  std::vector<GPUInfo> gpus;

  // 1. Check NVIDIA GPUs (CUDA)
#ifdef WITH_CUDA
  try {
    if (Backend::cuda::cuda_is_available()) {
      GPUInfo nvidia_gpu;
      nvidia_gpu.vendor = GPUVendor::NVIDIA;
      nvidia_gpu.name = "NVIDIA CUDA GPU";
      nvidia_gpu.memory_mb = 0;  // Will be populated by CUDA calls
      nvidia_gpu.compute_capable = true;
      nvidia_gpu.api_support = "CUDA";
      gpus.push_back(nvidia_gpu);
    }
  } catch (const std::exception& e) {
    // CUDA not available
  }
#endif

  // 2. Check AMD GPUs (ROCm/OpenCL)
  if (checkAMDGPU()) {
    GPUInfo amd_gpu;
    amd_gpu.vendor = GPUVendor::AMD;
    amd_gpu.name = detectAMDGPUName();
    amd_gpu.memory_mb = 0;
    amd_gpu.compute_capable = true;
    amd_gpu.api_support = "ROCm/OpenCL";
    gpus.push_back(amd_gpu);
  }

  // 3. Check Intel GPUs (oneAPI/OpenCL)
  if (checkIntelGPU()) {
    GPUInfo intel_gpu;
    intel_gpu.vendor = GPUVendor::INTEL_GPU;
    intel_gpu.name = detectIntelGPUName();
    intel_gpu.memory_mb = 0;
    intel_gpu.compute_capable = true;
    intel_gpu.api_support = "oneAPI/OpenCL";
    gpus.push_back(intel_gpu);
  }

  // 4. Check Apple Silicon GPUs (Metal)
#ifdef __APPLE__
  if (checkAppleGPU()) {
    GPUInfo apple_gpu;
    apple_gpu.vendor = GPUVendor::APPLE;
    apple_gpu.name = detectAppleGPUName();
    apple_gpu.memory_mb = 0;
    apple_gpu.compute_capable = true;
    apple_gpu.api_support = "Metal";
    gpus.push_back(apple_gpu);
  }
#endif

  return gpus;
}

GPUVendor Device::getPrimaryGPUVendor() {
  auto gpus = detectGPUs();
  if (gpus.empty()) {
    return GPUVendor::UNKNOWN;
  }

  // Priority order: NVIDIA > AMD > Intel > Apple
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::NVIDIA) return GPUVendor::NVIDIA;
  }
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::AMD) return GPUVendor::AMD;
  }
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::INTEL_GPU) return GPUVendor::INTEL_GPU;
  }
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::APPLE) return GPUVendor::APPLE;
  }

  return gpus[0].vendor;
}

bool Device::isGPUVendorAvailable(GPUVendor vendor) {
  auto gpus = detectGPUs();
  for (const auto& gpu : gpus) {
    if (gpu.vendor == vendor) {
      return true;
    }
  }
  return false;
}

bool Device::setDeviceWithValidation(DeviceType device, bool show_warnings) {
  if (device == DeviceType::GPU) {
    if (!isGPUAvailable()) {
      if (show_warnings) {
        printf("⚠️  WARNING: GPU device requested but no GPU found!\n");
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
      } else if (show_warnings) {
        printf("✅ GPU device successfully configured\n");
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

// Helper functions for GPU vendor detection
namespace {

bool checkAMDGPU() {
#ifdef __linux__
  // Check for AMD GPU via lspci or /sys
  std::ifstream lspci_proc("/proc/bus/pci/devices");
  if (lspci_proc.is_open()) {
    std::string line;
    while (std::getline(lspci_proc, line)) {
      if (line.find("1002") != std::string::npos) {  // AMD vendor ID
        return true;
      }
    }
  }

  // Check ROCm installation
  std::ifstream rocm_file("/opt/rocm/bin/rocm-smi");
  if (rocm_file.good()) {
    return true;
  }
#endif

#ifdef _WIN32
  // Windows: Check via WMI or device manager
  // TODO: Implement Windows AMD GPU detection
#endif

  return false;
}

std::string detectAMDGPUName() {
#ifdef __linux__
  // Try to get GPU name from ROCm
  FILE* pipe = popen("rocm-smi --showid 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.find("GPU") != std::string::npos) {
      return "AMD ROCm GPU";
    }
  }

  // Fallback to generic name
  return "AMD GPU";
#endif
  return "AMD GPU";
}

bool checkIntelGPU() {
#ifdef __linux__
  // Check for Intel GPU via /sys
  std::ifstream lspci_proc("/proc/bus/pci/devices");
  if (lspci_proc.is_open()) {
    std::string line;
    while (std::getline(lspci_proc, line)) {
      if (line.find("8086") != std::string::npos) {  // Intel vendor ID
        return true;
      }
    }
  }

  // Check for Intel oneAPI/OpenCL
  std::ifstream oneapi_file(
      "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icc");
  if (oneapi_file.good()) {
    return true;
  }
#endif

#ifdef _WIN32
  // Windows: Check for Intel GPU
  // TODO: Implement Windows Intel GPU detection
#endif

  return false;
}

std::string detectIntelGPUName() {
#ifdef __linux__
  // Try to detect Intel GPU model
  FILE* pipe = popen("lspci | grep -i intel | grep -i vga 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      return "Intel GPU";
    }
  }
#endif
  return "Intel GPU";
}

#ifdef __APPLE__
bool checkAppleGPU() {
  // Check for Apple Silicon
  size_t size = 0;
  if (sysctlbyname("hw.optional.arm64", nullptr, &size, nullptr, 0) == 0) {
    uint32_t val = 0;
    if (sysctlbyname("hw.optional.arm64", &val, &size, nullptr, 0) == 0) {
      return val == 1;  // Apple Silicon has integrated GPU
    }
  }

  // Check for discrete AMD/NVIDIA GPUs on Intel Macs
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      return true;
    }
  }

  return false;
}

std::string detectAppleGPUName() {
  // Try to get specific Apple GPU info
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' | head -1 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Extract GPU name from system_profiler output
      size_t pos = result.find(":");
      if (pos != std::string::npos) {
        std::string gpu_name = result.substr(pos + 1);
        // Remove whitespace and newlines
        gpu_name.erase(0, gpu_name.find_first_not_of(" \t\n\r"));
        gpu_name.erase(gpu_name.find_last_not_of(" \t\n\r") + 1);
        return gpu_name;
      }
    }
  }

  return "Apple GPU";
}
#endif

}  // anonymous namespace

}  // namespace MLLib
