#include "MLLib.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace MLLib;

// Enable debug output for GPU detection
#define DEBUG_GPU_DETECTION

int main() {
  std::cout << "=== MLLib GPU Detection Debug Sample ===" << std::endl;
  std::cout << "This sample provides detailed GPU detection debugging."
            << std::endl;

  std::cout << "\n--- System Information ---" << std::endl;

  // Check system architecture
  std::cout << "Architecture Detection:" << std::endl;

#ifdef __APPLE__
  // Check if we're on Apple Silicon
  FILE* pipe = popen("sysctl hw.optional.arm64 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    std::cout << "  " << result;
    if (result.find("1") != std::string::npos) {
      std::cout << "  → Apple Silicon (ARM64) detected" << std::endl;
    } else {
      std::cout << "  → Intel Mac (x86_64) detected" << std::endl;
    }
  }

  // Show system_profiler GPU output
  std::cout << "\nSystem Profiler GPU Output:" << std::endl;
  pipe = popen(
      "system_profiler SPDisplaysDataType | grep -A 5 -B 2 'Chipset Model' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      std::cout << "  " << buffer;
    }
    pclose(pipe);
  }
#endif

  std::cout << "\n--- Detailed GPU Detection Process ---" << std::endl;

  // Show actual detection results with debug info
  printf("🔍 Starting GPU detection with debug information...\n\n");

  auto gpus = Device::detectGPUs();

  std::cout << "\n--- GPU Detection Results ---" << std::endl;
  std::cout << "Detected " << gpus.size() << " GPU(s):" << std::endl;

  for (size_t i = 0; i < gpus.size(); ++i) {
    const auto& gpu = gpus[i];
    std::cout << "  GPU " << (i + 1) << ":" << std::endl;
    std::cout << "    Vendor: ";
    switch (gpu.vendor) {
    case GPUVendor::NVIDIA: std::cout << "NVIDIA"; break;
    case GPUVendor::AMD: std::cout << "AMD"; break;
    case GPUVendor::INTEL_GPU: std::cout << "Intel"; break;
    case GPUVendor::APPLE: std::cout << "Apple"; break;
    default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    std::cout << "    Name: " << gpu.name << std::endl;
    std::cout << "    Memory: " << gpu.memory_mb << " MB" << std::endl;
    std::cout << "    Compute Capable: " << (gpu.compute_capable ? "Yes" : "No")
              << std::endl;
    std::cout << "    API Support: " << gpu.api_support << std::endl;
  }

  std::cout << "\n--- Primary GPU Vendor ---" << std::endl;
  auto primary_vendor = Device::getPrimaryGPUVendor();
  std::cout << "Primary GPU vendor: ";
  switch (primary_vendor) {
  case GPUVendor::NVIDIA: std::cout << "NVIDIA"; break;
  case GPUVendor::AMD: std::cout << "AMD"; break;
  case GPUVendor::INTEL_GPU: std::cout << "Intel"; break;
  case GPUVendor::APPLE: std::cout << "Apple"; break;
  default: std::cout << "Unknown"; break;
  }
  std::cout << std::endl;

  std::cout << "\n--- Individual Vendor Checks ---" << std::endl;
  std::cout << "NVIDIA GPU: "
            << (Device::isGPUVendorAvailable(GPUVendor::NVIDIA)
                    ? "Available"
                    : "Not Available")
            << std::endl;
  std::cout << "AMD GPU: "
            << (Device::isGPUVendorAvailable(GPUVendor::AMD) ? "Available"
                                                             : "Not Available")
            << std::endl;
  std::cout << "Intel GPU: "
            << (Device::isGPUVendorAvailable(GPUVendor::INTEL_GPU)
                    ? "Available"
                    : "Not Available")
            << std::endl;
  std::cout << "Apple GPU: "
            << (Device::isGPUVendorAvailable(GPUVendor::APPLE)
                    ? "Available"
                    : "Not Available")
            << std::endl;

  std::cout << "\n--- General GPU Availability ---" << std::endl;
  std::cout << "GPU Available: " << (Device::isGPUAvailable() ? "Yes" : "No")
            << std::endl;

  std::cout << "\n--- Test GPU Device Configuration ---" << std::endl;
  std::cout << "Current Device: "
            << Device::getDeviceTypeString(Device::getCurrentDevice())
            << std::endl;

  std::cout << "Testing GPU device validation..." << std::endl;
  bool gpu_set = Device::setDeviceWithValidation(DeviceType::GPU, true);
  std::cout << "GPU device validation: " << (gpu_set ? "SUCCESS" : "FAILED")
            << std::endl;
  std::cout << "Current Device after GPU set: "
            << Device::getDeviceTypeString(Device::getCurrentDevice())
            << std::endl;

  std::cout << "\n=== Debug GPU Detection Sample Complete ===" << std::endl;

  return 0;
}
