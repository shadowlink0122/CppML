/**
 * @file device_detection.cpp
 * @brief Device detection sample program
 * 
 * This sample demonstrates how to:
 * - Detect available GPUs from different vendors
 * - Get primary GPU vendor information  
 * - Check vendor-specific GPU availability
 * - Configure GPU devices for computation
 */

#include <MLLib.hpp>
#include <cstdio>
#include <vector>

int main() {
  printf("=== MLLib Device Detection Sample ===\n");
  printf("This sample demonstrates GPU detection capabilities.\n");

  // Test GPU detection
  printf("\n--- GPU Detection ---\n");
  std::vector<MLLib::GPUInfo> gpus = MLLib::Device::detectGPUs();

  if (gpus.empty()) {
    printf("No GPUs detected\n");
  } else {
    printf("Detected %zu GPU(s):\n", gpus.size());
    for (size_t i = 0; i < gpus.size(); i++) {
      const auto& gpu = gpus[i];

      const char* vendor_name = gpu.vendor == MLLib::GPUVendor::NVIDIA
          ? "NVIDIA"
          : gpu.vendor == MLLib::GPUVendor::AMD       ? "AMD"
          : gpu.vendor == MLLib::GPUVendor::INTEL_GPU ? "Intel"
          : gpu.vendor == MLLib::GPUVendor::APPLE     ? "Apple"
                                                      : "Unknown";

      printf("  GPU %zu:\n", i + 1);
      printf("    Vendor: %s\n", vendor_name);
      printf("    Name: %s\n", gpu.name.c_str());
      printf("    Memory: %zu MB\n", gpu.memory_mb);
      printf("    Compute Capable: %s\n", gpu.compute_capable ? "Yes" : "No");
      printf("    API Support: %s\n", gpu.api_support.c_str());
      printf("\n");
    }
  }

  // Test primary GPU vendor
  printf("--- Primary GPU Vendor ---\n");
  MLLib::GPUVendor primary = MLLib::Device::getPrimaryGPUVendor();
  const char* primary_name = primary == MLLib::GPUVendor::NVIDIA ? "NVIDIA"
      : primary == MLLib::GPUVendor::AMD                         ? "AMD"
      : primary == MLLib::GPUVendor::INTEL_GPU                   ? "Intel"
      : primary == MLLib::GPUVendor::APPLE                       ? "Apple"
                                           : "Unknown/None";
  printf("Primary GPU vendor: %s\n", primary_name);

  // Test vendor availability
  printf("\n--- Vendor Availability Check ---\n");
  MLLib::GPUVendor vendors[] = {MLLib::GPUVendor::NVIDIA, MLLib::GPUVendor::AMD,
                                MLLib::GPUVendor::INTEL_GPU,
                                MLLib::GPUVendor::APPLE};
  const char* vendor_names[] = {"NVIDIA", "AMD", "Intel", "Apple"};

  for (size_t i = 0; i < 4; i++) {
    bool available = MLLib::Device::isGPUVendorAvailable(vendors[i]);
    printf("%s GPU: %s\n", vendor_names[i],
           available ? "Available" : "Not Available");
  }

  // Test general GPU availability
  printf("\n--- General GPU Availability ---\n");
  bool gpu_available = MLLib::Device::isGPUAvailable();
  printf("GPU Available: %s\n", gpu_available ? "Yes" : "No");

  // Test device configuration
  printf("\n--- Device Configuration ---\n");
  printf("Current Device: %s\n",
         MLLib::Device::getDeviceTypeString(MLLib::Device::getCurrentDevice()));

  if (gpu_available) {
    printf("Testing GPU device validation...\n");
    bool success =
        MLLib::Device::setDeviceWithValidation(MLLib::DeviceType::GPU, true);
    printf("GPU device validation: %s\n", success ? "SUCCESS" : "FAILED");
    printf(
        "Current Device after GPU set: %s\n",
        MLLib::Device::getDeviceTypeString(MLLib::Device::getCurrentDevice()));
  }

  printf("\n=== Device Detection Sample Complete ===\n");
  return 0;
}
