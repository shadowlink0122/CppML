/**
 * @file gpu_vendor_detection.cpp
 * @brief GPU vendor detection example for CppML
 *
 * This example demonstrates multi-vendor GPU detection and basic operations
 * across different GPU vendors (NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple
 * Metal).
 */

#include "MLLib.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

using namespace MLLib;

void printHeader(const std::string& title) {
  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << " " << title << std::endl;
  std::cout << std::string(50, '=') << std::endl;
}

void printGPUInfo() {
  printHeader("GPU VENDOR DETECTION");

  // Check overall GPU availability
  bool gpu_available = Device::isGPUAvailable();
  std::cout << "GPU Available: " << (gpu_available ? "Yes" : "No") << std::endl;

  if (!gpu_available) {
    std::cout << "No GPU detected. Will demonstrate CPU fallback behavior."
              << std::endl;
    return;
  }

  // Detect all available GPUs
  auto gpus = Device::detectGPUs();
  std::cout << "Total GPUs detected: " << gpus.size() << std::endl;

  std::cout << "\nDetailed GPU Information:" << std::endl;
  for (size_t i = 0; i < gpus.size(); i++) {
    std::cout << "  GPU " << i << ": " << gpus[i].name << std::endl;
  }
}

void testGPUVendors() {
  printHeader("GPU VENDOR TESTING");

    // Test individual GPU vendors
  std::vector<std::pair<std::string, GPUVendor>> vendors = {
      {"NVIDIA CUDA", GPUVendor::NVIDIA},
      {"AMD ROCm", GPUVendor::AMD},
      {"Intel oneAPI", GPUVendor::INTEL_GPU},
      {"Apple Metal", GPUVendor::APPLE}
  };

  for (const auto& vendor : vendors) {
    bool available = Device::isGPUVendorAvailable(vendor.second);
    std::cout << std::left << std::setw(15) << vendor.first << ": "
              << (available ? "Available" : "Not Available") << std::endl;
  }
}

void testBasicGPUOperations() {
  printHeader("BASIC GPU OPERATIONS TEST");

  // Test GPU device creation and operations
  std::cout << "Testing GPU device configuration..." << std::endl;

  // Attempt to set GPU device
  try {
    Device::setDevice(DeviceType::GPU);
    std::cout << "✅ GPU device successfully configured" << std::endl;

    // Get current device info
    DeviceType current = Device::getCurrentDevice();
    std::cout << "Current device: "
              << (current == DeviceType::GPU ? "GPU" : "CPU") << std::endl;

    // Test basic array operations on GPU
    std::cout << "\nTesting basic GPU array operations..." << std::endl;
    NDArray gpu_array({4, 4});
    std::cout << "✅ GPU array created successfully" << std::endl;

    // Simple computation
    gpu_array = gpu_array + 1.0;
    std::cout << "✅ GPU arithmetic operation completed" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "⚠️  GPU device not available, using CPU fallback: " << e.what()
              << std::endl;

    // Test CPU fallback
    DeviceType current = Device::getCurrentDevice();
    std::cout << "Current device: "
              << (current == DeviceType::GPU ? "GPU" : "CPU") << std::endl;
  }
}

void testModelOperations() {
  printHeader("MODEL OPERATIONS TEST");

  try {
    std::cout << "Creating a simple neural network model..." << std::endl;

    // Create a simple model
    MLLib::model::Sequential model;
    model.add(std::make_shared<MLLib::layer::Dense>(784, 128));
    model.add(std::make_shared<MLLib::layer::Dense>(128, 10));

    std::cout << "✅ Model created successfully" << std::endl;

    // Test forward pass
    std::cout << "Testing forward pass..." << std::endl;
    NDArray input({1, 784});

    // Fill with test data
    for (int i = 0; i < 784; i++) {
      input.at({0, static_cast<size_t>(i)}) = static_cast<double>(i % 100) / 100.0;
    }

    auto output = model.predict(input);
    std::cout << "✅ Forward pass completed" << std::endl;
    std::cout << "Output shape: [" << output.shape()[0] << ", "
              << output.shape()[1] << "]" << std::endl;

    // Display device info
    DeviceType device = Device::getCurrentDevice();
    std::cout << "Computation performed on: "
              << (device == DeviceType::GPU ? "GPU" : "CPU") << std::endl;

  } catch (const std::exception& e) {
    std::cout << "❌ Error during model testing: " << e.what() << std::endl;
  }
}

void printSummary() {
  printHeader("SUMMARY");

  std::cout << "GPU Vendor Detection Demo completed!" << std::endl;
  std::cout << "\nKey findings:" << std::endl;

  bool gpu_available = Device::isGPUAvailable();
  if (gpu_available) {
    auto gpus = Device::detectGPUs();
    std::cout << "✅ " << gpus.size() << " GPU(s) detected and available"
              << std::endl;

    // Show which vendors are available
    std::vector<std::pair<std::string, GPUVendor>> vendors = {
        {"NVIDIA", GPUVendor::NVIDIA},
        {"AMD", GPUVendor::AMD},
        {"Intel", GPUVendor::INTEL_GPU},
        {"Apple", GPUVendor::APPLE}};

    for (const auto& vendor : vendors) {
      if (Device::isGPUVendorAvailable(vendor.second)) {
        std::cout << "✅ " << vendor.first << " GPU support active"
                  << std::endl;
      }
    }

    std::cout << "🚀 GPU acceleration ready for machine learning workloads"
              << std::endl;
  } else {
    std::cout << "ℹ️  No GPU detected - CPU-only mode active" << std::endl;
    std::cout << "💡 MLLib will use optimized CPU operations" << std::endl;
  }

  std::cout << "\n📖 For more information about GPU support, see:" << std::endl;
  std::cout << "   - docs/MULTI_GPU_SUPPORT_en.md" << std::endl;
  std::cout << "   - docs/GPU_CI_SETUP_en.md" << std::endl;
}

int main() {
  std::cout << "🎯 CppML GPU Vendor Detection Demo" << std::endl;
  std::cout << "====================================" << std::endl;
  std::cout
      << "Testing multi-vendor GPU support across NVIDIA, AMD, Intel, and Apple platforms"
      << std::endl;

  try {
    // Run detection and testing sequence
    printGPUInfo();
    testGPUVendors();
    testBasicGPUOperations();
    testModelOperations();
    printSummary();

    std::cout << "\n🎉 Demo completed successfully!" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cout << "\n❌ Demo failed with error: " << e.what() << std::endl;
    return 1;
  }
}
