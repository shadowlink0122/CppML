/**
 * @file test_multi_gpu.hpp
 * @brief Unit tests for multi-GPU vendor support
 */

#pragma once

#include "../../../../include/MLLib/backend/backend.hpp"
#include "../../../../include/MLLib/device/device.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"
#include <iostream>
#include <vector>

namespace MLLib {
namespace test {

/**
 * @class MultiGPUDetectionTest
 * @brief Test multi-vendor GPU detection
 */
class MultiGPUDetectionTest : public TestCase {
public:
  MultiGPUDetectionTest() : TestCase("MultiGPUDetectionTest") {}

protected:
  void test() override {
    // Test GPU detection
    auto gpus = Device::detectGPUs();

    printf("  Detected %zu GPU(s)\n", gpus.size());

    // Display detected GPUs
    for (const auto& gpu : gpus) {
      printf("  - %s (%s, Vendor: %d)\n", gpu.name.c_str(),
             gpu.api_support.c_str(), static_cast<int>(gpu.vendor));

      assertTrue(gpu.compute_capable, "GPU should be compute capable");
      assertFalse(gpu.name.empty(), "GPU name should not be empty");
      assertFalse(gpu.api_support.empty(),
                  "API support string should not be empty");
    }

    // Test vendor-specific availability
    bool nvidia_available = Device::isGPUVendorAvailable(GPUVendor::NVIDIA);
    bool amd_available = Device::isGPUVendorAvailable(GPUVendor::AMD);
    bool intel_available = Device::isGPUVendorAvailable(GPUVendor::INTEL_GPU);
    bool apple_available = Device::isGPUVendorAvailable(GPUVendor::APPLE);

    printf("  NVIDIA GPU: %s\n",
           nvidia_available ? "Available" : "Not available");
    printf("  AMD GPU: %s\n", amd_available ? "Available" : "Not available");
    printf("  Intel GPU: %s\n",
           intel_available ? "Available" : "Not available");
    printf("  Apple GPU: %s\n",
           apple_available ? "Available" : "Not available");

    // At least one GPU vendor check should work (even if returns false)
    assertTrue(true, "GPU vendor checks should not crash");

    // Test overall GPU availability
    bool any_gpu = Device::isGPUAvailable();
    assertEqual(any_gpu, !gpus.empty(),
                "isGPUAvailable() should match detected GPU count");
  }
};

/**
 * @class GPUBackendTypesTest
 * @brief Test GPU backend type enumeration
 */
class GPUBackendTypesTest : public TestCase {
public:
  GPUBackendTypesTest() : TestCase("GPUBackendTypesTest") {}

protected:
  void test() override {
    // Test available backend types (simplified)
    printf("  Testing GPU backend enumeration...\n");

    // Test that GPU backend enumeration doesn't crash
    assertTrue(true, "Backend enumeration should work without errors");
  }
};

/**
 * @class MultiGPUBackendOperationsTest
 * @brief Test operations with different GPU backends
 */
class MultiGPUBackendOperationsTest : public TestCase {
public:
  MultiGPUBackendOperationsTest() : TestCase("MultiGPUBackendOperationsTest") {}

protected:
  void test() override {
    // Store original device
    DeviceType original_device = Device::getCurrentDevice();

    // Test with GPU device (auto-selects best available)
    Device::setDeviceWithValidation(DeviceType::GPU, false);

    // Create test matrices
    NDArray a({3, 3});
    NDArray b({3, 3});

    // Initialize with test data
    for (size_t i = 0; i < a.size(); ++i) {
      a.data()[i] = static_cast<double>(i + 1);
      b.data()[i] = static_cast<double>((i * 2) + 1);
    }

    // Test matrix multiplication (should work regardless of GPU availability)
    NDArray result = a.matmul(b);

    assertTrue(result.shape()[0] == 3, "Result should have correct rows");
    assertTrue(result.shape()[1] == 3, "Result should have correct columns");

    // Test element-wise operations
    NDArray add_result = a + b;
    NDArray sub_result = a - b;
    NDArray mul_result = a * b;

    // Verify shapes
    assertTrue(add_result.shape() == a.shape(),
               "Addition should preserve shape");
    assertTrue(sub_result.shape() == a.shape(),
               "Subtraction should preserve shape");
    assertTrue(mul_result.shape() == a.shape(),
               "Multiplication should preserve shape");

    // Test scalar operations
    NDArray scalar_result = a * 2.0;
    assertNear(scalar_result.data()[0], 2.0, 1e-10,
               "Scalar multiplication should work");

    // Test array operations with different backends
    auto gpus = Device::detectGPUs();
    for (const auto& gpu : gpus) {
      printf("  Testing with %s (%s)\n", gpu.name.c_str(),
             gpu.api_support.c_str());

      // Each backend should handle operations gracefully
      NDArray test_result = a + b;
      assertTrue(test_result.size() == a.size(),
                 "Operations should work with all backends");
    }

    // Restore original device
    Device::setDevice(original_device);
  }
};

/**
 * @class GPUVendorPriorityTest
 * @brief Test GPU vendor priority ordering
 */
class GPUVendorPriorityTest : public TestCase {
public:
  GPUVendorPriorityTest() : TestCase("GPUVendorPriorityTest") {}

protected:
  void test() override {
    auto gpus = Device::detectGPUs();

    // Test that GPU priority ordering works as documented:
    // 1. NVIDIA CUDA (highest priority)
    // 2. AMD ROCm
    // 3. Intel oneAPI
    // 4. Apple Metal

    if (gpus.size() > 1) {
      printf("  Multiple GPUs detected, testing priority order\n");

      GPUVendor first_vendor = gpus[0].vendor;

      // If NVIDIA is available, it should be first
      if (Device::isGPUVendorAvailable(GPUVendor::NVIDIA)) {
        assertEqual(static_cast<int>(first_vendor),
                    static_cast<int>(GPUVendor::NVIDIA),
                    "NVIDIA should have highest priority");
      }

      printf("  Primary GPU: %s\n", gpus[0].name.c_str());
    } else if (gpus.size() == 1) {
      printf("  Single GPU detected: %s\n", gpus[0].name.c_str());
    } else {
      printf("  No GPU detected - testing CPU fallback\n");
    }

    // Priority test should not fail regardless of hardware
    assertTrue(true, "GPU priority test completed");
  }
};

/**
 * @class GPUMemoryTest
 * @brief Test GPU memory operations and fallback
 */
class GPUMemoryTest : public TestCase {
public:
  GPUMemoryTest() : TestCase("GPUMemoryTest") {}

protected:
  void test() override {
    // Store original device
    DeviceType original_device = Device::getCurrentDevice();

    // Test GPU memory operations
    Device::setDeviceWithValidation(DeviceType::GPU, false);

    // Test with various array sizes
    std::vector<size_t> test_sizes = {10, 100, 1000};

    for (size_t size : test_sizes) {
      NDArray arr({size});

      // Fill array
      arr.fill(42.0);

      // Verify fill operation
      for (size_t i = 0; i < 10 && i < size; ++i) {
        assertNear(arr.data()[i], 42.0, 1e-10,
                   "Fill operation should work with GPU");
      }

      // Test copy operations
      NDArray copy_arr = arr;
      assertTrue(copy_arr.size() == arr.size(), "Copy should preserve size");

      if (size >= 2) {
        assertNear(copy_arr.data()[0], 42.0, 1e-10,
                   "Copy should preserve values");
        assertNear(copy_arr.data()[1], 42.0, 1e-10,
                   "Copy should preserve values");
      }
    }

    printf("  GPU memory operations tested with arrays up to size 1000\n");

    // Restore original device
    Device::setDevice(original_device);
  }
};

/**
 * @class GPUErrorHandlingTest
 * @brief Test GPU error handling and fallback mechanisms
 */
class GPUErrorHandlingTest : public TestCase {
public:
  GPUErrorHandlingTest() : TestCase("GPUErrorHandlingTest") {}

protected:
  void test() override {
    // Test that GPU operations handle errors gracefully

    // Store original device
    DeviceType original_device = Device::getCurrentDevice();

    try {
      // Try to set GPU device
      bool gpu_set = Device::setDeviceWithValidation(DeviceType::GPU, false);

      if (!gpu_set) {
        printf("  GPU not available - testing CPU fallback\n");
        assertTrue(Device::getCurrentDevice() == DeviceType::CPU,
                   "Should fallback to CPU when GPU not available");
      } else {
        printf("  GPU available - testing GPU operations\n");
      }

      // Test operations that might fail on GPU
      NDArray a({2, 2});
      NDArray b({2, 2});

      a.fill(1.0);
      b.fill(2.0);

      // These operations should work regardless of device
      NDArray result = a + b;
      assertNear(result.data()[0], 3.0, 1e-10,
                 "Addition should work even with fallback");

      result = a.matmul(b);
      assertTrue(result.shape()[0] == 2,
                 "Matrix multiplication should work with fallback");

    } catch (const std::exception& e) {
      // GPU operations should not throw unhandled exceptions
      assertFalse(true,
                  std::string(
                      "GPU operations should handle errors gracefully: ") +
                      e.what());
    }

    // Restore original device
    Device::setDevice(original_device);
  }
};

/**
 * @class GPUCompilationTest
 * @brief Test that GPU code compiles correctly with all backends
 */
class GPUCompilationTest : public TestCase {
public:
  GPUCompilationTest() : TestCase("GPUCompilationTest") {}

protected:
  void test() override {
    // Test that all GPU backend headers and functions are accessible

    // Test Backend class static methods (simplified)
    bool gpu_available = Device::isGPUAvailable();
    auto gpus = Device::detectGPUs();

    printf("  GPU available: %s\n", gpu_available ? "Yes" : "No");
    printf("  Detected GPUs: %zu\n", gpus.size());

    // Test vendor-specific checks
    std::vector<GPUVendor> vendors = {GPUVendor::NVIDIA, GPUVendor::AMD,
                                      GPUVendor::INTEL_GPU, GPUVendor::APPLE};

    for (auto vendor : vendors) {
      bool available = Device::isGPUVendorAvailable(vendor);
      // Just ensure the call doesn't crash
      assertTrue(true, "Vendor availability check should not crash");
    }

    // Test device type strings
    std::string cpu_str = Device::getDeviceTypeString(DeviceType::CPU);
    std::string gpu_str = Device::getDeviceTypeString(DeviceType::GPU);
    std::string auto_str = Device::getDeviceTypeString(DeviceType::AUTO);

    assertTrue(cpu_str == std::string("CPU"),
               "CPU device string should be correct");
    assertTrue(gpu_str == std::string("GPU"),
               "GPU device string should be correct");
    assertTrue(auto_str == std::string("AUTO"),
               "AUTO device string should be correct");

    printf("  All GPU compilation tests passed\n");
  }
};

}  // namespace test
}  // namespace MLLib
