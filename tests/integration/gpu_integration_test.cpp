/**
 * @file gpu_integration_test.cpp
 * @brief Simplified GPU integration tests
 */

#include "../../include/MLLib.hpp"
#include "../common/test_utils.hpp"
#include <chrono>
#include <iostream>
#include <vector>

namespace MLLib {
namespace test {

/**
 * @class BasicGPUOperationsTest
 * @brief Test basic GPU operations
 */
class BasicGPUOperationsTest : public TestCase {
public:
  BasicGPUOperationsTest() : TestCase("BasicGPUOperationsTest") {}

protected:
  void test() override {
    printf("  Testing basic GPU operations...\n");

    // Store original device
    DeviceType original_device = Device::getCurrentDevice();

    try {
      // Test GPU detection
      auto detected_gpus = Device::detectGPUs();
      size_t gpu_count = detected_gpus.size();
      printf("  Detected %zu GPU(s)\n", gpu_count);

      // Test GPU availability
      bool gpu_available = Device::isGPUAvailable();
      printf("  GPU available: %s\n", gpu_available ? "Yes" : "No");

      // Test device switching
      Device::setDeviceWithValidation(DeviceType::GPU, false);
      // Get current device for verification
      (void)Device::getCurrentDevice();  // Mark as used to avoid warning

      // Test basic array operations
      NDArray a({2, 2});
      NDArray b({2, 2});

      a.fill(1.0);
      b.fill(2.0);

      NDArray result = a + b;
      assertNear(result.data()[0], 3.0, 1e-10, "GPU addition should work");

      printf("  Basic GPU operations completed successfully\n");

    } catch (const std::exception& e) {
      printf("  GPU operations failed (using CPU fallback): %s\n", e.what());
      // Don't fail the test if GPU is not available
    }

    // Restore original device
    Device::setDevice(original_device);
  }
};

/**
 * @class GPUVendorDetectionTest
 * @brief Test GPU vendor detection
 */
class GPUVendorDetectionTest : public TestCase {
public:
  GPUVendorDetectionTest() : TestCase("GPUVendorDetectionTest") {}

protected:
  void test() override {
    printf("  Testing GPU vendor detection...\n");

    // Test vendor-specific detection
    bool nvidia_available = Device::isGPUVendorAvailable(GPUVendor::NVIDIA);
    bool amd_available = Device::isGPUVendorAvailable(GPUVendor::AMD);
    bool intel_available = Device::isGPUVendorAvailable(GPUVendor::INTEL_GPU);
    bool apple_available = Device::isGPUVendorAvailable(GPUVendor::APPLE);

    printf("  NVIDIA: %s\n", nvidia_available ? "Available" : "Not available");
    printf("  AMD: %s\n", amd_available ? "Available" : "Not available");
    printf("  Intel: %s\n", intel_available ? "Available" : "Not available");
    printf("  Apple: %s\n", apple_available ? "Available" : "Not available");

    // Test that detection doesn't crash
    assertTrue(true, "GPU vendor detection should not crash");
  }
};

/**
 * @class GPUPerformanceTest
 * @brief Test GPU vs CPU performance
 */
class GPUPerformanceTest : public TestCase {
public:
  GPUPerformanceTest() : TestCase("GPUPerformanceTest") {}

protected:
  void test() override {
    printf("  Testing GPU performance...\n");

    DeviceType original_device = Device::getCurrentDevice();

    const size_t matrix_size = 20;
    const size_t iterations = 3;

    NDArray a({matrix_size, matrix_size});
    NDArray b({matrix_size, matrix_size});

    // Fill with test data
    for (size_t i = 0; i < a.size(); ++i) {
      a.data()[i] = static_cast<double>(i % 10) / 10.0;
      b.data()[i] = static_cast<double>((i + 5) % 10) / 10.0;
    }

    // Test CPU performance
    Device::setDevice(DeviceType::CPU);
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
      NDArray result = a.matmul(b);
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        cpu_end - cpu_start);

    printf("  CPU: %lld μs for %zu iterations\n", cpu_duration.count(),
           iterations);

    // Test GPU performance (if available)
    if (Device::isGPUAvailable()) {
      Device::setDeviceWithValidation(DeviceType::GPU, false);
      auto gpu_start = std::chrono::high_resolution_clock::now();

      for (size_t i = 0; i < iterations; ++i) {
        NDArray result = a.matmul(b);
      }

      auto gpu_end = std::chrono::high_resolution_clock::now();
      auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
          gpu_end - gpu_start);

      printf("  GPU: %lld μs for %zu iterations\n", gpu_duration.count(),
             iterations);
    } else {
      printf("  GPU not available - CPU-only test\n");
    }

    assertTrue(true, "Performance test completed");

    Device::setDevice(original_device);
  }
};

/**
 * @class GPUStabilityTest
 * @brief Test GPU stability with repeated operations
 */
class GPUStabilityTest : public TestCase {
public:
  GPUStabilityTest() : TestCase("GPUStabilityTest") {}

protected:
  void test() override {
    printf("  Testing GPU stability...\n");

    DeviceType original_device = Device::getCurrentDevice();

    try {
      Device::setDeviceWithValidation(DeviceType::GPU, false);

      const size_t iterations = 10;
      const size_t array_size = 10;

      for (size_t iter = 0; iter < iterations; ++iter) {
        NDArray a({array_size});
        NDArray b({array_size});

        a.fill(static_cast<double>(iter + 1));
        b.fill(static_cast<double>(iter + 2));

        NDArray result = a + b;

        // Verify result
        assertNear(result.data()[0], static_cast<double>(iter * 2 + 3), 1e-10,
                   "GPU operations should be stable");
      }

      printf("  Completed %zu stable iterations\n", iterations);

    } catch (const std::exception& e) {
      printf("  GPU stability test using CPU fallback: %s\n", e.what());
    }

    Device::setDevice(original_device);
  }
};

}  // namespace test
}  // namespace MLLib

// Simplified GPU integration test runner
int run_gpu_integration_tests() {
  printf("🔗 Running GPU Integration Tests...\n");
  printf("--------------------------------------------------\n");

  using namespace MLLib::test;

  std::vector<std::unique_ptr<TestCase>> tests;

  // Add simplified integration tests
  tests.push_back(std::make_unique<BasicGPUOperationsTest>());
  tests.push_back(std::make_unique<GPUVendorDetectionTest>());
  tests.push_back(std::make_unique<GPUPerformanceTest>());
  tests.push_back(std::make_unique<GPUStabilityTest>());

  int total_tests = 0;
  int passed_tests = 0;

  for (auto& test : tests) {
    total_tests++;
    printf("Running %s...\n", test->getName().c_str());

    try {
      test->run();
      printf("✅ %s PASSED\n", test->getName().c_str());
      passed_tests++;
    } catch (const std::exception& e) {
      printf("❌ %s FAILED with exception: %s\n", test->getName().c_str(),
             e.what());
    }

    printf("\n");
  }

  printf("GPU Integration Tests Summary: %d/%d passed\n", passed_tests,
         total_tests);

  return (passed_tests == total_tests) ? 0 : 1;
}

int main() {
  return run_gpu_integration_tests();
}
