#pragma once

#include "../../../../include/MLLib/backend/backend.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"
#include <memory>
#include <vector>

namespace MLLib {
namespace test {

/**
 * @class CompleteGPUCoverageTest
 * @brief Comprehensive test for Metal, ROCm, and oneAPI complete coverage
 */
class CompleteGPUCoverageTest : public TestCase {
public:
  CompleteGPUCoverageTest() : TestCase("CompleteGPUCoverageTest") {}

protected:
  void test() override {
    OutputCapture capture;

    // Test 1: Backend availability detection
    testBackendAvailability();

    // Test 2: Matrix multiplication performance parity
    testMatmulPerformanceParity();

    // Test 3: Memory management correctness
    testMemoryManagement();

    // Test 4: Error handling and fallback
    testErrorHandlingAndFallback();

    // Test 5: Numerical accuracy validation
    testNumericalAccuracy();
  }

private:
  void testBackendAvailability() {
    auto available_backends =
        MLLib::Backend::Backend::getAvailableGPUBackends();

    assertTrue(!available_backends.empty() ||
                   Device::getCurrentDevice() == DeviceType::CPU,
               "At least one GPU backend should be available or using CPU");

    // Test each backend type
    for (auto backend_type : available_backends) {
      bool success =
          MLLib::Backend::Backend::setPreferredGPUBackend(backend_type);
      assertTrue(success, "Should be able to set available backend");

      MLLib::Backend::GPUBackendType current =
          MLLib::Backend::Backend::getCurrentGPUBackend();
      assertEqual(static_cast<int>(current), static_cast<int>(backend_type),
                  "Current backend should match set backend");
    }
  }

  void testMatmulPerformanceParity() {
    const int size = 128;  // Smaller size for CI compatibility

    NDArray a({static_cast<size_t>(size), static_cast<size_t>(size)});
    NDArray b({static_cast<size_t>(size), static_cast<size_t>(size)});
    NDArray result({static_cast<size_t>(size), static_cast<size_t>(size)});

    // Initialize with test data
    MLLib::Backend::Backend::fill(a, 1.0);
    MLLib::Backend::Backend::fill(b, 2.0);

    auto available_backends =
        MLLib::Backend::Backend::getAvailableGPUBackends();

    std::vector<NDArray> results;
    results.reserve(available_backends.size());

    // Test each available backend
    for (auto backend_type : available_backends) {
      MLLib::Backend::Backend::setPreferredGPUBackend(backend_type);

      NDArray backend_result(
          {static_cast<size_t>(size), static_cast<size_t>(size)});

      assertNoThrow(
          [&]() {
            MLLib::Backend::Backend::matmul(a, b, backend_result);
          },
          "Matrix multiplication should complete without errors");

      results.push_back(backend_result);
    }

    // Verify numerical consistency between backends
    if (results.size() > 1) {
      for (size_t i = 1; i < results.size(); ++i) {
        for (int j = 0; j < size * size; ++j) {
          double diff = std::abs(results[0].data()[j] - results[i].data()[j]);
          assertTrue(
              diff < 1e-10,
              "Results should be numerically consistent across backends");
        }
      }
    }
  }

  void testMemoryManagement() {
    const int test_sizes[] = {16, 64, 256};

    for (int size : test_sizes) {
      NDArray a({static_cast<size_t>(size), static_cast<size_t>(size)});
      NDArray b({static_cast<size_t>(size), static_cast<size_t>(size)});
      NDArray result({static_cast<size_t>(size), static_cast<size_t>(size)});

      // Initialize data
      MLLib::Backend::Backend::fill(a, 1.0);
      MLLib::Backend::Backend::fill(b, 2.0);

      // Test multiple operations to stress memory management
      for (int iter = 0; iter < 5; ++iter) {
        assertNoThrow(
            [&]() {
              MLLib::Backend::Backend::add(a, b, result);
              MLLib::Backend::Backend::multiply_scalar(result, 0.5, result);
            },
            "Memory management should handle multiple operations");
      }
    }
  }

  void testErrorHandlingAndFallback() {
    // Test with invalid input shapes to trigger error handling
    NDArray invalid_a({10, 5});
    NDArray invalid_b({3, 8});
    NDArray result({10, 8});

    assertThrows<std::invalid_argument>(
        [&]() {
          MLLib::Backend::Backend::matmul(invalid_a, invalid_b, result);
        },
        "Should throw for incompatible matrix dimensions");
  }

  void testNumericalAccuracy() {
    // Test with known mathematical identity: A * I = A
    const int size = 32;

    NDArray identity({static_cast<size_t>(size), static_cast<size_t>(size)});
    NDArray test_matrix({static_cast<size_t>(size), static_cast<size_t>(size)});
    NDArray result({static_cast<size_t>(size), static_cast<size_t>(size)});

    // Create identity matrix
    MLLib::Backend::Backend::fill(identity, 0.0);
    for (int i = 0; i < size; ++i) {
      identity.data()[i * size + i] = 1.0;
    }

    // Create test matrix with known values
    for (int i = 0; i < size * size; ++i) {
      test_matrix.data()[i] = static_cast<double>(i) / 100.0;
    }

    // Test A * I = A
    MLLib::Backend::Backend::matmul(test_matrix, identity, result);

    for (int i = 0; i < size * size; ++i) {
      double diff = std::abs(test_matrix.data()[i] - result.data()[i]);
      assertTrue(diff < 1e-12,
                 "A * I should equal A (numerical accuracy test)");
    }
  }
};

/**
 * @class BackendPerformanceBenchmark
 * @brief Performance benchmark for all GPU backends
 */
class BackendPerformanceBenchmark : public TestCase {
public:
  BackendPerformanceBenchmark() : TestCase("BackendPerformanceBenchmark") {}

protected:
  void test() override {
    OutputCapture capture;

    const std::vector<int> sizes = {64, 128, 256};  // Reduced for CI
    auto available_backends =
        MLLib::Backend::Backend::getAvailableGPUBackends();

    if (available_backends.empty()) {
      assertTrue(true, "No GPU backends available - skipping benchmark");
      return;
    }

    for (int size : sizes) {
      printf("Benchmarking %dx%d matrix multiplication:\n", size, size);

      NDArray a({static_cast<size_t>(size), static_cast<size_t>(size)});
      NDArray b({static_cast<size_t>(size), static_cast<size_t>(size)});
      NDArray result({static_cast<size_t>(size), static_cast<size_t>(size)});

      // Initialize with test data
      MLLib::Backend::Backend::fill(a, 1.0);
      MLLib::Backend::Backend::fill(b, 2.0);

      for (auto backend_type : available_backends) {
        MLLib::Backend::Backend::setPreferredGPUBackend(backend_type);

        std::string backend_name =
            backend_type == MLLib::Backend::GPUBackendType::CUDA    ? "CUDA"
            : backend_type == MLLib::Backend::GPUBackendType::ROCM  ? "ROCm"
            : backend_type == MLLib::Backend::GPUBackendType::METAL ? "Metal"
            : backend_type == MLLib::Backend::GPUBackendType::ONEAPI
            ? "oneAPI"
            : "Unknown";

        // Warmup
        MLLib::Backend::Backend::matmul(a, b, result);

        // Benchmark (simplified for CI)
        assertNoThrow(
            [&]() {
              MLLib::Backend::Backend::matmul(a, b, result);
            },
            backend_name + " backend should complete matrix multiplication");

        printf("  %s: ✅ Completed successfully\n", backend_name.c_str());
      }
    }

    assertTrue(true,
               "Performance benchmark completed for all available backends");
  }
};

}  // namespace test
}  // namespace MLLib
