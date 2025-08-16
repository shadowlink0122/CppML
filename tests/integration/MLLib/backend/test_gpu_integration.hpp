#pragma once

#include "../../../../include/MLLib.hpp"
#include "../../../common/test_utils.hpp"
#include <memory>

/**
 * @file test_gpu_integration.hpp
 * @brief GPU backend integration tests for MLLib
 *
 * Tests GPU backend integration:
 * - GPU/CPU fallback behavior
 * - GPU-model interaction
 * - Cross-device compatibility
 * - Performance characteristics
 * - Memory management
 */

namespace MLLib {
namespace test {

/**
 * @class GPUCPUFallbackIntegrationTest
 * @brief Test GPU-CPU fallback behavior in complete workflows
 */
class GPUCPUFallbackIntegrationTest : public TestCase {
public:
  GPUCPUFallbackIntegrationTest() : TestCase("GPUCPUFallbackIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Store original device
    DeviceType original_device = Device::getCurrentDevice();

    try {
      // Try to create GPU-based model
      auto gpu_model = std::make_unique<Sequential>(DeviceType::GPU);

      // Add layers - should work regardless of actual GPU availability
      gpu_model->add(std::make_shared<Dense>(2, 4));
      gpu_model->add(std::make_shared<activation::ReLU>());
      gpu_model->add(std::make_shared<Dense>(4, 1));
      gpu_model->add(std::make_shared<activation::Sigmoid>());

      // Check actual device (may have fallen back to CPU)
      DeviceType actual_device = gpu_model->get_device();
      if (Device::isGPUAvailable()) {
        assertTrue(actual_device == DeviceType::GPU,
                   "Model should use GPU when available");
      } else {
        assertTrue(actual_device == DeviceType::CPU,
                   "Model should fallback to CPU when GPU unavailable");
      }

      // Test XOR training to ensure functionality
      std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
      std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

      MSELoss loss;
      SGD optimizer(0.01);

      // Test training (simplified version)
      assertNoThrow(
          [&]() {
            gpu_model->train(X, y, loss, optimizer, nullptr, 10);
          },
          "GPU model training should complete without errors");

      // Test prediction
      std::vector<double> test_result = gpu_model->predict({0, 1});
      assertTrue(test_result.size() == 1, "Output should have correct size");
      assertTrue(std::isfinite(test_result[0]), "Output should be finite");

      // Test device switching
      gpu_model->set_device(DeviceType::CPU);
      assertTrue(gpu_model->get_device() == DeviceType::CPU,
                 "Should switch to CPU");

      // Test that inference still works after device switch
      std::vector<double> test_after_switch = gpu_model->predict({0, 1});
      assertTrue(test_after_switch.size() == 1,
                 "Output should have correct size after device switch");

    } catch (const std::exception& e) {
      assertFalse(true,
                  std::string("GPU integration test failed: ") + e.what());
    }

    // Restore original device
    Device::setDevice(original_device);
  }
};

/**
 * @class GPUModelComplexityIntegrationTest
 * @brief Test GPU handling with complex model architectures
 */
class GPUModelComplexityIntegrationTest : public TestCase {
public:
  GPUModelComplexityIntegrationTest()
      : TestCase("GPUModelComplexityIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;
    DeviceType original_device = Device::getCurrentDevice();

    try {
      // Create more complex model with GPU device
      auto model = std::make_unique<Sequential>(DeviceType::GPU);

      // Build deeper network
      model->add(std::make_shared<Dense>(4, 8));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(8, 16));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(16, 8));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(8, 1));
      model->add(std::make_shared<activation::Sigmoid>());

      // Generate synthetic data
      std::vector<std::vector<double>> X;
      std::vector<std::vector<double>> y;

      for (int i = 0; i < 16; ++i) {
        std::vector<double> input = {static_cast<double>(i % 4) / 4.0,
                                     static_cast<double>((i + 1) % 4) / 4.0,
                                     static_cast<double>((i + 2) % 4) / 4.0,
                                     static_cast<double>((i + 3) % 4) / 4.0};
        double target = (input[0] + input[1] > input[2] + input[3]) ? 1.0 : 0.0;

        X.push_back(input);
        y.push_back({target});
      }

      MSELoss loss;
      SGD optimizer(0.001);

      // Test training with larger dataset
      assertNoThrow(
          [&]() {
            model->train(X, y, loss, optimizer, nullptr, 5);
          },
          "Complex model training should complete without errors");

      // Test batch prediction
      for (size_t i = 0; i < X.size(); i += 4) {
        size_t batch_end = std::min(i + 4, X.size());
        for (size_t j = i; j < batch_end; ++j) {
          std::vector<double> output = model->predict(X[j]);
          assertTrue(output.size() == 1, "Output size should be consistent");
          assertTrue(std::isfinite(output[0]), "Output should be finite");
        }
      }

    } catch (const std::exception& e) {
      assertFalse(true,
                  std::string("Complex GPU model test failed: ") + e.what());
    }

    Device::setDevice(original_device);
  }
};

/**
 * @class GPUMemoryIntegrationTest
 * @brief Test GPU memory management in integration scenarios
 */
class GPUMemoryIntegrationTest : public TestCase {
public:
  GPUMemoryIntegrationTest() : TestCase("GPUMemoryIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;
    DeviceType original_device = Device::getCurrentDevice();

    try {
      // Test multiple model creation and destruction
      for (int test_iter = 0; test_iter < 3; ++test_iter) {
        auto model = std::make_unique<Sequential>(DeviceType::GPU);

        model->add(std::make_shared<Dense>(10, 20));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(20, 10));
        model->add(std::make_shared<activation::Sigmoid>());

        // Test large array processing
        std::vector<double> large_input;
        for (size_t i = 0; i < 10; ++i) {
          large_input.push_back(static_cast<double>(i) / 10.0);
        }

        std::vector<double> output = model->predict(large_input);
        assertTrue(output.size() == 10, "Large array processing should work");

        // Test multiple predictions
        for (int pass = 0; pass < 5; ++pass) {
          std::vector<double> temp_output = model->predict(large_input);
          assertTrue(temp_output.size() == 10, "Multiple passes should work");
        }

        // Model will be destroyed at end of loop iteration
      }

      // Test array operations with GPU device
      Device::setDeviceWithValidation(DeviceType::GPU, false);

      NDArray arr1({100});
      NDArray arr2({100});

      arr1.fill(1.5);
      arr2.fill(2.5);

      NDArray result = arr1 + arr2;
      assertNear(result.data()[0], 4.0, 1e-10,
                 "Array operations should work correctly");
      assertNear(result.data()[99], 4.0, 1e-10,
                 "Array operations should work for all elements");

      // Test matrix operations
      NDArray mat1({10, 10});
      NDArray mat2({10, 10});

      mat1.fill(0.1);
      mat2.fill(0.2);

      NDArray mat_result = mat1.matmul(mat2);
      assertTrue(mat_result.shape()[0] == 10,
                 "Matrix result should have correct shape");
      assertTrue(mat_result.shape()[1] == 10,
                 "Matrix result should have correct shape");

      // Check that result is reasonable (0.1 * 0.2 * 10 = 0.2 for each element)
      assertNear(mat_result.data()[0], 0.2, 0.01,
                 "Matrix multiplication should produce expected results");

    } catch (const std::exception& e) {
      assertFalse(true,
                  std::string("GPU memory integration test failed: ") +
                      e.what());
    }

    Device::setDevice(original_device);
  }
};

/**
 * @class GPUCrossDeviceIntegrationTest
 * @brief Test cross-device operations and data transfer
 */
class GPUCrossDeviceIntegrationTest : public TestCase {
public:
  GPUCrossDeviceIntegrationTest() : TestCase("GPUCrossDeviceIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;
    DeviceType original_device = Device::getCurrentDevice();

    try {
      // Create two models - one for GPU, one for CPU
      auto gpu_model = std::make_unique<Sequential>(DeviceType::GPU);
      auto cpu_model = std::make_unique<Sequential>(DeviceType::CPU);

      // Add identical architectures
      gpu_model->add(std::make_shared<Dense>(2, 4));
      gpu_model->add(std::make_shared<activation::ReLU>());
      gpu_model->add(std::make_shared<Dense>(4, 1));

      cpu_model->add(std::make_shared<Dense>(2, 4));
      cpu_model->add(std::make_shared<activation::ReLU>());
      cpu_model->add(std::make_shared<Dense>(4, 1));

      // Test with same input
      std::vector<double> test_input = {0.5, 0.7};

      std::vector<double> gpu_output = gpu_model->predict(test_input);
      std::vector<double> cpu_output = cpu_model->predict(test_input);

      assertTrue(gpu_output.size() == 1,
                 "GPU model output should have correct size");
      assertTrue(cpu_output.size() == 1,
                 "CPU model output should have correct size");
      assertTrue(std::isfinite(gpu_output[0]), "GPU output should be finite");
      assertTrue(std::isfinite(cpu_output[0]), "CPU output should be finite");

      // Test device switching within same model
      auto switchable_model = std::make_unique<Sequential>(DeviceType::GPU);
      switchable_model->add(std::make_shared<Dense>(3, 5));
      switchable_model->add(std::make_shared<activation::Tanh>());
      switchable_model->add(std::make_shared<Dense>(5, 2));

      std::vector<double> switch_input = {0.1, 0.2, 0.3};

      // Predict on initial device
      std::vector<double> output1 = switchable_model->predict(switch_input);

      // Switch device
      switchable_model->set_device(DeviceType::CPU);
      std::vector<double> output2 = switchable_model->predict(switch_input);
      DeviceType device2 = switchable_model->get_device();

      // Switch back (if GPU was available)
      switchable_model->set_device(DeviceType::GPU);
      std::vector<double> output3 = switchable_model->predict(switch_input);
      DeviceType device3 = switchable_model->get_device();

      assertTrue(output1.size() == 2,
                 "Output should maintain size through device switches");
      assertTrue(output2.size() == 2,
                 "Output should maintain size through device switches");
      assertTrue(output3.size() == 2,
                 "Output should maintain size through device switches");

      assertTrue(device2 == DeviceType::CPU, "Should switch to CPU");

      // If GPU is available, device3 should be GPU, otherwise CPU
      if (Device::isGPUAvailable()) {
        assertTrue(device3 == DeviceType::GPU,
                   "Should switch back to GPU when available");
      } else {
        assertTrue(device3 == DeviceType::CPU,
                   "Should stay on CPU when GPU not available");
      }

    } catch (const std::exception& e) {
      assertFalse(true,
                  std::string("Cross-device integration test failed: ") +
                      e.what());
    }

    Device::setDevice(original_device);
  }
};

}  // namespace test
}  // namespace MLLib
