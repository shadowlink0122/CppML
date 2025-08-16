#pragma once

#include "../../../common/test_utils.hpp"
#pragma once

#include "../../../../include/MLLib.hpp"
#include <memory>

/**
 * @file test_device_integration.hpp
 * @brief Device integration tests for MLLib
 *
 * Tests device integration:
 * - CPU device functionality
 * - Memory allocation and management
 * - Device-specific operations
 * - Performance characteristics
 */

namespace MLLib {
namespace test {

/**
 * @class CPUDeviceIntegrationTest
 * @brief Test CPU device functionality in real scenarios
 */
class CPUDeviceIntegrationTest : public TestCase {
public:
  CPUDeviceIntegrationTest() : TestCase("CPUDeviceIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test CPU device with model training
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(4, 6));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(6, 3));
    model->add(std::make_shared<activation::Sigmoid>());

    // Training data for CPU processing
    std::vector<std::vector<double>> X = {{0.1, 0.2, 0.3, 0.4},
                                          {0.5, 0.6, 0.7, 0.8},
                                          {0.2, 0.4, 0.6, 0.8},
                                          {0.1, 0.3, 0.5, 0.7},
                                          {0.9, 0.8, 0.7, 0.6},
                                          {0.3, 0.6, 0.9, 0.2}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                                          {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0},
                                          {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    MSELoss loss;
    SGD optimizer(0.1);

    bool cpu_training_stable = true;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  cpu_training_stable = false;
                }
              },
              100);
        },
        "CPU device training should complete");

    assertTrue(cpu_training_stable, "CPU device training should be stable");

    // Test CPU device predictions
    for (const auto& input : X) {
      std::vector<double> pred = model->predict(input);
      assertEqual(size_t(3), pred.size(),
                  "CPU device should produce correct output size");

      for (double val : pred) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "CPU device should produce valid outputs");
        assertTrue(val >= 0.0 && val <= 1.0,
                   "CPU sigmoid outputs should be in valid range");
      }
    }

    // Test CPU device performance with larger input
    std::vector<double> large_input(4, 0.5);
    std::vector<double> large_output = model->predict(large_input);

    assertEqual(size_t(3), large_output.size(),
                "CPU device should handle standard inputs efficiently");
  }
};

/**
 * @class DeviceMemoryIntegrationTest
 * @brief Test device memory management
 */
class DeviceMemoryIntegrationTest : public TestCase {
public:
  DeviceMemoryIntegrationTest() : TestCase("DeviceMemoryIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test memory management with multiple models
    std::vector<std::unique_ptr<Sequential>> models;

    for (int i = 0; i < 3; i++) {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(5, 10));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(10, 5));
      model->add(std::make_shared<activation::Sigmoid>());

      // Test that each model can make predictions
      std::vector<double> test_input = {0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i,
                                        0.5 * i};
      std::vector<double> output = model->predict(test_input);

      assertEqual(size_t(5), output.size(),
                  "Device memory should support multiple models");

      for (double val : output) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Device memory should maintain data integrity");
      }

      models.push_back(std::move(model));
    }

    // Test that all models work independently
    for (size_t i = 0; i < models.size(); i++) {
      std::vector<double> test_input = {0.5, 0.5, 0.5, 0.5, 0.5};
      std::vector<double> output = models[i]->predict(test_input);

      assertTrue(output.size() == 5,
                 "Device memory should preserve model functionality");
    }

    // Test memory cleanup (implicit through RAII)
    models.clear();

    // Create new model to test memory reuse
    auto new_model = std::make_unique<Sequential>();
    new_model->add(std::make_shared<Dense>(3, 4));
    new_model->add(std::make_shared<activation::Tanh>());

    std::vector<double> cleanup_test = new_model->predict({0.1, 0.2, 0.3});
    assertEqual(size_t(4), cleanup_test.size(),
                "Device memory cleanup should allow new allocations");
  }
};

/**
 * @class DeviceOperationsIntegrationTest
 * @brief Test device-specific operations
 */
class DeviceOperationsIntegrationTest : public TestCase {
public:
  DeviceOperationsIntegrationTest()
      : TestCase("DeviceOperationsIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test various device operations through different model configurations

    // Test 1: Linear operations (Dense layers)
    {
      auto linear_model = std::make_unique<Sequential>();
      linear_model->add(std::make_shared<Dense>(3, 6));
      linear_model->add(std::make_shared<Dense>(6, 2));

      std::vector<double> linear_input = {1.0, 2.0, 3.0};
      std::vector<double> linear_output = linear_model->predict(linear_input);

      assertEqual(size_t(2), linear_output.size(),
                  "Device should handle linear operations");

      for (double val : linear_output) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Linear device operations should be stable");
      }
    }

    // Test 2: Non-linear operations (Activation functions)
    {
      auto nonlinear_model = std::make_unique<Sequential>();
      nonlinear_model->add(std::make_shared<Dense>(2, 4));
      nonlinear_model->add(std::make_shared<activation::ReLU>());
      nonlinear_model->add(std::make_shared<activation::Sigmoid>());
      nonlinear_model->add(std::make_shared<activation::Tanh>());

      std::vector<double> nonlinear_input = {-1.0, 1.0};
      std::vector<double> nonlinear_output =
          nonlinear_model->predict(nonlinear_input);

      assertEqual(size_t(4), nonlinear_output.size(),
                  "Device should handle non-linear operations");

      for (double val : nonlinear_output) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Non-linear device operations should be stable");
      }
    }

    // Test 3: Training operations (Gradient computation and updates)
    {
      auto training_model = std::make_unique<Sequential>();
      training_model->add(std::make_shared<Dense>(2, 3));
      training_model->add(std::make_shared<activation::ReLU>());
      training_model->add(std::make_shared<Dense>(3, 1));
      training_model->add(std::make_shared<activation::Sigmoid>());

      std::vector<std::vector<double>> X = {{0.1, 0.9}, {0.9, 0.1}, {0.5, 0.5}};
      std::vector<std::vector<double>> Y = {{1.0}, {0.0}, {0.5}};

      MSELoss loss;
      SGD optimizer(0.1);

      bool device_training_stable = true;

      assertNoThrow(
          [&]() {
            training_model->train(
                X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                  if (std::isnan(current_loss) || std::isinf(current_loss)) {
                    device_training_stable = false;
                  }
                },
                30);
          },
          "Device training operations should complete");

      assertTrue(device_training_stable,
                 "Device training operations should be stable");

      // Verify training affected the model
      std::vector<double> trained_output = training_model->predict({0.2, 0.8});
      assertTrue(!std::isnan(trained_output[0]) &&
                     !std::isinf(trained_output[0]),
                 "Device training should produce valid trained model");
    }
  }
};

/**
 * @class DevicePerformanceIntegrationTest
 * @brief Test device performance characteristics
 */
class DevicePerformanceIntegrationTest : public TestCase {
public:
  DevicePerformanceIntegrationTest()
      : TestCase("DevicePerformanceIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test performance with different model sizes
    std::vector<std::pair<int, int>> size_configs = {{5, 10},
                                                     {10, 20},
                                                     {20, 30}};

    for (const auto& config : size_configs) {
      int input_size = config.first;
      int hidden_size = config.second;

      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(input_size, hidden_size));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(hidden_size, input_size));
      model->add(std::make_shared<activation::Sigmoid>());

      // Performance test: multiple predictions
      std::vector<double> test_input(input_size, 0.5);

      for (int i = 0; i < 50; i++) {
        std::vector<double> output = model->predict(test_input);
        assertEqual(static_cast<size_t>(input_size), output.size(),
                    "Device should maintain performance across predictions");

        for (double val : output) {
          assertTrue(!std::isnan(val) && !std::isinf(val),
                     "Device performance should not degrade with repeated use");
        }
      }
    }

    // Test batch-like processing
    auto batch_model = std::make_unique<Sequential>();
    batch_model->add(std::make_shared<Dense>(4, 6));
    batch_model->add(std::make_shared<activation::Tanh>());
    batch_model->add(std::make_shared<Dense>(6, 2));

    std::vector<std::vector<double>> batch_inputs;
    for (int i = 0; i < 20; i++) {
      std::vector<double> input(4);
      for (int j = 0; j < 4; j++) {
        input[j] = (i + j) * 0.05;
      }
      batch_inputs.push_back(input);
    }

    // Process batch sequentially (simulating batch processing)
    std::vector<std::vector<double>> batch_outputs;
    for (const auto& input : batch_inputs) {
      batch_outputs.push_back(batch_model->predict(input));
    }

    assertEqual(batch_inputs.size(), batch_outputs.size(),
                "Device should handle batch-like processing");

    for (const auto& output : batch_outputs) {
      assertEqual(size_t(2), output.size(),
                  "Device batch processing should maintain output consistency");
      for (double val : output) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Device batch processing should be stable");
      }
    }
  }
};

}  // namespace test
}  // namespace MLLib
