#pragma once

#include "../../../../include/MLLib.hpp"
#include "../../../common/test_utils.hpp"
#include <memory>

/**
 * @file test_layer_integration.hpp
 * @brief Layer integration tests for MLLib
 *
 * Tests the integration of different layer types in real scenarios:
 * - Layer combination in sequential models
 * - Data flow through multiple layer types
 * - Gradient flow in complex architectures
 * - Performance with different layer configurations
 */

namespace MLLib {
namespace test {

/**
 * @class LayerCombinationIntegrationTest
 * @brief Test various layer combinations in integrated scenarios
 */
class LayerCombinationIntegrationTest : public TestCase {
public:
  LayerCombinationIntegrationTest()
      : TestCase("LayerCombinationIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test 1: Dense + ReLU + Dense combination
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 5));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(5, 2));

      NDArray input({1, 3});
      input[0] = 1.0;
      input[1] = -0.5;
      input[2] = 0.3;

      NDArray output = model->predict(input);
      assertEqual(size_t(2), output.shape().size(), "Output should be 2D");
      assertEqual(size_t(1), output.shape()[0], "Batch size should be 1");
      assertEqual(size_t(2), output.shape()[1], "Output features should be 2");

      // Test that ReLU eliminates negative values in intermediate layer
      // (we can't directly access intermediate values, but the integration
      // should work)
      assertTrue(output[0] != 0.0 || output[1] != 0.0,
                 "Output should not be zero vector");
    }

    // Test 2: All activation functions in sequence
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 4));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(4, 4));
      model->add(std::make_shared<activation::Sigmoid>());
      model->add(std::make_shared<Dense>(4, 4));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(4, 1));

      std::vector<double> input = {0.5, -0.3};
      std::vector<double> output = model->predict(input);

      assertEqual(size_t(1), output.size(), "Final output should be scalar");
      assertTrue(output[0] >= -1.5 && output[0] <= 1.5,
                 "Output should be in reasonable range");
    }

    // Test 3: Wide vs Deep network comparison
    {
      // Wide network (fewer layers, more neurons)
      auto wide_model = std::make_unique<Sequential>();
      wide_model->add(std::make_shared<Dense>(3, 20));
      wide_model->add(std::make_shared<activation::ReLU>());
      wide_model->add(std::make_shared<Dense>(20, 1));

      // Deep network (more layers, fewer neurons)
      auto deep_model = std::make_unique<Sequential>();
      deep_model->add(std::make_shared<Dense>(3, 5));
      deep_model->add(std::make_shared<activation::ReLU>());
      deep_model->add(std::make_shared<Dense>(5, 5));
      deep_model->add(std::make_shared<activation::ReLU>());
      deep_model->add(std::make_shared<Dense>(5, 5));
      deep_model->add(std::make_shared<activation::ReLU>());
      deep_model->add(std::make_shared<Dense>(5, 1));

      std::vector<double> test_input = {0.2, 0.8, -0.4};

      // Both should produce valid outputs
      std::vector<double> wide_output = wide_model->predict(test_input);
      std::vector<double> deep_output = deep_model->predict(test_input);

      assertEqual(size_t(1), wide_output.size(),
                  "Wide model should produce scalar output");
      assertEqual(size_t(1), deep_output.size(),
                  "Deep model should produce scalar output");

      assertTrue(!std::isnan(wide_output[0]),
                 "Wide model output should not be NaN");
      assertTrue(!std::isnan(deep_output[0]),
                 "Deep model output should not be NaN");
    }

    // Test 4: Batch processing through layer combinations
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(3, 2));
      model->add(std::make_shared<activation::Sigmoid>());

      // Process multiple inputs
      std::vector<std::vector<double>> batch_inputs = {
          {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, -0.5}};

      for (size_t i = 0; i < batch_inputs.size(); ++i) {
        std::vector<double> output = model->predict(batch_inputs[i]);
        assertEqual(size_t(2), output.size(),
                    "Each batch output should have 2 elements");

        // Sigmoid outputs should be in [0, 1]
        assertTrue(output[0] >= 0.0 && output[0] <= 1.0,
                   "Sigmoid output[0] should be in [0,1]");
        assertTrue(output[1] >= 0.0 && output[1] <= 1.0,
                   "Sigmoid output[1] should be in [0,1]");
      }
    }
  }
};

/**
 * @class ActivationIntegrationTest
 * @brief Test activation functions in integrated training scenarios
 */
class ActivationIntegrationTest : public TestCase {
public:
  ActivationIntegrationTest() : TestCase("ActivationIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test 1: ReLU activation in training
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 4));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(4, 1));

      std::vector<std::vector<double>> X = {
          {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
      std::vector<std::vector<double>> Y = {{1.0}, {1.0}, {0.0}, {0.0}};

      MSELoss loss;
      SGD optimizer(0.1);

      bool training_stable = true;
      assertNoThrow(
          [&]() {
            model->train(
                X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                  if (std::isnan(current_loss) || std::isinf(current_loss)) {
                    training_stable = false;
                  }
                },
                50);
          },
          "ReLU training should not throw");

      assertTrue(training_stable, "ReLU training should be stable");
    }

    // Test 2: Sigmoid activation in training
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::Sigmoid>());
      model->add(std::make_shared<Dense>(3, 1));
      model->add(std::make_shared<activation::Sigmoid>());

      std::vector<std::vector<double>> X = {{0.1, 0.9}, {0.9, 0.1}, {0.5, 0.5}};
      std::vector<std::vector<double>> Y = {{0.8}, {0.2}, {0.5}};

      MSELoss loss;
      SGD optimizer(0.5);

      bool training_stable = true;
      assertNoThrow(
          [&]() {
            model->train(
                X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                  if (std::isnan(current_loss) || std::isinf(current_loss)) {
                    training_stable = false;
                  }
                },
                50);
          },
          "Sigmoid training should not throw");

      assertTrue(training_stable, "Sigmoid training should be stable");

      // Test that outputs are in valid range
      for (const auto& input : X) {
        std::vector<double> output = model->predict(input);
        assertTrue(output[0] >= 0.0 && output[0] <= 1.0,
                   "Sigmoid output should be in [0,1]");
      }
    }

    // Test 3: Tanh activation in training
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(1, 3));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(3, 1));

      // Simple function approximation: y = sin(x)
      std::vector<std::vector<double>> X;
      std::vector<std::vector<double>> Y;

      for (int i = 0; i < 10; ++i) {
        double x = (i - 5) * 0.2;  // Range [-1, 1]
        X.push_back({x});
        Y.push_back({std::sin(x)});
      }

      MSELoss loss;
      SGD optimizer(0.1);

      bool training_stable = true;
      assertNoThrow(
          [&]() {
            model->train(
                X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                  if (std::isnan(current_loss) || std::isinf(current_loss)) {
                    training_stable = false;
                  }
                },
                100);
          },
          "Tanh training should not throw");

      assertTrue(training_stable, "Tanh training should be stable");

      // Test reasonable approximation
      std::vector<double> test_input = {0.0};
      std::vector<double> test_output = model->predict(test_input);
      assertTrue(test_output[0] >= -1.5 && test_output[0] <= 1.5,
                 "Tanh output should be in reasonable range");
    }

    // Test 4: Mixed activation functions
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 4));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(4, 3));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(3, 1));
      model->add(std::make_shared<activation::Sigmoid>());

      std::vector<std::vector<double>> X = {
          {0.3, 0.7}, {0.8, 0.2}, {0.1, 0.9}, {0.6, 0.4}};
      std::vector<std::vector<double>> Y = {{0.6}, {0.4}, {0.8}, {0.5}};

      MSELoss loss;
      SGD optimizer(0.3);

      bool training_completed = false;
      assertNoThrow(
          [&]() {
            model->train(X, Y, loss, optimizer, nullptr, 30);
            training_completed = true;
          },
          "Mixed activation training should work");

      assertTrue(training_completed,
                 "Mixed activation training should complete");

      // Verify outputs are in expected ranges
      for (const auto& input : X) {
        std::vector<double> output = model->predict(input);
        assertTrue(output[0] >= 0.0 && output[0] <= 1.0,
                   "Final sigmoid output should be in [0,1]");
      }
    }
  }
};

/**
 * @class LayerPerformanceIntegrationTest
 * @brief Test layer performance in realistic scenarios
 */
class LayerPerformanceIntegrationTest : public TestCase {
public:
  LayerPerformanceIntegrationTest()
      : TestCase("LayerPerformanceIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test 1: Large input processing
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(50, 20));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(20, 10));
      model->add(std::make_shared<activation::Sigmoid>());
      model->add(std::make_shared<Dense>(10, 1));

      // Generate large input
      std::vector<double> large_input(50);
      for (size_t i = 0; i < 50; ++i) {
        large_input[i] = (i % 10) * 0.1 - 0.5;
      }

      std::vector<double> output;
      assertNoThrow([&]() { output = model->predict(large_input); },
                    "Large input processing should not throw");

      assertEqual(size_t(1), output.size(),
                  "Large input should produce scalar output");
      assertTrue(!std::isnan(output[0]),
                 "Large input output should not be NaN");
    }

    // Test 2: Many layer deep network
    {
      auto model = std::make_unique<Sequential>();

      // Build deep network
      model->add(std::make_shared<Dense>(5, 8));
      for (int i = 0; i < 6; ++i) {  // 6 hidden layers
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(8, 8));
      }
      model->add(std::make_shared<activation::Sigmoid>());
      model->add(std::make_shared<Dense>(8, 1));

      assertTrue(model->num_layers() > 10, "Should create deep network");

      std::vector<double> input = {0.1, 0.2, 0.3, 0.4, 0.5};
      std::vector<double> output;

      assertNoThrow([&]() { output = model->predict(input); },
                    "Deep network prediction should not throw");

      assertEqual(size_t(1), output.size(),
                  "Deep network should produce scalar output");
      assertTrue(!std::isnan(output[0]),
                 "Deep network output should not be NaN");
      assertTrue(!std::isinf(output[0]),
                 "Deep network output should not be infinite");
    }

    // Test 3: Batch processing performance
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(10, 15));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(15, 10));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(10, 5));

      // Generate batch of inputs
      std::vector<std::vector<double>> batch_inputs;
      for (int i = 0; i < 20; ++i) {
        std::vector<double> input(10);
        for (int j = 0; j < 10; ++j) {
          input[j] = (i + j) * 0.05 - 0.5;
        }
        batch_inputs.push_back(input);
      }

      assertNoThrow(
          [&]() {
            for (const auto& input : batch_inputs) {
              std::vector<double> output = model->predict(input);
              assertEqual(size_t(5), output.size(),
                          "Each output should have 5 elements");
            }
          },
          "Batch processing should not throw");
    }

    // Test 4: Memory stability with repeated predictions
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 5));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(5, 3));

      std::vector<double> test_input = {0.5, -0.3, 0.8};
      std::vector<double> first_output = model->predict(test_input);

      // Repeated predictions should give consistent results
      for (int i = 0; i < 100; ++i) {
        std::vector<double> output = model->predict(test_input);
        assertVectorNear(first_output, output, 1e-12,
                         "Repeated predictions should be identical");
      }
    }
  }
};

}  // namespace test
}  // namespace MLLib
