#include "../../include/MLLib.hpp"
#include "../common/test_utils.hpp"
#include <memory>
#include <vector>

/**
 * @file integration_test_main.cpp
 * @brief Integration tests for MLLib
 *
 * These tests verify that different components of MLLib work together correctly
 * to solve real machine learning problems. Tests include:
 * - End-to-end model training and prediction
 * - Model saving and loading workflows
 * - Complex model architectures
 * - Performance benchmarks
 */

namespace MLLib {
namespace test {

/**
 * @class XORIntegrationTest
 * @brief End-to-end test using XOR problem
 */
class XORIntegrationTest : public TestCase {
public:
  XORIntegrationTest() : TestCase("XORIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    // Capture output during training to keep test output clean
    OutputCapture capture;

    // Create XOR model
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // XOR training data
    std::vector<std::vector<double>> X = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> Y = {{0.0}, {1.0}, {1.0}, {0.0}};

    // Train model
    MSE loss;
    SGD optimizer(0.5);  // Higher learning rate for faster convergence in test

    bool training_completed = false;
    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                // Training progress callback - output is captured
                if (epoch % 100 == 0) {
                  std::cout << "Epoch " << epoch << ", Loss: " << current_loss
                            << std::endl;
                }
              },
              500);  // Limited epochs for test
          training_completed = true;
        },
        "XOR training should complete without errors");

    assertTrue(training_completed, "Training should complete successfully");

    // Test predictions (should be approximately correct)
    std::vector<double> pred_00 = model->predict({0.0, 0.0});
    std::vector<double> pred_01 = model->predict({0.0, 1.0});
    std::vector<double> pred_10 = model->predict({1.0, 0.0});
    std::vector<double> pred_11 = model->predict({1.0, 1.0});

    // XOR truth table verification (with some tolerance)
    assertTrue(pred_00[0] < 0.3, "XOR(0,0) should be close to 0");
    assertTrue(pred_01[0] > 0.7, "XOR(0,1) should be close to 1");
    assertTrue(pred_10[0] > 0.7, "XOR(1,0) should be close to 1");
    assertTrue(pred_11[0] < 0.3, "XOR(1,1) should be close to 0");
  }
};

/**
 * @class ModelIOIntegrationTest
 * @brief Test complete model save/load workflow
 */
class ModelIOIntegrationTest : public TestCase {
public:
  ModelIOIntegrationTest() : TestCase("ModelIOIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Create and train a model
    auto original_model = std::make_unique<Sequential>();
    original_model->add(std::make_shared<Dense>(3, 5));
    original_model->add(std::make_shared<activation::ReLU>());
    original_model->add(std::make_shared<Dense>(5, 2));
    original_model->add(std::make_shared<activation::Sigmoid>());

    // Simple training data
    std::vector<std::vector<double>> X = {
        {1.0, 0.0, 0.5}, {0.0, 1.0, 0.3}, {0.5, 0.5, 1.0}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}};

    MLLib::loss::MSE loss;
    MLLib::optimizer::SGD optimizer(0.1);

    // Quick training
    original_model->train(X, Y, loss, optimizer, nullptr, 50);

    // Test predictions from original model
    std::vector<double> original_pred =
        original_model->predict({0.5, 0.5, 0.5});

    // Save model in different formats
    std::string temp_dir = createTempDirectory();

    std::string binary_path = temp_dir + "/test_model.bin";
    std::string json_path = temp_dir + "/test_model.json";
    std::string config_path = temp_dir + "/test_model.config";

    // Test binary format
    assertTrue(
        ModelIO::save_model(*original_model, binary_path, ModelFormat::BINARY),
        "Binary save should succeed");

    auto loaded_binary = ModelIO::load_model(binary_path, ModelFormat::BINARY);
    assertNotNull(loaded_binary.get(), "Binary load should succeed");

    std::vector<double> binary_pred = loaded_binary->predict({0.5, 0.5, 0.5});
    assertVectorNear(original_pred, binary_pred, 1e-6,
                     "Binary format should preserve model predictions");

    // Test JSON format
    assertTrue(
        ModelIO::save_model(*original_model, json_path, ModelFormat::JSON),
        "JSON save should succeed");

    auto loaded_json = ModelIO::load_model(json_path, ModelFormat::JSON);
    assertNotNull(loaded_json.get(), "JSON load should succeed");

    std::vector<double> json_pred = loaded_json->predict({0.5, 0.5, 0.5});
    assertVectorNear(original_pred, json_pred, 1e-6,
                     "JSON format should preserve model predictions");

    // Test config format (architecture only)
    assertTrue(ModelIO::save_config(*original_model, config_path),
               "Config save should succeed");

    auto loaded_config = ModelIO::load_config(config_path);
    assertNotNull(loaded_config.get(), "Config load should succeed");
    assertEqual(original_model->num_layers(), loaded_config->num_layers(),
                "Config should preserve model architecture");

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class MultiLayerIntegrationTest
 * @brief Test complex model architectures
 */
class MultiLayerIntegrationTest : public TestCase {
public:
  MultiLayerIntegrationTest() : TestCase("MultiLayerIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Create a deeper network
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(4, 8));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(8, 6));
    model->add(std::make_shared<activation::Tanh>());
    model->add(std::make_shared<Dense>(6, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    assertEqual(size_t(8), model->num_layers(), "Model should have 8 layers");

    // Test forward propagation through all layers
    NDArray input({1, 4});
    input[0] = 0.1;
    input[1] = 0.2;
    input[2] = 0.3;
    input[3] = 0.4;

    NDArray output = model->predict(input);
    assertEqual(size_t(2), output.shape().size(), "Output should be 2D");
    assertEqual(size_t(1), output.shape()[0], "Batch size should be 1");
    assertEqual(size_t(2), output.shape()[1], "Output should have 2 features");

    // Test batch prediction
    std::vector<NDArray> batch_inputs;
    for (int i = 0; i < 5; ++i) {
      NDArray batch_input({4});
      for (int j = 0; j < 4; ++j) {
        batch_input[j] = (i + j) * 0.1;
      }
      batch_inputs.push_back(batch_input);
    }

    std::vector<NDArray> batch_outputs = model->predict(batch_inputs);
    assertEqual(size_t(5), batch_outputs.size(),
                "Should predict for all batch inputs");

    for (const auto& out : batch_outputs) {
      assertEqual(size_t(1), out.shape().size(), "Each output should be 1D");
      assertEqual(size_t(2), out.shape()[0],
                  "Each output should have 2 features");
      // Sigmoid outputs should be in [0, 1]
      assertTrue(out[0] >= 0.0 && out[0] <= 1.0,
                 "Sigmoid output should be in [0,1]");
      assertTrue(out[1] >= 0.0 && out[1] <= 1.0,
                 "Sigmoid output should be in [0,1]");
    }
  }
};

/**
 * @class PerformanceIntegrationTest
 * @brief Basic performance and stability test
 */
class PerformanceIntegrationTest : public TestCase {
public:
  PerformanceIntegrationTest() : TestCase("PerformanceIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(10, 20));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(20, 10));
    model->add(std::make_shared<activation::Sigmoid>());

    // Generate larger dataset
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;

    for (int i = 0; i < 100; ++i) {
      std::vector<double> x(10);
      std::vector<double> y(10);
      for (int j = 0; j < 10; ++j) {
        x[j] = (i + j) * 0.01;
        y[j] = std::sin(x[j]);  // Some non-linear target
      }
      X.push_back(x);
      Y.push_back(y);
    }

    MLLib::loss::MSE loss;
    MLLib::optimizer::SGD optimizer(0.01);

    // Test training stability
    bool training_stable = true;
    double previous_loss = std::numeric_limits<double>::max();
    int increasing_loss_count = 0;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  training_stable = false;
                }
                if (current_loss > previous_loss) {
                  increasing_loss_count++;
                }
                previous_loss = current_loss;
              },
              100);
        },
        "Training should complete without throwing");

    assertTrue(training_stable, "Training should be numerically stable");
    assertTrue(increasing_loss_count < 50,
               "Loss should generally decrease during training");

    // Test batch prediction performance
    std::vector<NDArray> test_inputs;
    for (int i = 0; i < 50; ++i) {
      NDArray input({10});
      for (int j = 0; j < 10; ++j) {
        input[j] = i * 0.02 + j * 0.01;
      }
      test_inputs.push_back(input);
    }

    assertNoThrow(
        [&]() {
          std::vector<NDArray> predictions = model->predict(test_inputs);
          assertEqual(size_t(50), predictions.size(),
                      "Should handle batch prediction");
        },
        "Batch prediction should not throw");
  }
};

}  // namespace test
}  // namespace MLLib

int main() {
  using namespace MLLib::test;

  std::cout << "=== MLLib Integration Test Suite ===" << std::endl;
  std::cout << "Testing end-to-end functionality and workflows" << std::endl;
  std::cout << "Output capture enabled for clean test reporting" << std::endl;
  std::cout << std::endl;

  bool all_tests_passed = true;

  // XOR problem integration test
  {
    TestSuite xor_suite("XOR Problem Integration");
    xor_suite.addTest(std::make_unique<XORIntegrationTest>());

    bool suite_result = xor_suite.runAll();
    all_tests_passed &= suite_result;
  }

  // Model I/O integration test
  {
    TestSuite io_suite("Model I/O Integration");
    io_suite.addTest(std::make_unique<ModelIOIntegrationTest>());

    bool suite_result = io_suite.runAll();
    all_tests_passed &= suite_result;
  }

  // Multi-layer architecture test
  {
    TestSuite arch_suite("Multi-Layer Architecture");
    arch_suite.addTest(std::make_unique<MultiLayerIntegrationTest>());

    bool suite_result = arch_suite.runAll();
    all_tests_passed &= suite_result;
  }

  // Performance and stability test
  {
    TestSuite perf_suite("Performance and Stability");
    perf_suite.addTest(std::make_unique<PerformanceIntegrationTest>());

    bool suite_result = perf_suite.runAll();
    all_tests_passed &= suite_result;
  }

  // Final summary
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "INTEGRATION TEST SUMMARY" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  if (all_tests_passed) {
    std::cout << "🎉 ALL INTEGRATION TESTS PASSED! 🎉" << std::endl;
    std::cout << "MLLib components work together correctly." << std::endl;
  } else {
    std::cout << "❌ SOME INTEGRATION TESTS FAILED" << std::endl;
    std::cout << "Please review the test output and fix integration issues."
              << std::endl;
  }

  std::cout << std::string(60, '=') << std::endl;

  return all_tests_passed ? 0 : 1;
}
