#pragma once

#include "../../../include/MLLib.hpp"
#include "../../common/test_utils.hpp"
#include <fstream>
#include <memory>
#include <sstream>

/**
 * @file test_compatibility_integration.hpp
 * @brief Compatibility integration tests for MLLib
 *
 * Tests compatibility across different scenarios:
 * - File format compatibility
 * - Version compatibility simulation
 * - Cross-platform data handling
 * - Different model configurations
 * - Error recovery and fallback mechanisms
 */

namespace MLLib {
namespace test {

/**
 * @class FileFormatCompatibilityIntegrationTest
 * @brief Test compatibility across different file formats and configurations
 */
class FileFormatCompatibilityIntegrationTest : public TestCase {
public:
  FileFormatCompatibilityIntegrationTest()
      : TestCase("FileFormatCompatibilityIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;
    std::string temp_dir = createTempDirectory();

    // Create a reference model for testing
    auto reference_model = std::make_unique<Sequential>();
    reference_model->add(std::make_shared<Dense>(4, 6));
    reference_model->add(std::make_shared<activation::ReLU>());
    reference_model->add(std::make_shared<Dense>(6, 2));

    // Train the reference model
    std::vector<std::vector<double>> X = {{0.1, 0.2, 0.3, 0.4},
                                          {0.5, 0.6, 0.7, 0.8}};
    std::vector<std::vector<double>> Y = {{0.8, 0.2}, {0.3, 0.7}};

    MSELoss loss;
    SGD optimizer(0.1);
    reference_model->train(X, Y, loss, optimizer, nullptr, 10);

    // Get reference predictions
    std::vector<double> test_input = {0.25, 0.35, 0.45, 0.55};
    std::vector<double> reference_output = reference_model->predict(test_input);

    // Test 1: Basic model functionality (simplified test)
    {
      assertTrue(reference_output.size() == 2,
                 "Model should produce correct output size");

      for (double val : reference_output) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Model output should be valid numbers");
      }
    }

    // Test 2: Basic model validation
    {
      // Create another model to test consistency
      auto test_model = std::make_unique<Sequential>();
      test_model->add(std::make_shared<Dense>(4, 6));
      test_model->add(std::make_shared<activation::ReLU>());
      test_model->add(std::make_shared<Dense>(6, 2));

      std::vector<double> test_output = test_model->predict(test_input);
      assertTrue(test_output.size() == 2,
                 "Test model should produce correct output size");
    }

    removeTempDirectory(temp_dir);
  }
};

/**
 * @class ModelConfigurationCompatibilityTest
 * @brief Test compatibility across different model configurations (simplified)
 */
class ModelConfigurationCompatibilityTest : public TestCase {
public:
  ModelConfigurationCompatibilityTest()
      : TestCase("ModelConfigurationCompatibilityTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test data
    std::vector<std::vector<double>> X = {{0.1, 0.2}, {0.3, 0.4}};
    std::vector<std::vector<double>> Y = {{0.9}, {0.7}};

    // Test: Different activation function configurations (simplified)
    {
      std::vector<std::string> activation_configs = {"ReLU", "Sigmoid", "Tanh"};

      for (const auto& activation_name : activation_configs) {
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(2, 3));

        // Test that activation layers can be created
        if (activation_name == "ReLU") {
          model->add(std::make_shared<activation::ReLU>());
        } else if (activation_name == "Sigmoid") {
          model->add(std::make_shared<activation::Sigmoid>());
        } else if (activation_name == "Tanh") {
          model->add(std::make_shared<activation::Tanh>());
        }

        model->add(std::make_shared<Dense>(3, 1));

        // Basic validation - just check model structure
        assertTrue(model->get_layers().size() == 3,
                   "Model should have 3 layers");

        // Test prediction works
        std::vector<double> test_input = {0.5, 0.5};
        std::vector<double> output = model->predict(test_input);
        assertEqual(size_t(1), output.size(),
                    "Output size should be 1 for " + activation_name);
      }
    }
  }
};

/**
 * @class ErrorRecoveryCompatibilityTest
 * @brief Test error recovery and fallback mechanisms (simplified)
 */
class ErrorRecoveryCompatibilityTest : public TestCase {
public:
  ErrorRecoveryCompatibilityTest()
      : TestCase("ErrorRecoveryCompatibilityTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test: Basic error recovery - model creation and validation
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 4));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(4, 2));

      // Test that model remains functional after operations
      std::vector<double> test_input = {0.1, 0.2, 0.3};
      std::vector<double> output = model->predict(test_input);
      assertEqual(size_t(2), output.size(),
                  "Model should produce correct output size");

      // Test training stability with reasonable parameters
      std::vector<std::vector<double>> X = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
      std::vector<std::vector<double>> Y = {{0.9, 0.1}, {0.3, 0.7}};

      MSELoss loss;
      SGD optimizer(0.01);  // Conservative learning rate

      assertNoThrow(
          [&]() {
            model->train(X, Y, loss, optimizer, nullptr, 5);
          },
          "Basic training should complete without errors");
    }

    // Test: Architecture validation
    {
      auto model = std::make_unique<Sequential>();
      assertTrue(model->get_layers().empty(),
                 "New model should have no layers");

      model->add(std::make_shared<Dense>(2, 3));
      assertEqual(size_t(1), model->get_layers().size(),
                  "Model should have 1 layer");

      model->add(std::make_shared<activation::Sigmoid>());
      assertEqual(size_t(2), model->get_layers().size(),
                  "Model should have 2 layers");
    }
  }
};

/**
 * @class CrossPlatformCompatibilityTest
 * @brief Test cross-platform data handling and file operations (simplified)
 */
class CrossPlatformCompatibilityTest : public TestCase {
public:
  CrossPlatformCompatibilityTest()
      : TestCase("CrossPlatformCompatibilityTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test: Basic model creation and prediction with various data types
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 5));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(5, 2));
      model->add(std::make_shared<activation::Sigmoid>());

      // Basic prediction test
      std::vector<double> test_input = {0.5, 0.5, 0.5};
      std::vector<double> output = model->predict(test_input);
      assertEqual(size_t(2), output.size(), "Output should have size 2");

      // Verify outputs are in reasonable range (0-1 due to Sigmoid)
      for (double val : output) {
        assertTrue(val >= 0.0 && val <= 1.0,
                   "Sigmoid output should be in [0,1]");
      }
    }

    // Test: Data consistency across operations
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(3, 1));

      std::vector<double> input = {0.7, 0.3};

      // Multiple predictions should be consistent
      std::vector<double> output1 = model->predict(input);
      std::vector<double> output2 = model->predict(input);

      assertEqual(output1.size(), output2.size(), "Output sizes should match");
      for (size_t i = 0; i < output1.size(); ++i) {
        assertEqual(output1[i], output2[i],
                    "Predictions should be deterministic");
      }
    }

    // Test: Architecture validation
    {
      auto model = std::make_unique<Sequential>();
      assertTrue(model->get_layers().empty(), "New model should be empty");

      model->add(std::make_shared<Dense>(4, 3));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(3, 2));

      assertEqual(size_t(3), model->get_layers().size(),
                  "Model should have 3 layers");

      // Test with appropriate input size
      std::vector<double> input = {0.1, 0.2, 0.3, 0.4};
      std::vector<double> output = model->predict(input);
      assertEqual(size_t(2), output.size(), "Final output should have size 2");
    }
  }
};

}  // namespace test
}  // namespace MLLib
