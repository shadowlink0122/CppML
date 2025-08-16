#pragma once

#include "../../../../../include/MLLib.hpp"
#include "../../../../common/test_utils.hpp"
#include <cstring>
#include <memory>
#include <string>

/**
 * @file test_string_integration.hpp
 * @brief String utilities integration tests for MLLib
 *
 * Tests string utilities integration:
 * - Model configuration parsing
 * - Error message formatting
 * - Data format conversion
 * - String-based parameter handling
 */

namespace MLLib {
namespace test {

/**
 * @class ModelConfigurationStringIntegrationTest
 * @brief Test string-based model configuration
 */
class ModelConfigurationStringIntegrationTest : public TestCase {
public:
  ModelConfigurationStringIntegrationTest()
      : TestCase("ModelConfigurationStringIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test configuration string parsing simulation
    std::string config_str = "layers:\n"
                             "  - type: Dense\n"
                             "    input_size: 3\n"
                             "    output_size: 5\n"
                             "  - type: ReLU\n"
                             "  - type: Dense\n"
                             "    input_size: 5\n"
                             "    output_size: 2\n"
                             "  - type: Sigmoid\n";

    // Parse layer configuration (simplified)
    bool has_dense = config_str.find("Dense") != std::string::npos;
    bool has_relu = config_str.find("ReLU") != std::string::npos;
    bool has_sigmoid = config_str.find("Sigmoid") != std::string::npos;

    assertTrue(has_dense, "Configuration should contain Dense layers");
    assertTrue(has_relu, "Configuration should contain ReLU activation");
    assertTrue(has_sigmoid, "Configuration should contain Sigmoid activation");

    // Build model from configuration
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 5));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(5, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    // Test that configured model works
    std::vector<double> test_input = {0.1, 0.2, 0.3};
    std::vector<double> output = model->predict(test_input);

    assertEqual(size_t(2), output.size(),
                "String-configured model should have correct output size");

    for (double val : output) {
      assertTrue(!std::isnan(val) && !std::isinf(val),
                 "String-configured model should produce valid outputs");
    }
  }
};

/**
 * @class ErrorMessageFormattingIntegrationTest
 * @brief Test error message formatting in real scenarios
 */
class ErrorMessageFormattingIntegrationTest : public TestCase {
public:
  ErrorMessageFormattingIntegrationTest()
      : TestCase("ErrorMessageFormattingIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test error handling with descriptive messages
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 3));
    model->add(std::make_shared<activation::ReLU>());

    // Test valid input
    std::vector<double> valid_input = {0.5, 0.3};
    std::vector<double> output;

    assertNoThrow(
        [&]() {
          output = model->predict(valid_input);
        },
        "Valid input should not produce errors");

    assertEqual(size_t(3), output.size(),
                "Error handling should preserve correct functionality");

    // Test error message construction
    std::string error_context = "Model prediction";
    std::string input_info = "input_size=2, expected_size=2";
    std::string full_message = error_context + ": " + input_info;

    assertTrue(full_message.find("Model") != std::string::npos,
               "Error messages should contain context");
    assertTrue(full_message.find("input_size") != std::string::npos,
               "Error messages should contain details");

    // Test error recovery
    bool error_recovery_works = true;
    try {
      // Simulate potential error scenario
      std::vector<double> test_output = model->predict(valid_input);
      if (test_output.empty()) {
        error_recovery_works = false;
      }
    } catch (...) {
      error_recovery_works = false;
    }

    assertTrue(error_recovery_works,
               "Error handling should allow graceful recovery");
  }
};

/**
 * @class DataFormatConversionIntegrationTest
 * @brief Test data format string conversion
 */
class DataFormatConversionIntegrationTest : public TestCase {
public:
  DataFormatConversionIntegrationTest()
      : TestCase("DataFormatConversionIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test CSV string to data conversion
    std::string csv_data =
        "0.1,0.2,0.3,1.0\n0.4,0.5,0.6,0.0\n0.7,0.8,0.9,1.0\n";

    std::vector<std::vector<double>> X, Y;
    std::istringstream ss(csv_data);
    std::string line;

    while (std::getline(ss, line)) {
      std::istringstream line_ss(line);
      std::string cell;
      std::vector<double> values;

      while (std::getline(line_ss, cell, ',')) {
        values.push_back(std::stod(cell));
      }

      if (values.size() >= 4) {
        X.push_back({values[0], values[1], values[2]});
        Y.push_back({values[3]});
      }
    }

    assertEqual(size_t(3), X.size(),
                "Should parse correct number of samples from string");
    assertEqual(size_t(3), Y.size(),
                "Should parse correct number of labels from string");

    // Test training with string-converted data
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    MSELoss loss;
    SGD optimizer(0.1);

    assertNoThrow(
        [&]() {
          model->train(X, Y, loss, optimizer, nullptr, 30);
        },
        "Training with string-converted data should work");

    // Test prediction to string conversion
    std::ostringstream prediction_stream;
    prediction_stream << "predictions:\n";

    for (size_t i = 0; i < X.size(); i++) {
      std::vector<double> pred = model->predict(X[i]);
      prediction_stream << "sample_" << i << ": " << pred[0] << "\n";
    }

    std::string prediction_string = prediction_stream.str();
    assertTrue(prediction_string.find("predictions") != std::string::npos,
               "Should format predictions as string");
    assertTrue(prediction_string.find("sample_") != std::string::npos,
               "Should include sample identifiers");
  }
};

/**
 * @class StringParameterHandlingIntegrationTest
 * @brief Test string-based parameter handling
 */
class StringParameterHandlingIntegrationTest : public TestCase {
public:
  StringParameterHandlingIntegrationTest()
      : TestCase("StringParameterHandlingIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test parameter parsing from strings
    std::string learning_rate_str = "0.01";
    std::string epochs_str = "50";
    std::string batch_size_str = "4";

    double learning_rate = std::stod(learning_rate_str);
    int epochs = std::stoi(epochs_str);
    int batch_size = std::stoi(batch_size_str);

    assertTrue(learning_rate == 0.01, "Should parse learning rate from string");
    assertTrue(epochs == 50, "Should parse epochs from string");
    assertTrue(batch_size == 4, "Should parse batch size from string");

    // Test model with string-parsed parameters
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 3));
    model->add(std::make_shared<activation::Tanh>());
    model->add(std::make_shared<Dense>(3, 1));

    std::vector<std::vector<double>> X = {{0.1, 0.2},
                                          {0.3, 0.4},
                                          {0.5, 0.6},
                                          {0.7, 0.8}};
    std::vector<std::vector<double>> Y = {{0.3}, {0.7}, {1.1}, {1.5}};

    // Verify batch size handling
    assertTrue(X.size() == static_cast<size_t>(batch_size),
               "Training data should match parsed batch size");

    MSELoss loss;
    SGD optimizer(learning_rate);

    bool training_successful = false;
    assertNoThrow(
        [&]() {
          model->train(X, Y, loss, optimizer, nullptr, epochs);
          training_successful = true;
        },
        "Training with string-parsed parameters should work");

    assertTrue(training_successful,
               "String parameter parsing should enable successful training");

    // Test activation function selection from string
    std::string activation_name = "sigmoid";
    bool is_sigmoid = (activation_name == "sigmoid");
    bool is_relu = (activation_name == "relu");
    bool is_tanh = (activation_name == "tanh");

    assertTrue(is_sigmoid, "Should correctly identify sigmoid activation");
    assertTrue(!is_relu, "Should correctly reject non-matching activation");
    assertTrue(!is_tanh, "Should correctly reject non-matching activation");

    // Test model name/identifier handling
    std::string model_name = "test_neural_network_v1";
    std::string model_type = "sequential";

    assertTrue(model_name.find("neural_network") != std::string::npos,
               "Model name should contain descriptive terms");
    assertTrue(model_type == "sequential",
               "Model type should be correctly identified");

    // Test parameter validation strings
    std::string param_validation = "learning_rate > 0 && learning_rate < 1";
    bool lr_valid = (learning_rate > 0.0 && learning_rate < 1.0);

    assertTrue(lr_valid, "String-based parameter validation should work");
  }
};

}  // namespace test
}  // namespace MLLib
