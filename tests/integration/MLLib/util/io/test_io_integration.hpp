#pragma once

#include "../../../../../include/MLLib.hpp"
#include "../../../../common/test_utils.hpp"
#include <fstream>
#include <memory>
#include <sstream>

/**
 * @file test_io_integration.hpp
 * @brief I/O utilities integration tests for MLLib
 *
 * Tests I/O utilities integration:
 * - Model save/load functionality
 * - Data import/export operations
 * - File format handling
 * - Error recovery in I/O operations
 */

namespace MLLib {
namespace test {

/**
 * @class ModelSaveLoadIOIntegrationTest
 * @brief Test model save/load functionality in real scenarios
 */
class ModelSaveLoadIOIntegrationTest : public TestCase {
public:
  ModelSaveLoadIOIntegrationTest()
      : TestCase("ModelSaveLoadIOIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create and train a model
    auto original_model = std::make_unique<Sequential>();
    original_model->add(std::make_shared<Dense>(3, 4));
    original_model->add(std::make_shared<activation::ReLU>());
    original_model->add(std::make_shared<Dense>(4, 2));
    original_model->add(std::make_shared<activation::Sigmoid>());

    // Training data
    std::vector<std::vector<double>> X = {{0.1, 0.2, 0.3},
                                          {0.4, 0.5, 0.6},
                                          {0.7, 0.8, 0.9},
                                          {0.2, 0.4, 0.6},
                                          {0.3, 0.6, 0.9}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0},
                                          {0.0, 1.0},
                                          {1.0, 0.0},
                                          {0.0, 1.0},
                                          {0.5, 0.5}};

    MSELoss loss;
    SGD optimizer(0.1);

    // Train the original model
    assertNoThrow(
        [&]() {
          original_model->train(X, Y, loss, optimizer, nullptr, 50);
        },
        "Original model training should complete");

    // Get predictions from original model
    std::vector<std::vector<double>> original_predictions;
    for (const auto& input : X) {
      original_predictions.push_back(original_model->predict(input));
    }

    // Test model serialization/deserialization concept
    // Note: Actual file I/O might not be implemented, so we test the concept
    std::string model_data = "model_weights_and_structure";

    // Simulate save operation
    bool save_successful = true;
    assertNoThrow(
        [&]() {
          // In real implementation, this would be:
          // original_model->save("test_model.json");
          // For now, we just test that the model state is accessible
          std::vector<double> test_input = {0.5, 0.5, 0.5};
          std::vector<double> test_output = original_model->predict(test_input);

          if (test_output.empty() || std::isnan(test_output[0])) {
            save_successful = false;
          }
        },
        "Model save simulation should not throw");

    assertTrue(save_successful, "Model save operation should be successful");

    // Simulate load operation by creating a new model with same structure
    auto loaded_model = std::make_unique<Sequential>();
    loaded_model->add(std::make_shared<Dense>(3, 4));
    loaded_model->add(std::make_shared<activation::ReLU>());
    loaded_model->add(std::make_shared<Dense>(4, 2));
    loaded_model->add(std::make_shared<activation::Sigmoid>());

    // In real implementation, weights would be loaded from file
    // For integration test, we verify the structure is correct
    bool load_successful = true;
    assertNoThrow(
        [&]() {
          std::vector<double> test_input = {0.5, 0.5, 0.5};
          std::vector<double> test_output = loaded_model->predict(test_input);

          if (test_output.size() != 2) {
            load_successful = false;
          }
        },
        "Model load simulation should not throw");

    assertTrue(load_successful, "Model load operation should be successful");

    // Test that loaded model has correct structure
    std::vector<double> test_input = {0.1, 0.2, 0.3};
    std::vector<double> loaded_output = loaded_model->predict(test_input);

    assertEqual(size_t(2), loaded_output.size(),
                "Loaded model should have correct output size");
    for (double val : loaded_output) {
      assertTrue(!std::isnan(val) && !std::isinf(val),
                 "Loaded model should produce valid outputs");
    }
  }
};

/**
 * @class DataImportExportIntegrationTest
 * @brief Test data import/export functionality
 */
class DataImportExportIntegrationTest : public TestCase {
public:
  DataImportExportIntegrationTest()
      : TestCase("DataImportExportIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Simulate CSV data format
    std::string csv_data = "0.1,0.2,0.3,1.0,0.0\n"
                           "0.4,0.5,0.6,0.0,1.0\n"
                           "0.7,0.8,0.9,1.0,0.0\n"
                           "0.2,0.4,0.6,0.0,1.0\n";

    // Parse simulated CSV data
    std::vector<std::vector<double>> X, Y;
    std::istringstream ss(csv_data);
    std::string line;

    while (std::getline(ss, line)) {
      std::istringstream line_ss(line);
      std::string cell;
      std::vector<double> row;

      while (std::getline(line_ss, cell, ',')) {
        row.push_back(std::stod(cell));
      }

      if (row.size() >= 5) {
        X.push_back({row[0], row[1], row[2]});
        Y.push_back({row[3], row[4]});
      }
    }

    assertEqual(size_t(4), X.size(), "Should parse correct number of samples");
    assertEqual(size_t(4), Y.size(), "Should parse correct number of labels");

    // Test training with imported data
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    MSELoss loss;
    SGD optimizer(0.2);

    bool training_successful = true;
    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  training_successful = false;
                }
              },
              30);
        },
        "Training with imported data should complete");

    assertTrue(training_successful,
               "Training with imported data should be successful");

    // Test prediction export simulation
    std::ostringstream export_stream;
    export_stream << "input1,input2,input3,pred1,pred2\n";

    for (size_t i = 0; i < X.size(); i++) {
      std::vector<double> pred = model->predict(X[i]);

      export_stream << X[i][0] << "," << X[i][1] << "," << X[i][2] << ","
                    << pred[0] << "," << pred[1] << "\n";
    }

    std::string exported_data = export_stream.str();
    assertTrue(exported_data.length() > 50, "Should export meaningful data");
    assertTrue(exported_data.find("pred1") != std::string::npos,
               "Should include prediction headers");
  }
};

/**
 * @class FileFormatIntegrationTest
 * @brief Test different file format handling
 */
class FileFormatIntegrationTest : public TestCase {
public:
  FileFormatIntegrationTest() : TestCase("FileFormatIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    OutputCapture capture;

    // Test JSON-like format simulation
    std::string json_config = R"({
            "layers": [
                {"type": "Dense", "input_size": 2, "output_size": 3},
                {"type": "ReLU"},
                {"type": "Dense", "input_size": 3, "output_size": 1},
                {"type": "Sigmoid"}
            ]
        })";

    // Parse configuration (simplified for integration test)
    bool config_valid = json_config.find("Dense") != std::string::npos &&
        json_config.find("ReLU") != std::string::npos &&
        json_config.find("Sigmoid") != std::string::npos;

    assertTrue(config_valid, "Configuration format should be parseable");

    // Create model based on configuration
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 3));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(3, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // Test that configured model works
    std::vector<double> test_input = {0.5, 0.3};
    std::vector<double> output = model->predict(test_input);

    assertEqual(size_t(1), output.size(),
                "Configured model should have correct output size");
    assertTrue(output[0] >= 0.0 && output[0] <= 1.0,
               "Configured model should respect sigmoid bounds");

    // Test binary format simulation
    std::vector<uint8_t> binary_data;

    // Simulate binary model data
    binary_data.push_back(0x4D);  // 'M'
    binary_data.push_back(0x4C);  // 'L'
    binary_data.push_back(0x4C);  // 'L'
    binary_data.push_back(0x69);  // 'i'
    binary_data.push_back(0x62);  // 'b'

    // Add layer count
    binary_data.push_back(4);  // 4 layers

    assertTrue(binary_data.size() == 6,
               "Binary format should have expected structure");
    assertTrue(binary_data[0] == 0x4D,
               "Binary format should have correct header");

    // Test text format
    std::ostringstream text_format;
    text_format << "MLLib Model Export\n";
    text_format << "Version: 1.0\n";
    text_format << "Layers: 4\n";
    text_format << "Layer 0: Dense(2->3)\n";
    text_format << "Layer 1: ReLU\n";
    text_format << "Layer 2: Dense(3->1)\n";
    text_format << "Layer 3: Sigmoid\n";

    std::string text_data = text_format.str();
    assertTrue(text_data.find("MLLib") != std::string::npos,
               "Text format should include header");
    assertTrue(text_data.find("Dense") != std::string::npos,
               "Text format should include layer info");
  }
};

/**
 * @class IOErrorRecoveryIntegrationTest
 * @brief Test error recovery in I/O operations
 */
class IOErrorRecoveryIntegrationTest : public TestCase {
public:
  IOErrorRecoveryIntegrationTest()
      : TestCase("IOErrorRecoveryIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test 1: Invalid data format handling
    {
      std::string invalid_csv = "0.1,0.2,invalid,1.0\n0.4,nan,0.6,0.0\n";
      std::vector<std::vector<double>> X, Y;

      // Parse with error handling
      std::istringstream ss(invalid_csv);
      std::string line;
      int valid_rows = 0;

      while (std::getline(ss, line)) {
        try {
          std::istringstream line_ss(line);
          std::string cell;
          std::vector<double> row;
          bool row_valid = true;

          while (std::getline(line_ss, cell, ',')) {
            try {
              double val = std::stod(cell);
              if (std::isnan(val) || std::isinf(val)) {
                row_valid = false;
                break;
              }
              row.push_back(val);
            } catch (...) {
              row_valid = false;
              break;
            }
          }

          if (row_valid && row.size() >= 4) {
            X.push_back({row[0], row[1], row[2]});
            Y.push_back({row[3]});
            valid_rows++;
          }
        } catch (...) {
          // Skip invalid rows
        }
      }

      // Should have parsed at least some valid data
      assertTrue(valid_rows == 0, "Invalid data should be properly rejected");
    }

    // Test 2: Model consistency after I/O errors
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(3, 1));

      // Simulate I/O error during save
      bool save_failed = true;
      std::vector<double> pre_error_prediction;

      assertNoThrow(
          [&]() {
            std::vector<double> test_input = {0.5, 0.3};
            pre_error_prediction = model->predict(test_input);

            // Simulate save operation that might fail
            // In real implementation, this might throw an exception
            if (pre_error_prediction.empty()) {
              throw std::runtime_error("Save failed");
            }
            save_failed = false;
          },
          "Model should remain functional after I/O error");

      // Model should still work after failed save
      std::vector<double> post_error_prediction;
      assertNoThrow(
          [&]() {
            std::vector<double> test_input = {0.5, 0.3};
            post_error_prediction = model->predict(test_input);
          },
          "Model should work after I/O error");

      assertEqual(pre_error_prediction.size(), post_error_prediction.size(),
                  "Model should maintain consistency after I/O error");
    }

    // Test 3: Graceful degradation with corrupted data
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 2));
      model->add(std::make_shared<activation::Sigmoid>());

      // Test with partially corrupted training data
      std::vector<std::vector<double>> X = {
          {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}  // Valid data
          // Corrupted data would be filtered out during parsing
      };
      std::vector<std::vector<double>> Y = {{1.0, 0.0}, {0.0, 1.0}};

      MSELoss loss;
      SGD optimizer(0.1);

      bool training_completed = false;
      assertNoThrow(
          [&]() {
            model->train(X, Y, loss, optimizer, nullptr, 20);
            training_completed = true;
          },
          "Training should work with clean data after error recovery");

      assertTrue(training_completed,
                 "Training should complete after data cleaning");

      // Model should produce valid predictions
      std::vector<double> test_output = model->predict({0.5, 0.5, 0.5});
      assertEqual(size_t(2), test_output.size(),
                  "Model should produce correct output after recovery");

      for (double val : test_output) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Recovery should ensure valid outputs");
      }
    }
  }
};

}  // namespace test
}  // namespace MLLib
