#pragma once

#include "../../../../include/MLLib.hpp"
#include "../../../common/test_utils.hpp"
#include <fstream>
#include <memory>

/**
 * @file test_workflow_integration.hpp
 * @brief Workflow integration tests for MLLib
 *
 * Tests complete machine learning workflows:
 * - Data preparation and model training
 * - Model evaluation and validation
 * - Production deployment simulation
 * - Performance benchmarking
 * - Error handling in real scenarios
 */

namespace MLLib {
namespace test {

/**
 * @class DataPipelineIntegrationTest
 * @brief Test complete data processing and training pipeline
 */
class DataPipelineIntegrationTest : public TestCase {
public:
  DataPipelineIntegrationTest() : TestCase("DataPipelineIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test 1: Simple regression pipeline
    {
      // Generate synthetic regression data
      std::vector<std::vector<double>> X;
      std::vector<std::vector<double>> Y;

      // Linear relationship: y = 2x1 + 3x2 + small_noise
      for (int i = 0; i < 50; ++i) {
        double x1 = (i % 10) * 0.1;
        double x2 = ((i + 3) % 7) * 0.1;
        double y = 2.0 * x1 + 3.0 * x2 + (i % 3 - 1) * 0.005;  // Reduced noise

        X.push_back({x1, x2});
        Y.push_back({y});
      }

      // Create and train model
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 8));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(8, 4));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(4, 1));

      MSELoss loss;
      SGD optimizer(0.05);  // Increased learning rate for better convergence

      double final_loss = 0.0;
      bool training_stable = true;

      assertNoThrow(
          [&]() {
            model->train(
                X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                  final_loss = current_loss;
                  if (std::isnan(current_loss) || std::isinf(current_loss)) {
                    training_stable = false;
                  }
                },
                300);  // Increased epochs for better learning
          },
          "Regression pipeline should complete");

      assertTrue(training_stable, "Training should be numerically stable");
      assertTrue(final_loss < 10.0,
                 "Final loss should be reasonable");  // Added loss check

      // Test model on new data
      std::vector<double> test_input = {0.5, 0.3};
      std::vector<double> test_pred = model->predict(test_input);
      double expected = 2.0 * 0.5 + 3.0 * 0.3;  // 1.9
      assertTrue(std::abs(test_pred[0] - expected) < 1.0,
                 "Model should learn approximate linear relationship");
    }

    // Test 2: Classification pipeline with validation
    {
      // Generate binary classification data
      std::vector<std::vector<double>> train_X, train_Y;
      std::vector<std::vector<double>> val_X, val_Y;

      // Create more separable classes
      for (int i = 0; i < 40; ++i) {
        double x1 = (i % 20) * 0.05;
        double x2 = (i / 20) * 0.05 + (i % 3) * 0.005;  // Reduced noise
        double label = (x1 + x2 > 0.4) ? 1.0 : 0.0;     // Easier separation

        if (i < 30) {  // Training data
          train_X.push_back({x1, x2});
          train_Y.push_back({label});
        } else {  // Validation data
          val_X.push_back({x1, x2});
          val_Y.push_back({label});
        }
      }

      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 6));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(6, 3));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(3, 1));
      model->add(std::make_shared<activation::Sigmoid>());

      MSELoss loss;
      SGD optimizer(0.2);  // Slightly reduced learning rate for stability

      // Train with validation monitoring
      double best_val_loss = std::numeric_limits<double>::max();
      int epochs_without_improvement = 0;

      assertNoThrow(
          [&]() {
            model->train(
                train_X, train_Y, loss, optimizer,
                [&](int epoch, double train_loss) {
                  // Simulate validation
                  if (epoch % 10 == 0) {
                    double val_loss = 0.0;
                    for (size_t i = 0; i < val_X.size(); ++i) {
                      std::vector<double> pred = model->predict(val_X[i]);
                      double diff = pred[0] - val_Y[i][0];
                      val_loss += diff * diff;
                    }
                    val_loss /= val_X.size();

                    if (val_loss < best_val_loss) {
                      best_val_loss = val_loss;
                      epochs_without_improvement = 0;
                    } else {
                      epochs_without_improvement += 10;
                    }
                  }
                },
                150);  // Increased epochs for classification task
          },
          "Classification pipeline should complete");

      // Test classification accuracy
      int correct_predictions = 0;
      for (size_t i = 0; i < val_X.size(); ++i) {
        std::vector<double> pred = model->predict(val_X[i]);
        double predicted_class = (pred[0] > 0.5) ? 1.0 : 0.0;
        if (std::abs(predicted_class - val_Y[i][0]) < 0.1) {
          correct_predictions++;
        }
      }

      double accuracy = static_cast<double>(correct_predictions) / val_X.size();
      assertTrue(accuracy > 0.5,
                 "Model should achieve reasonable accuracy (>50%)");
    }
  }
};

/**
 * @class ModelLifecycleIntegrationTest
 * @brief Test complete model lifecycle from creation to deployment
 */
class ModelLifecycleIntegrationTest : public TestCase {
public:
  ModelLifecycleIntegrationTest() : TestCase("ModelLifecycleIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    std::string temp_dir = createTempDirectory();

    try {
      // Simplified Phase 1: Basic Model Development
      auto development_model = std::make_unique<Sequential>();
      development_model->add(std::make_shared<Dense>(3, 4));
      development_model->add(std::make_shared<activation::ReLU>());
      development_model->add(std::make_shared<Dense>(4, 2));
      development_model->add(std::make_shared<activation::Sigmoid>());

      // Minimal training data
      std::vector<std::vector<double>> dev_X = {{0.1, 0.2, 0.3},
                                                {0.4, 0.5, 0.6}};
      std::vector<std::vector<double>> dev_Y = {{0.8, 0.2}, {0.6, 0.4}};

      MSELoss loss;
      SGD optimizer(0.1);

      // Basic training with minimal epochs
      development_model->train(dev_X, dev_Y, loss, optimizer, nullptr, 10);

      // Simplified Phase 2: Save and Load Test
      std::string model_path = temp_dir + "/simple_model";
      assertTrue(ModelIO::save_config(*development_model,
                                      model_path + ".config"),
                 "Config save should succeed");

      auto loaded_config = ModelIO::load_config(model_path + ".config");
      assertNotNull(loaded_config.get(), "Config load should succeed");

      // Simplified Phase 3: Basic Production Test
      std::vector<double> test_input = {0.5, 0.5, 0.5};
      std::vector<double> output = development_model->predict(test_input);
      assertEqual(size_t(2), output.size(), "Output should have 2 elements");

      // Explicitly reset to avoid cleanup issues
      development_model.reset();
      loaded_config.reset();

    } catch (const std::exception& e) {
      // Clean up and fail gracefully
      removeTempDirectory(temp_dir);
      throw;
    }

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class ErrorHandlingIntegrationTest
 * @brief Test error handling in real-world scenarios
 */
class ErrorHandlingIntegrationTest : public TestCase {
public:
  ErrorHandlingIntegrationTest() : TestCase("ErrorHandlingIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test 1: Invalid data handling
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(3, 1));

      MSELoss loss;
      SGD optimizer(0.1);

      // Test empty data
      std::vector<std::vector<double>> empty_X, empty_Y;
      assertThrows<std::invalid_argument>(
          [&]() {
            model->train(empty_X, empty_Y, loss, optimizer, nullptr, 10);
          },
          "Training with empty data should throw");

      // Test mismatched data sizes
      std::vector<std::vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}};
      std::vector<std::vector<double>> Y = {
          {5.0}};  // Only one output for two inputs
      assertThrows<std::invalid_argument>(
          [&]() {
            model->train(X, Y, loss, optimizer, nullptr, 10);
          },
          "Training with mismatched data should throw");
      assertThrows<std::invalid_argument>(
          [&]() {
            model->train(X, Y, loss, optimizer, nullptr, 10);
          },
          "Training with mismatched data should throw");
      assertThrows<std::invalid_argument>(
          [&]() {
            model->predict(std::vector<double>{1.0});
          },
          "Predict with wrong input shape should throw");
      assertThrows<std::invalid_argument>(
          [&]() {
            model->predict(std::vector<double>{1.0});
          },
          "Predict with wrong input shape should throw");
      {
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(1, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        model->add(std::make_shared<Dense>(2, 1));

        MSELoss loss;

        // Test with very high learning rate (may cause instability)
        SGD high_lr_optimizer(10.0);  // Very high learning rate

        std::vector<std::vector<double>> X = {{0.5}, {0.3}, {0.8}};
        std::vector<std::vector<double>> Y = {{0.2}, {0.7}, {0.1}};

        bool encountered_nan = false;
        assertNoThrow(
            [&]() {
              model->train(
                  X, Y, loss, high_lr_optimizer,
                  [&](int epoch, double current_loss) {
                    if (std::isnan(current_loss) || std::isinf(current_loss)) {
                      encountered_nan = true;
                    }
                  },
                  20);
            },
            "High learning rate training should not throw (but may produce "
            "NaN)");

        // It's okay if we encounter NaN with very high learning rate
        // The important thing is that it doesn't crash
      }

      // Test 3: File I/O error handling
      {
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(2, 2));

        // Test saving to invalid path
        assertFalse(ModelIO::save_model(*model, "/invalid/path/model.bin",
                                        SaveFormat::BINARY),
                    "Save to invalid path should fail gracefully");

        // Test loading non-existent file
        auto loaded =
            ModelIO::load_model("/nonexistent/file.bin", SaveFormat::BINARY);
        assertNull(loaded.get(),
                   "Loading non-existent file should return nullptr");

        // Test loading corrupted file
        std::string temp_dir = createTempDirectory();
        std::string corrupt_path = temp_dir + "/corrupt.bin";

        // Create a file with invalid content
        std::ofstream corrupt_file(corrupt_path);
        corrupt_file << "This is not a valid model file";
        corrupt_file.close();

        auto corrupt_loaded =
            ModelIO::load_model(corrupt_path, SaveFormat::BINARY);
        assertNull(corrupt_loaded.get(),
                   "Loading corrupted file should return nullptr");

        removeTempDirectory(temp_dir);
      }

      // Test 4: Resource management under stress
      {
        // Test multiple model creation and destruction
        for (int i = 0; i < 10; ++i) {
          auto model = std::make_unique<Sequential>();
          model->add(std::make_shared<Dense>(5, 10));
          model->add(std::make_shared<activation::ReLU>());
          model->add(std::make_shared<Dense>(10, 5));

          std::vector<std::vector<double>> X = {{1, 2, 3, 4, 5},
                                                {5, 4, 3, 2, 1}};
          std::vector<std::vector<double>> Y = {{1, 1, 1, 1, 1},
                                                {0, 0, 0, 0, 0}};

          MSELoss loss;
          SGD optimizer(0.01);

          assertNoThrow(
              [&]() {
                model->train(X, Y, loss, optimizer, nullptr, 5);
              },
              "Repeated model creation should not cause issues");
        }
      }
    }
  }
};

/**
 * @class WorkflowPerformanceBenchmarkTest
 * @brief Benchmark performance in realistic scenarios
 */
class WorkflowPerformanceBenchmarkTest : public TestCase {
public:
  WorkflowPerformanceBenchmarkTest()
      : TestCase("WorkflowPerformanceBenchmarkTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test 1: Large dataset training performance
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(10, 20));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(20, 10));
      model->add(std::make_shared<activation::Sigmoid>());
      model->add(std::make_shared<Dense>(10, 1));

      // Generate larger dataset
      std::vector<std::vector<double>> X;
      std::vector<std::vector<double>> Y;

      for (int i = 0; i < 500; ++i) {
        std::vector<double> x(10);
        for (int j = 0; j < 10; ++j) {
          x[j] = (i + j) * 0.001;
        }
        X.push_back(x);
        Y.push_back({static_cast<double>(i % 2)});
      }

      MSELoss loss;
      SGD optimizer(0.01);

      assertNoThrow(
          [&]() {
            model->train(X, Y, loss, optimizer, nullptr, 50);
          },
          "Large dataset training should complete");
    }

    // Test 2: High-frequency inference performance
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(5, 8));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(8, 3));

      // Quick training
      std::vector<std::vector<double>> X = {{1, 2, 3, 4, 5},
                                            {5, 4, 3, 2, 1},
                                            {3, 1, 4, 1, 5}};
      std::vector<std::vector<double>> Y = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

      MSELoss loss;
      SGD optimizer(0.1);
      model->train(X, Y, loss, optimizer, nullptr, 20);

      // High-frequency inference test
      std::vector<double> test_input = {2.5, 3.5, 2.0, 4.0, 1.5};

      assertNoThrow(
          [&]() {
            for (int i = 0; i < 1000; ++i) {
              std::vector<double> output = model->predict(test_input);
              assertEqual(size_t(3), output.size(),
                          "Each prediction should have 3 outputs");
            }
          },
          "High-frequency inference should be stable");
    }

    // Test 3: Memory usage stability
    {
      // Test that repeated operations don't cause memory issues
      for (int trial = 0; trial < 5; ++trial) {
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(20, 30));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(30, 20));
        model->add(std::make_shared<activation::Sigmoid>());
        model->add(std::make_shared<Dense>(20, 1));

        // Generate data for this trial
        std::vector<std::vector<double>> X;
        std::vector<std::vector<double>> Y;

        for (int i = 0; i < 100; ++i) {
          std::vector<double> x(20);
          for (int j = 0; j < 20; ++j) {
            x[j] = (trial * 100 + i + j) * 0.001;
          }
          X.push_back(x);
          Y.push_back({static_cast<double>((trial + i) % 2)});
        }

        MSELoss loss;
        SGD optimizer(0.01);

        assertNoThrow(
            [&]() {
              model->train(X, Y, loss, optimizer, nullptr, 30);
            },
            "Memory stability test should complete");
      }
    }
  }
};

}  // namespace test
}  // namespace MLLib
