#pragma once

#include "../../../../../include/MLLib.hpp"
#include "../../../../common/test_utils.hpp"
#include <chrono>
#include <fstream>
#include <memory>
#include <string>

/**
 * @file test_misc_integration.hpp
 * @brief Miscellaneous utilities integration tests for MLLib
 *
 * Tests miscellaneous utilities integration:
 * - Matrix utilities in training scenarios
 * - Random utilities for initialization
 * - Validation utilities
 * - General purpose utilities
 */

namespace MLLib {
namespace test {

/**
 * @class MatrixUtilIntegrationTest
 * @brief Test matrix utilities in real training scenarios
 */
class MatrixUtilIntegrationTest : public TestCase {
public:
  MatrixUtilIntegrationTest() : TestCase("MatrixUtilIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test matrix operations in real training context
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    // Training data that exercises matrix operations
    std::vector<std::vector<double>> X = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},    {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}, {0.33, 0.33, 0.33}, {1.0, 1.0, 1.0}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0},
                                          {0.5, 0.5}, {0.8, 0.2}, {0.2, 0.8},
                                          {0.6, 0.4}, {0.3, 0.7}};

    MSELoss loss;
    SGD optimizer(0.1);

    bool training_stable = true;
    double max_loss = 0.0;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  training_stable = false;
                }
                max_loss = std::max(max_loss, current_loss);
              },
              100);
        },
        "Matrix util training should complete");

    assertTrue(training_stable, "Matrix operations should remain stable");
    assertTrue(max_loss < 100.0,
               "Matrix operations should produce reasonable values");

    // Test prediction accuracy (matrix operations should work correctly)
    for (const auto& input : X) {
      std::vector<double> pred = model->predict(input);
      assertEqual(size_t(2), pred.size(),
                  "Matrix operations should preserve dimensions");

      for (double val : pred) {
        assertTrue(val >= 0.0 && val <= 1.0,
                   "Matrix operations should respect sigmoid bounds");
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Matrix operations should produce valid results");
      }
    }
  }
};

/**
 * @class RandomUtilIntegrationTest
 * @brief Test random utilities in initialization and training
 */
class RandomUtilIntegrationTest : public TestCase {
public:
  RandomUtilIntegrationTest() : TestCase("RandomUtilIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test that different random initializations lead to different models
    std::vector<std::unique_ptr<Sequential>> models;
    std::vector<std::vector<double>> initial_predictions;

    std::vector<double> test_input = {0.5, 0.3, 0.8};

    // Create multiple models with random initialization
    for (int i = 0; i < 3; i++) {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 5));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(5, 2));
      model->add(std::make_shared<activation::Sigmoid>());

      // Get initial prediction (should be different due to random
      // initialization)
      std::vector<double> pred = model->predict(test_input);
      initial_predictions.push_back(pred);
      models.push_back(std::move(model));
    }

    // Verify that random initialization produces different results
    bool different_initializations = false;
    for (size_t i = 0; i < initial_predictions.size(); i++) {
      for (size_t j = i + 1; j < initial_predictions.size(); j++) {
        for (size_t k = 0; k < initial_predictions[i].size(); k++) {
          if (std::abs(initial_predictions[i][k] - initial_predictions[j][k]) >
              1e-6) {
            different_initializations = true;
            break;
          }
        }
        if (different_initializations) break;
      }
      if (different_initializations) break;
    }

    assertTrue(different_initializations,
               "Random initialization should produce different models");

    // Train one model to verify random initialization allows learning
    std::vector<std::vector<double>> X = {
        {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}};

    MSELoss loss;
    SGD optimizer(0.1);

    assertNoThrow(
        [&]() { models[0]->train(X, Y, loss, optimizer, nullptr, 50); },
        "Randomly initialized model should train successfully");

    // Verify final predictions are valid
    for (const auto& input : X) {
      std::vector<double> pred = models[0]->predict(input);
      for (double val : pred) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Random init model should produce valid predictions");
      }
    }
  }
};

/**
 * @class ValidationUtilIntegrationTest
 * @brief Test validation utilities in training scenarios
 */
class ValidationUtilIntegrationTest : public TestCase {
public:
  ValidationUtilIntegrationTest() : TestCase("ValidationUtilIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test validation during training
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // Training data
    std::vector<std::vector<double>> X_train = {
        {0.1, 0.9}, {0.9, 0.1}, {0.3, 0.7}, {0.7, 0.3}};
    std::vector<std::vector<double>> Y_train = {{1.0}, {0.0}, {1.0}, {0.0}};

    // Validation data
    std::vector<std::vector<double>> X_val = {{0.2, 0.8}, {0.8, 0.2}};
    std::vector<std::vector<double>> Y_val = {{1.0}, {0.0}};

    MSELoss loss;
    SGD optimizer(0.2);

    std::vector<double> training_losses;
    std::vector<double> validation_accuracies;

    assertNoThrow(
        [&]() {
          model->train(
              X_train, Y_train, loss, optimizer,
              [&](int epoch, double current_loss) {
                training_losses.push_back(current_loss);

                // Validate model performance
                double correct_predictions = 0.0;
                for (size_t i = 0; i < X_val.size(); i++) {
                  std::vector<double> pred = model->predict(X_val[i]);
                  double prediction = pred[0] > 0.5 ? 1.0 : 0.0;
                  if (std::abs(prediction - Y_val[i][0]) < 0.1) {
                    correct_predictions += 1.0;
                  }
                }
                double accuracy = correct_predictions / X_val.size();
                validation_accuracies.push_back(accuracy);
              },
              100);
        },
        "Training with validation should complete");

    assertTrue(training_losses.size() > 0, "Should record training losses");
    assertTrue(validation_accuracies.size() > 0,
               "Should record validation accuracies");

    // Validation should show reasonable performance
    bool found_decent_accuracy = false;
    for (double acc : validation_accuracies) {
      if (acc >= 0.5) {  // At least random performance
        found_decent_accuracy = true;
        break;
      }
    }

    assertTrue(found_decent_accuracy,
               "Validation should show learning progress");

    // Test edge case validation
    std::vector<double> edge_input1 = {0.0, 0.0};
    std::vector<double> edge_input2 = {1.0, 1.0};

    std::vector<double> pred1 = model->predict(edge_input1);
    std::vector<double> pred2 = model->predict(edge_input2);

    assertTrue(!std::isnan(pred1[0]) && !std::isinf(pred1[0]),
               "Edge case 1 should be valid");
    assertTrue(!std::isnan(pred2[0]) && !std::isinf(pred2[0]),
               "Edge case 2 should be valid");
  }
};

/**
 * @class MiscUtilIntegrationTest
 * @brief Test miscellaneous utilities in various scenarios
 */
class MiscUtilIntegrationTest : public TestCase {
public:
  MiscUtilIntegrationTest() : TestCase("MiscUtilIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test 1: Model configuration utilities
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(3, 5));
      model->add(std::make_shared<activation::ReLU>());
      model->add(std::make_shared<Dense>(5, 2));
      model->add(std::make_shared<activation::Sigmoid>());

      // Test model has expected structure
      assertTrue(model->num_layers() >= 4,
                 "Model should have expected number of layers");

      // Test prediction utilities
      std::vector<double> input = {0.5, 0.2, 0.8};
      std::vector<double> output = model->predict(input);

      assertEqual(size_t(2), output.size(),
                  "Prediction utilities should work correctly");
    }

    // Test 2: Training utilities
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::Tanh>());
      model->add(std::make_shared<Dense>(3, 1));

      std::vector<std::vector<double>> X = {
          {-1.0, 1.0}, {1.0, -1.0}, {0.0, 0.0}};
      std::vector<std::vector<double>> Y = {{1.0}, {-1.0}, {0.0}};

      MSELoss loss;
      SGD optimizer(0.1);

      int epoch_count = 0;
      double final_loss = 0.0;

      assertNoThrow(
          [&]() {
            model->train(
                X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                  epoch_count++;
                  final_loss = current_loss;
                },
                30);
          },
          "Training utilities should work correctly");

      assertTrue(epoch_count > 0, "Training callback utilities should work");
      assertTrue(!std::isnan(final_loss),
                 "Loss computation utilities should work");
    }

    // Test 3: Memory management utilities
    {
      // Create multiple models to test memory handling
      std::vector<std::unique_ptr<Sequential>> models;

      for (int i = 0; i < 5; i++) {
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(4, 6));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(6, 3));

        // Quick test to ensure model works
        std::vector<double> test_input = {0.1, 0.2, 0.3, 0.4};
        std::vector<double> output = model->predict(test_input);

        assertTrue(output.size() == 3,
                   "Memory management should maintain functionality");
        models.push_back(std::move(model));
      }

      // All models should work independently
      for (auto& model : models) {
        std::vector<double> test_input = {0.5, 0.6, 0.7, 0.8};
        std::vector<double> output = model->predict(test_input);

        for (double val : output) {
          assertTrue(!std::isnan(val) && !std::isinf(val),
                     "Memory utilities should preserve validity");
        }
      }
    }

    // Test 4: Error handling utilities
    {
      auto model = std::make_unique<Sequential>();
      model->add(std::make_shared<Dense>(2, 3));
      model->add(std::make_shared<activation::Sigmoid>());

      // Test with valid inputs
      std::vector<double> valid_input = {0.5, 0.3};
      std::vector<double> output;

      assertNoThrow([&]() { output = model->predict(valid_input); },
                    "Error handling should allow valid operations");

      assertEqual(size_t(3), output.size(),
                  "Error handling should preserve correct behavior");

      // Test with edge case inputs
      std::vector<double> zero_input = {0.0, 0.0};
      std::vector<double> large_input = {100.0, 100.0};
      std::vector<double> negative_input = {-50.0, -50.0};

      assertNoThrow(
          [&]() {
            model->predict(zero_input);
            model->predict(large_input);
            model->predict(negative_input);
          },
          "Error handling should manage edge cases gracefully");
    }
  }
};

}  // namespace test
}  // namespace MLLib
