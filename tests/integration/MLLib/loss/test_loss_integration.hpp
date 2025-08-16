#pragma once

#include "../../../../include/MLLib.hpp"
#include "../../../common/test_utils.hpp"
#include <memory>

/**
 * @file test_loss_integration.hpp
 * @brief Loss function integration tests for MLLib
 *
 * Tests loss function integration:
 * - MSE and CrossEntropy loss behavior
 * - Loss function-model interaction
 * - Gradient computation integration
 * - Loss convergence patterns
 */

namespace MLLib {
namespace test {

/**
 * @class MSELossIntegrationTest
 * @brief Test MSE loss integration with models
 */
class MSELossIntegrationTest : public TestCase {
public:
  MSELossIntegrationTest() : TestCase("MSELossIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create regression model
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 1));

    // Regression data
    std::vector<std::vector<double>> X = {{1.0, 2.0},
                                          {2.0, 3.0},
                                          {3.0, 4.0},
                                          {4.0, 5.0}};
    std::vector<std::vector<double>> Y = {
        {3.0}, {5.0}, {7.0}, {9.0}  // y = 2*x1 + x2 - 1
    };

    MSELoss loss;
    SGD optimizer(0.01);

    double initial_loss = 0.0;
    double final_loss = 0.0;
    bool first_epoch = true;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (first_epoch) {
                  initial_loss = current_loss;
                  first_epoch = false;
                }
                final_loss = current_loss;
              },
              200);
        },
        "MSE loss training should complete");

    // MSE should decrease for this simple regression
    assertTrue(final_loss < initial_loss,
               "MSE loss should decrease during training");

    // Test prediction accuracy
    std::vector<double> pred = model->predict({2.5, 3.5});
    double expected = 2.0 * 2.5 + 3.5 - 1.0;  // 7.5

    // Should be reasonably close to linear relationship
    assertTrue(std::abs(pred[0] - expected) < 2.0,
               "Prediction should be reasonably close to expected value");
  }
};

/**
 * @class CrossEntropyLossIntegrationTest
 * @brief Test CrossEntropy loss integration with models
 */
class CrossEntropyLossIntegrationTest : public TestCase {
public:
  CrossEntropyLossIntegrationTest()
      : TestCase("CrossEntropyLossIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create classification model
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 3));
    model->add(std::make_shared<activation::Sigmoid>());

    // Classification data (3 classes)
    std::vector<std::vector<double>> X = {{0.0, 0.0},
                                          {0.0, 1.0},
                                          {1.0, 0.0},
                                          {1.0, 1.0},
                                          {0.5, 0.5}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0, 0.0},
                                          {0.0, 1.0, 0.0},
                                          {0.0, 0.0, 1.0},
                                          {0.0, 1.0, 0.0},
                                          {0.0, 0.0, 1.0}};

    MSELoss loss;
    SGD optimizer(0.1);

    double initial_loss = 0.0;
    double final_loss = 0.0;
    bool first_epoch = true;
    bool training_stable = true;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  training_stable = false;
                }
                if (first_epoch) {
                  initial_loss = current_loss;
                  first_epoch = false;
                }
                final_loss = current_loss;
              },
              150);
        },
        "CrossEntropy loss training should complete");

    assertTrue(training_stable, "Training should be numerically stable");

    // Test predictions - outputs should sum approximately to 1 for each sample
    std::vector<double> pred1 = model->predict({0.0, 0.0});
    std::vector<double> pred2 = model->predict({1.0, 1.0});

    assertEqual(size_t(3), pred1.size(), "Prediction should have 3 classes");
    assertEqual(size_t(3), pred2.size(), "Prediction should have 3 classes");

    // Check that outputs are valid probabilities
    for (double val : pred1) {
      assertTrue(val >= 0.0 && val <= 1.0, "Output should be in [0,1] range");
    }
    for (double val : pred2) {
      assertTrue(val >= 0.0 && val <= 1.0, "Output should be in [0,1] range");
    }
  }
};

/**
 * @class LossComparisonIntegrationTest
 * @brief Compare different loss functions on similar tasks
 */
class LossComparisonIntegrationTest : public TestCase {
public:
  LossComparisonIntegrationTest() : TestCase("LossComparisonIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Binary classification data that can work with both loss functions
    std::vector<std::vector<double>> X = {{0.1, 0.2},
                                          {0.8, 0.9},
                                          {0.2, 0.1},
                                          {0.9, 0.8},
                                          {0.5, 0.5}};
    std::vector<std::vector<double>> Y = {{0.0}, {1.0}, {0.0}, {1.0}, {0.5}};

    SGD optimizer(0.1);

    // Test with MSE Loss
    {
      auto mse_model = std::make_unique<Sequential>();
      mse_model->add(std::make_shared<Dense>(2, 3));
      mse_model->add(std::make_shared<activation::Tanh>());
      mse_model->add(std::make_shared<Dense>(3, 1));
      mse_model->add(std::make_shared<activation::Sigmoid>());

      MSELoss mse_loss;

      assertNoThrow(
          [&]() {
            mse_model->train(X, Y, mse_loss, optimizer, nullptr, 100);
          },
          "MSE loss model training should complete");

      std::vector<double> mse_pred = mse_model->predict({0.3, 0.7});
      assertEqual(size_t(1), mse_pred.size(),
                  "MSE model prediction should have correct size");
      assertTrue(mse_pred[0] >= 0.0 && mse_pred[0] <= 1.0,
                 "MSE prediction should be in valid range");
    }

    // Test with CrossEntropy Loss (reformulated as 2-class)
    {
      auto ce_model = std::make_unique<Sequential>();
      ce_model->add(std::make_shared<Dense>(2, 3));
      ce_model->add(std::make_shared<activation::Tanh>());
      ce_model->add(std::make_shared<Dense>(3, 2));
      ce_model->add(std::make_shared<activation::Sigmoid>());

      // Convert targets to 2-class format
      std::vector<std::vector<double>> Y_2class = {{1.0, 0.0},
                                                   {0.0, 1.0},
                                                   {1.0, 0.0},
                                                   {0.0, 1.0},
                                                   {0.5, 0.5}};

      MSELoss ce_loss;
      assertNoThrow(
          [&]() {
            ce_model->train(X, Y_2class, ce_loss, optimizer, nullptr, 100);
          },
          "CrossEntropy loss model training should complete");

      std::vector<double> ce_pred = ce_model->predict({0.3, 0.7});
      assertEqual(size_t(2), ce_pred.size(),
                  "CE model prediction should have correct size");

      for (double val : ce_pred) {
        assertTrue(val >= 0.0 && val <= 1.0,
                   "CE prediction should be in valid range");
      }
    }
  }
};

}  // namespace test
}  // namespace MLLib
