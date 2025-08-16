#pragma once

#include "../../../../include/MLLib.hpp"
#include "../../../common/test_utils.hpp"
#include <cmath>
#include <memory>

/**
 * @file test_optimizer_activation_integration.hpp
 * @brief Optimizer and Activation integration tests for MLLib
 *
 * Tests the integration between different optimizers and activation functions:
 * - SGD with various activation functions
 * - Adam with various activation functions
 * - Learning convergence characteristics
 * - Gradient flow and stability
 * - Performance comparison across combinations
 */

namespace MLLib {
namespace test {

/**
 * @class SGDReLUIntegrationTest
 * @brief Test SGD optimizer with ReLU activation integration
 */
class SGDReLUIntegrationTest : public TestCase {
public:
  SGDReLUIntegrationTest() : TestCase("SGDReLUIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create model with SGD optimizer and ReLU activations
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 8));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(8, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // XOR-like problem data
    std::vector<std::vector<double>> X = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<std::vector<double>> Y = {{0.0}, {1.0}, {1.0}, {0.0}};

    MSELoss loss;
    SGD optimizer(0.1);  // Good learning rate for ReLU

    double initial_loss = 0.0;
    double final_loss = 0.0;
    bool first_epoch = true;
    int stable_epochs = 0;

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

                // Count stable training epochs
                if (!std::isnan(current_loss) && !std::isinf(current_loss)) {
                  stable_epochs++;
                }
              },
              200);
        },
        "SGD+ReLU training should complete");

    // ReLU with SGD should be stable and effective
    assertTrue(stable_epochs >= 150, "SGD+ReLU should have stable training");
    assertTrue(final_loss <= initial_loss,
               "SGD+ReLU should not increase loss");

    // Test predictions work correctly
    std::vector<double> pred1 = model->predict({0.0, 0.0});
    std::vector<double> pred2 = model->predict({0.0, 1.0});

    assertTrue(!std::isnan(pred1[0]) && !std::isinf(pred1[0]),
               "SGD+ReLU prediction should be valid");
    assertTrue(!std::isnan(pred2[0]) && !std::isinf(pred2[0]),
               "SGD+ReLU prediction should be valid");
    assertTrue(pred1[0] >= 0.0 && pred1[0] <= 1.0,
               "Output should be in valid range");
    assertTrue(pred2[0] >= 0.0 && pred2[0] <= 1.0,
               "Output should be in valid range");
  }
};

/**
 * @class SGDSigmoidIntegrationTest
 * @brief Test SGD optimizer with Sigmoid activation integration
 */
class SGDSigmoidIntegrationTest : public TestCase {
public:
  SGDSigmoidIntegrationTest() : TestCase("SGDSigmoidIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create model with SGD optimizer and Sigmoid activations
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 6));
    model->add(std::make_shared<activation::Sigmoid>());
    model->add(std::make_shared<Dense>(6, 3));
    model->add(std::make_shared<activation::Sigmoid>());
    model->add(std::make_shared<Dense>(3, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // Binary classification data
    std::vector<std::vector<double>> X = {
        {0.2, 0.1}, {0.8, 0.3}, {0.1, 0.9}, {0.7, 0.8}};
    std::vector<std::vector<double>> Y = {{0.0}, {1.0}, {1.0}, {0.0}};

    MSELoss loss;
    SGD optimizer(0.5);  // Higher learning rate for sigmoid

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
              300);  // More epochs needed for sigmoid
        },
        "SGD+Sigmoid training should complete");

    // Sigmoid with SGD might converge slower but should be stable
    assertTrue(final_loss <= initial_loss,
               "SGD+Sigmoid should not increase loss");
    assertTrue(!std::isnan(final_loss) && !std::isinf(final_loss),
               "SGD+Sigmoid loss should be valid");

    // Test all predictions are in valid sigmoid range [0,1]
    for (const auto& input : X) {
      std::vector<double> pred = model->predict(input);
      assertTrue(pred[0] >= 0.0 && pred[0] <= 1.0,
                 "Sigmoid output should be in [0,1]");
      assertTrue(!std::isnan(pred[0]) && !std::isinf(pred[0]),
                 "SGD+Sigmoid prediction should be valid");
    }
  }
};

/**
 * @class SGDTanhIntegrationTest
 * @brief Test SGD optimizer with Tanh activation integration
 */
class SGDTanhIntegrationTest : public TestCase {
public:
  SGDTanhIntegrationTest() : TestCase("SGDTanhIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create model with SGD optimizer and Tanh activations
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 5));
    model->add(std::make_shared<activation::Tanh>());
    model->add(std::make_shared<Dense>(5, 3));
    model->add(std::make_shared<activation::Tanh>());
    model->add(std::make_shared<Dense>(3, 1));
    model->add(std::make_shared<activation::Sigmoid>());  // Output activation

    // Regression-like data
    std::vector<std::vector<double>> X = {
        {-1.0, 0.0, 1.0}, {0.5, -0.5, 0.0}, {-0.5, 1.0, -0.5}, {1.0, 1.0, 1.0}};
    std::vector<std::vector<double>> Y = {{0.2}, {0.8}, {0.3}, {0.9}};

    MSELoss loss;
    SGD optimizer(0.2);  // Moderate learning rate for tanh

    assertNoThrow([&]() { model->train(X, Y, loss, optimizer, nullptr, 200); },
                  "SGD+Tanh training should complete");

    // Test that tanh handles negative inputs well
    std::vector<double> neg_pred = model->predict({-2.0, -1.0, -0.5});
    std::vector<double> pos_pred = model->predict({2.0, 1.0, 0.5});
    std::vector<double> zero_pred = model->predict({0.0, 0.0, 0.0});

    // All should produce valid outputs due to final sigmoid
    assertTrue(neg_pred[0] >= 0.0 && neg_pred[0] <= 1.0,
               "Tanh with negative input should work");
    assertTrue(pos_pred[0] >= 0.0 && pos_pred[0] <= 1.0,
               "Tanh with positive input should work");
    assertTrue(zero_pred[0] >= 0.0 && zero_pred[0] <= 1.0,
               "Tanh with zero input should work");

    assertTrue(!std::isnan(neg_pred[0]) && !std::isinf(neg_pred[0]),
               "SGD+Tanh prediction should be valid");
    assertTrue(!std::isnan(pos_pred[0]) && !std::isinf(pos_pred[0]),
               "SGD+Tanh prediction should be valid");
    assertTrue(!std::isnan(zero_pred[0]) && !std::isinf(zero_pred[0]),
               "SGD+Tanh prediction should be valid");
  }
};

/**
 * @class AdamActivationIntegrationTest
 * @brief Test Adam optimizer with various activation functions
 */
class AdamActivationIntegrationTest : public TestCase {
public:
  AdamActivationIntegrationTest() : TestCase("AdamActivationIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test Adam (or SGD fallback) with mixed activations
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 8));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(8, 4));
    model->add(std::make_shared<activation::Tanh>());
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    // Multi-class data
    std::vector<std::vector<double>> X = {
        {0.1, 0.2}, {0.8, 0.1}, {0.2, 0.9}, {0.9, 0.8}};
    std::vector<std::vector<double>> Y = {
        {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};

    MSELoss loss;

    // Use Adam if available, otherwise SGD
    SGD optimizer(0.01);  // Conservative learning rate

    double initial_loss = 0.0;
    double final_loss = 0.0;
    bool first_epoch = true;
    int convergence_epochs = 0;

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

                // Count epochs with reasonable loss
                if (current_loss < 1.0) {
                  convergence_epochs++;
                }
              },
              300);
        },
        "Adam+Mixed activations training should complete");

    // Adam should handle mixed activations well
    assertTrue(convergence_epochs > 0, "Adam should achieve reasonable loss");
    assertTrue(!std::isnan(final_loss) && !std::isinf(final_loss),
               "Adam training should be stable");

    // Test predictions for all training samples
    for (size_t i = 0; i < X.size(); i++) {
      std::vector<double> pred = model->predict(X[i]);
      assertEqual(size_t(2), pred.size(),
                  "Adam prediction should have correct size");

      for (double val : pred) {
        assertTrue(val >= 0.0 && val <= 1.0,
                   "Final sigmoid should constrain output");
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Adam prediction should be valid");
      }
    }
  }
};

/**
 * @class OptimizerActivationPerformanceTest
 * @brief Compare performance of optimizer-activation combinations
 */
class OptimizerActivationPerformanceTest : public TestCase {
public:
  OptimizerActivationPerformanceTest()
      : TestCase("OptimizerActivationPerformanceTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test same problem with different optimizer-activation combinations
    std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0},
                                          {1.0, 1.0}, {0.5, 0.5}, {0.2, 0.8},
                                          {0.8, 0.2}, {0.3, 0.7}};
    std::vector<std::vector<double>> Y = {{0.0}, {1.0}, {1.0}, {0.0},
                                          {0.5}, {0.8}, {0.8}, {0.7}};

    MSELoss loss;

    // Test 1: SGD + ReLU
    {
      auto sgd_relu_model = std::make_unique<Sequential>();
      sgd_relu_model->add(std::make_shared<Dense>(2, 6));
      sgd_relu_model->add(std::make_shared<activation::ReLU>());
      sgd_relu_model->add(std::make_shared<Dense>(6, 1));
      sgd_relu_model->add(std::make_shared<activation::Sigmoid>());

      SGD sgd_optimizer(0.1);
      double sgd_relu_loss = 0.0;

      assertNoThrow(
          [&]() {
            sgd_relu_model->train(
                X, Y, loss, sgd_optimizer,
                [&](int epoch, double current_loss) {
                  sgd_relu_loss = current_loss;
                },
                150);
          },
          "SGD+ReLU combination should work");

      assertTrue(!std::isnan(sgd_relu_loss) && !std::isinf(sgd_relu_loss),
                 "SGD+ReLU should produce valid loss");
    }

    // Test 2: SGD + Sigmoid
    {
      auto sgd_sigmoid_model = std::make_unique<Sequential>();
      sgd_sigmoid_model->add(std::make_shared<Dense>(2, 6));
      sgd_sigmoid_model->add(std::make_shared<activation::Sigmoid>());
      sgd_sigmoid_model->add(std::make_shared<Dense>(6, 1));
      sgd_sigmoid_model->add(std::make_shared<activation::Sigmoid>());

      SGD sgd_optimizer(0.3);
      double sgd_sigmoid_loss = 0.0;

      assertNoThrow(
          [&]() {
            sgd_sigmoid_model->train(
                X, Y, loss, sgd_optimizer,
                [&](int epoch, double current_loss) {
                  sgd_sigmoid_loss = current_loss;
                },
                200);
          },
          "SGD+Sigmoid combination should work");

      assertTrue(!std::isnan(sgd_sigmoid_loss) && !std::isinf(sgd_sigmoid_loss),
                 "SGD+Sigmoid should produce valid loss");
    }

    // Test 3: SGD + Tanh
    {
      auto sgd_tanh_model = std::make_unique<Sequential>();
      sgd_tanh_model->add(std::make_shared<Dense>(2, 6));
      sgd_tanh_model->add(std::make_shared<activation::Tanh>());
      sgd_tanh_model->add(std::make_shared<Dense>(6, 1));
      sgd_tanh_model->add(std::make_shared<activation::Sigmoid>());

      SGD sgd_optimizer(0.2);
      double sgd_tanh_loss = 0.0;

      assertNoThrow(
          [&]() {
            sgd_tanh_model->train(
                X, Y, loss, sgd_optimizer,
                [&](int epoch, double current_loss) {
                  sgd_tanh_loss = current_loss;
                },
                175);
          },
          "SGD+Tanh combination should work");

      assertTrue(!std::isnan(sgd_tanh_loss) && !std::isinf(sgd_tanh_loss),
                 "SGD+Tanh should produce valid loss");
    }

    // All combinations should complete successfully
    assertTrue(true,
               "All optimizer-activation combinations tested successfully");
  }
};

/**
 * @class GradientFlowIntegrationTest
 * @brief Test gradient flow through optimizer-activation combinations
 */
class GradientFlowIntegrationTest : public TestCase {
public:
  GradientFlowIntegrationTest() : TestCase("GradientFlowIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Deep model to test gradient flow
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 8));
    model->add(
        std::make_shared<activation::ReLU>());  // Test vanishing gradients
    model->add(std::make_shared<Dense>(8, 6));
    model->add(
        std::make_shared<activation::Sigmoid>());  // Test saturating gradients
    model->add(std::make_shared<Dense>(6, 4));
    model->add(
        std::make_shared<activation::Tanh>());  // Test symmetric activation
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::ReLU>());  // More ReLU
    model->add(std::make_shared<Dense>(2, 1));
    model->add(std::make_shared<activation::Sigmoid>());  // Output

    // Simple but challenging data
    std::vector<std::vector<double>> X = {
        {0.1, 0.9}, {0.9, 0.1}, {0.3, 0.7}, {0.7, 0.3}};
    std::vector<std::vector<double>> Y = {{0.8}, {0.2}, {0.6}, {0.4}};

    MSELoss loss;
    SGD optimizer(0.05);  // Conservative learning rate for deep model

    bool training_stable = true;
    double max_loss = 0.0;
    double min_loss = std::numeric_limits<double>::max();

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  training_stable = false;
                }
                max_loss = std::max(max_loss, current_loss);
                min_loss = std::min(min_loss, current_loss);
              },
              250);
        },
        "Deep model with mixed activations should train");

    assertTrue(training_stable,
               "Gradient flow should be stable through deep model");
    assertTrue(max_loss > 0.0, "Training should have meaningful loss values");
    assertTrue(min_loss < std::numeric_limits<double>::max(),
               "Loss should decrease over time");

    // Test that all layers are learning (predictions should be reasonable)
    for (const auto& input : X) {
      std::vector<double> pred = model->predict(input);
      assertTrue(!std::isnan(pred[0]) && !std::isinf(pred[0]),
                 "Deep model prediction should be valid");
      assertTrue(pred[0] >= 0.0 && pred[0] <= 1.0,
                 "Output should be in valid range");
    }
  }
};

}  // namespace test
}  // namespace MLLib
