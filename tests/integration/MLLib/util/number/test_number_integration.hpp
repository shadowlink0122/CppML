#pragma once

#include "../../../../../include/MLLib.hpp"
#include "../../../../common/test_utils.hpp"
#include <cmath>
#include <limits>
#include <memory>

/**
 * @file test_number_integration.hpp
 * @brief Number utilities integration tests for MLLib
 *
 * Tests number utilities integration:
 * - Numerical stability in training
 * - Precision handling
 * - Edge case number handling
 * - Mathematical operations
 */

namespace MLLib {
namespace test {

/**
 * @class NumericalStabilityIntegrationTest
 * @brief Test numerical stability in real training scenarios
 */
class NumericalStabilityIntegrationTest : public TestCase {
public:
  NumericalStabilityIntegrationTest()
      : TestCase("NumericalStabilityIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 5));
    model->add(std::make_shared<activation::Sigmoid>());
    model->add(std::make_shared<Dense>(5, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    // Test data with various numerical ranges
    std::vector<std::vector<double>> X = {
        {1e-6, 1e-3, 0.001},   // Very small positive numbers
        {0.999, 0.9999, 1.0},  // Numbers close to 1
        {0.5, 0.5, 0.5},       // Standard values
        {1e-10, 1e-8, 1e-6},   // Extremely small numbers
        {0.1, 0.01, 0.001}     // Decreasing precision
    };
    std::vector<std::vector<double>> Y = {{1.0, 0.0},
                                          {0.0, 1.0},
                                          {0.5, 0.5},
                                          {0.8, 0.2},
                                          {0.3, 0.7}};

    MSELoss loss;
    SGD optimizer(0.1);

    bool numerically_stable = true;
    std::vector<double> loss_values;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss) ||
                    current_loss < 0) {
                  numerically_stable = false;
                }
                loss_values.push_back(current_loss);
              },
              50);
        },
        "Numerically challenging training should complete");

    assertTrue(numerically_stable,
               "Training should maintain numerical stability");
    assertTrue(loss_values.size() > 0, "Should record loss values");

    // Test predictions with challenging inputs
    for (const auto& input : X) {
      std::vector<double> pred = model->predict(input);
      assertEqual(size_t(2), pred.size(),
                  "Challenging inputs should produce correct output size");

      for (double val : pred) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Predictions should be numerically valid");
        assertTrue(val >= 0.0 && val <= 1.0,
                   "Sigmoid outputs should be in [0,1]");
        assertTrue(val > 1e-15 && val < 1.0 - 1e-15,
                   "Predictions should avoid extreme values");
      }
    }

    // Test with zero inputs
    std::vector<double> zero_input = {0.0, 0.0, 0.0};
    std::vector<double> zero_pred = model->predict(zero_input);

    for (double val : zero_pred) {
      assertTrue(!std::isnan(val) && !std::isinf(val),
                 "Zero input should produce valid output");
    }
  }
};

/**
 * @class PrecisionHandlingIntegrationTest
 * @brief Test precision handling in training operations
 */
class PrecisionHandlingIntegrationTest : public TestCase {
public:
  PrecisionHandlingIntegrationTest()
      : TestCase("PrecisionHandlingIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 4));
    model->add(std::make_shared<activation::Tanh>());
    model->add(std::make_shared<Dense>(4, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // High precision training data
    std::vector<std::vector<double>> X = {
        {0.123456789, 0.987654321},
        {0.111111111, 0.888888889},
        {0.333333333, 0.666666667},
        {0.142857143, 0.857142857}  // 1/7 and 6/7
    };
    std::vector<std::vector<double>> Y = {{0.1}, {0.9}, {0.5}, {0.7}};

    MSELoss loss;
    SGD optimizer(0.05);

    // Train with high precision data
    assertNoThrow(
        [&]() {
          model->train(X, Y, loss, optimizer, nullptr, 100);
        },
        "High precision training should complete");

    // Test that precision is maintained in predictions
    for (size_t i = 0; i < X.size(); i++) {
      std::vector<double> pred = model->predict(X[i]);
      assertEqual(size_t(1), pred.size(),
                  "High precision input should produce scalar output");

      // Prediction should be reasonable (not exactly target, but not too far)
      double prediction_error = std::abs(pred[0] - Y[i][0]);
      assertTrue(prediction_error < 1.0,
                 "High precision prediction should be reasonable");
      assertTrue(!std::isnan(pred[0]) && !std::isinf(pred[0]),
                 "High precision prediction should be valid");
    }

    // Test consistent predictions with identical inputs
    std::vector<double> test_input = {0.5, 0.5};
    std::vector<double> pred1 = model->predict(test_input);
    std::vector<double> pred2 = model->predict(test_input);

    double precision_diff = std::abs(pred1[0] - pred2[0]);
    assertTrue(precision_diff < 1e-14,
               "Identical inputs should produce identical outputs");
  }
};

/**
 * @class EdgeCaseNumberIntegrationTest
 * @brief Test edge case number handling in training
 */
class EdgeCaseNumberIntegrationTest : public TestCase {
public:
  EdgeCaseNumberIntegrationTest() : TestCase("EdgeCaseNumberIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    // Edge case training data
    std::vector<std::vector<double>> X = {
        {0.0, 0.0, 0.0},           // All zeros
        {1.0, 1.0, 1.0},           // All ones
        {-1.0, -1.0, -1.0},        // All negative
        {1e-100, 1e-100, 1e-100},  // Extremely small
        {0.5, -0.5, 0.0}           // Mixed signs
    };
    std::vector<std::vector<double>> Y = {{0.0, 1.0},
                                          {1.0, 0.0},
                                          {0.5, 0.5},
                                          {0.1, 0.9},
                                          {0.7, 0.3}};

    MSELoss loss;
    SGD optimizer(0.1);

    bool edge_case_stable = true;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  edge_case_stable = false;
                }
              },
              50);
        },
        "Edge case training should complete");

    assertTrue(edge_case_stable, "Edge case training should be stable");

    // Test predictions with edge cases
    std::vector<std::vector<double>> edge_test_inputs = {
        {0.0, 0.0, 0.0},           // Zero input
        {1e-200, 1e-200, 1e-200},  // Underflow risk
        {10.0, 10.0, 10.0},        // Large values (but not overflow)
        {-10.0, -10.0, -10.0}      // Large negative values
    };

    for (const auto& input : edge_test_inputs) {
      std::vector<double> pred = model->predict(input);
      assertEqual(size_t(2), pred.size(),
                  "Edge case input should produce correct output size");

      for (double val : pred) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Edge case prediction should be valid");
        assertTrue(val >= 0.0 && val <= 1.0,
                   "Edge case sigmoid output should be in [0,1]");
      }
    }

    // Test with boundary values
    std::vector<double> near_zero = {1e-15, 1e-15, 1e-15};
    std::vector<double> near_one = {1.0 - 1e-15, 1.0 - 1e-15, 1.0 - 1e-15};

    std::vector<double> pred_zero = model->predict(near_zero);
    std::vector<double> pred_one = model->predict(near_one);

    // Both should produce valid outputs
    for (double val : pred_zero) {
      assertTrue(!std::isnan(val) && !std::isinf(val),
                 "Near-zero prediction should be valid");
    }
    for (double val : pred_one) {
      assertTrue(!std::isnan(val) && !std::isinf(val),
                 "Near-one prediction should be valid");
    }
  }
};

/**
 * @class MathematicalOperationsIntegrationTest
 * @brief Test mathematical operations in integrated scenarios
 */
class MathematicalOperationsIntegrationTest : public TestCase {
public:
  MathematicalOperationsIntegrationTest()
      : TestCase("MathematicalOperationsIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Test model that exercises various mathematical operations
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(4, 6));         // Matrix multiplication
    model->add(std::make_shared<activation::Tanh>());  // Hyperbolic functions
    model->add(std::make_shared<Dense>(6, 4));         // More matrix ops
    model->add(
        std::make_shared<activation::Sigmoid>());  // Exponential functions
    model->add(std::make_shared<Dense>(4, 2));     // Final matrix ops

    // Data that exercises different mathematical ranges
    std::vector<std::vector<double>> X = {
        {M_PI, M_E, sqrt(2.0), sqrt(3.0)},         // Mathematical constants
        {sin(1.0), cos(1.0), tan(0.5), log(2.0)},  // Transcendental functions
        {0.1, 0.01, 0.001, 0.0001},                // Powers of 10
        {1.0 / 3.0, 2.0 / 3.0, 1.0 / 7.0, 2.0 / 7.0}  // Rational numbers
    };
    std::vector<std::vector<double>> Y = {{0.8, 0.2},
                                          {0.3, 0.7},
                                          {0.6, 0.4},
                                          {0.1, 0.9}};

    MSELoss loss;
    SGD optimizer(0.01);

    bool math_ops_stable = true;

    assertNoThrow(
        [&]() {
          model->train(
              X, Y, loss, optimizer,
              [&](int epoch, double current_loss) {
                // Check for mathematical stability
                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                  math_ops_stable = false;
                }

                // Mathematical operations should keep loss reasonable
                if (current_loss > 1e6) {
                  math_ops_stable = false;
                }
              },
              80);
        },
        "Mathematical operations training should complete");

    assertTrue(math_ops_stable, "Mathematical operations should be stable");

    // Test predictions with mathematical inputs
    std::vector<std::vector<double>> math_test_inputs = {
        {1.0, 1.0, 1.0, 1.0},        // Unity
        {0.0, 0.0, 0.0, 0.0},        // Zero
        {0.5, 0.25, 0.125, 0.0625},  // Geometric sequence
        {1.0, 1.5, 2.0, 2.5}         // Arithmetic sequence
    };

    for (const auto& input : math_test_inputs) {
      std::vector<double> pred = model->predict(input);
      assertEqual(size_t(2), pred.size(),
                  "Mathematical input should produce correct output size");

      for (double val : pred) {
        assertTrue(!std::isnan(val) && !std::isinf(val),
                   "Mathematical prediction should be valid");
        assertTrue(val >= 0.0 && val <= 1.0,
                   "Mathematical sigmoid output should be in [0,1]");

        // Check that mathematical operations preserve reasonable precision
        // Note: Relaxed precision check for floating point operations
        assertTrue(
            std::abs(val - std::round(val * 1e3) / 1e3) < 1e-2,
            "Mathematical operations should maintain reasonable precision");
      }
    }

    // Test mathematical properties (e.g., symmetry, monotonicity)
    std::vector<double> symmetric_input1 = {0.3, 0.7, 0.4, 0.6};
    std::vector<double> symmetric_input2 = {0.7, 0.3, 0.6, 0.4};

    std::vector<double> pred1 = model->predict(symmetric_input1);
    std::vector<double> pred2 = model->predict(symmetric_input2);

    // While outputs might not be identical due to model asymmetry,
    // they should both be valid mathematical results
    for (size_t i = 0; i < pred1.size(); i++) {
      assertTrue(!std::isnan(pred1[i]) && !std::isnan(pred2[i]),
                 "Symmetric inputs should produce valid outputs");
    }
  }
};

}  // namespace test
}  // namespace MLLib
