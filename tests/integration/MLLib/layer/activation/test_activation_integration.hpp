#pragma once

#include "../../../../common/test_utils.hpp"
#include "../../../../../include/MLLib.hpp"
#include <memory>

/**
 * @file test_activation_integration.hpp
 * @brief Activation layer integration tests for MLLib
 * 
 * Tests activation layer integration:
 * - ReLU, Sigmoid, Tanh activation behavior in models
 * - Activation-layer combinations
 * - Gradient flow through activations
 * - Activation impact on training
 */

namespace MLLib {
namespace test {

/**
 * @class ReLUActivationIntegrationTest
 * @brief Test ReLU activation integration in models
 */
class ReLUActivationIntegrationTest : public TestCase {
public:
    ReLUActivationIntegrationTest() : TestCase("ReLUActivationIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create model with ReLU activations
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(2, 4));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(4, 3));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(3, 1));
        model->add(std::make_shared<activation::Sigmoid>()); // Output activation
        
        // Training data
        std::vector<std::vector<double>> X = {
            {-1.0, -1.0}, {-1.0, 1.0}, {1.0, -1.0}, {1.0, 1.0}
        };
        std::vector<std::vector<double>> Y = {
            {0.0}, {1.0}, {1.0}, {0.0}
        };
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        bool training_stable = true;
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                    if (std::isnan(current_loss) || std::isinf(current_loss)) {
                        training_stable = false;
                    }
                }, 100);
        }, "ReLU model training should complete");
        
        assertTrue(training_stable, "ReLU model training should be stable");
        
        // Test that ReLU allows learning of XOR-like pattern
        std::vector<double> pred1 = model->predict({-0.5, -0.5});
        std::vector<double> pred2 = model->predict({0.5, 0.5});
        
        assertEqual(size_t(1), pred1.size(), "ReLU prediction should have correct size");
        assertEqual(size_t(1), pred2.size(), "ReLU prediction should have correct size");
        
        // ReLU should handle negative inputs properly (dead neurons possible but training should work)
        assertTrue(!std::isnan(pred1[0]) && !std::isinf(pred1[0]), "ReLU prediction should be valid");
        assertTrue(!std::isnan(pred2[0]) && !std::isinf(pred2[0]), "ReLU prediction should be valid");
    }
};

/**
 * @class SigmoidActivationIntegrationTest
 * @brief Test Sigmoid activation integration in models
 */
class SigmoidActivationIntegrationTest : public TestCase {
public:
    SigmoidActivationIntegrationTest() : TestCase("SigmoidActivationIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create model with Sigmoid activations
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(2, 3));
        model->add(std::make_shared<activation::Sigmoid>());
        model->add(std::make_shared<Dense>(3, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Binary classification data
        std::vector<std::vector<double>> X = {
            {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}
        };
        
        MSELoss loss;
        SGD optimizer(0.5); // Higher learning rate for sigmoid
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 150);
        }, "Sigmoid model training should complete");
        
        // Test predictions - all outputs should be in [0,1] due to sigmoid
        for (const auto& input : X) {
            std::vector<double> pred = model->predict(input);
            assertEqual(size_t(2), pred.size(), "Sigmoid prediction should have correct size");
            
            for (double val : pred) {
                assertTrue(val >= 0.0 && val <= 1.0, "Sigmoid output should be in [0,1]");
                assertTrue(!std::isnan(val) && !std::isinf(val), "Sigmoid output should be valid");
            }
        }
    }
};

/**
 * @class TanhActivationIntegrationTest
 * @brief Test Tanh activation integration in models
 */
class TanhActivationIntegrationTest : public TestCase {
public:
    TanhActivationIntegrationTest() : TestCase("TanhActivationIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create model with Tanh activations
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 4));
        model->add(std::make_shared<activation::Tanh>());
        model->add(std::make_shared<Dense>(4, 2));
        model->add(std::make_shared<activation::Tanh>());
        model->add(std::make_shared<Dense>(2, 1));
        model->add(std::make_shared<activation::Sigmoid>()); // Output in [0,1]
        
        // Regression-like data
        std::vector<std::vector<double>> X = {
            {-1.0, 0.0, 1.0}, {0.0, 1.0, -1.0}, {1.0, -1.0, 0.0}, {-0.5, 0.5, 0.5}
        };
        std::vector<std::vector<double>> Y = {
            {0.2}, {0.8}, {0.6}, {0.4}
        };
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 100);
        }, "Tanh model training should complete");
        
        // Test predictions - intermediate layers use tanh, output uses sigmoid
        std::vector<double> pred1 = model->predict({0.0, 0.0, 0.0});
        std::vector<double> pred2 = model->predict({1.0, 1.0, 1.0});
        
        assertEqual(size_t(1), pred1.size(), "Tanh model prediction should have correct size");
        assertEqual(size_t(1), pred2.size(), "Tanh model prediction should have correct size");
        
        // Output should be in [0,1] due to final sigmoid
        assertTrue(pred1[0] >= 0.0 && pred1[0] <= 1.0, "Final output should be in [0,1]");
        assertTrue(pred2[0] >= 0.0 && pred2[0] <= 1.0, "Final output should be in [0,1]");
        
        assertTrue(!std::isnan(pred1[0]) && !std::isinf(pred1[0]), "Tanh prediction should be valid");
        assertTrue(!std::isnan(pred2[0]) && !std::isinf(pred2[0]), "Tanh prediction should be valid");
    }
};

/**
 * @class MixedActivationIntegrationTest
 * @brief Test models with mixed activation functions
 */
class MixedActivationIntegrationTest : public TestCase {
public:
    MixedActivationIntegrationTest() : TestCase("MixedActivationIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create model mixing different activations
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 6));
        model->add(std::make_shared<activation::ReLU>());      // Layer 1: ReLU
        model->add(std::make_shared<Dense>(6, 4));
        model->add(std::make_shared<activation::Tanh>());      // Layer 2: Tanh
        model->add(std::make_shared<Dense>(4, 2));
        model->add(std::make_shared<activation::Sigmoid>());   // Layer 3: Sigmoid
        
        // Multi-class data
        std::vector<std::vector<double>> X = {
            {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9},
            {0.2, 0.1, 0.4}, {0.5, 0.3, 0.7}, {0.8, 0.6, 0.2}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}
        };
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        bool training_completed = false;
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 100);
            training_completed = true;
        }, "Mixed activation model training should complete");
        
        assertTrue(training_completed, "Mixed activation training should complete");
        
        // Test that mixed activations work together
        for (const auto& input : X) {
            std::vector<double> pred = model->predict(input);
            assertEqual(size_t(2), pred.size(), "Mixed activation prediction should have correct size");
            
            for (double val : pred) {
                assertTrue(val >= 0.0 && val <= 1.0, "Final sigmoid should constrain to [0,1]");
                assertTrue(!std::isnan(val) && !std::isinf(val), "Mixed activation output should be valid");
            }
        }
        
        // Test edge cases
        std::vector<double> zero_input = {0.0, 0.0, 0.0};
        std::vector<double> large_input = {10.0, 10.0, 10.0};
        std::vector<double> negative_input = {-5.0, -5.0, -5.0};
        
        std::vector<double> pred_zero = model->predict(zero_input);
        std::vector<double> pred_large = model->predict(large_input);
        std::vector<double> pred_neg = model->predict(negative_input);
        
        // All should produce valid outputs
        for (double val : pred_zero) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Zero input should produce valid output");
        }
        for (double val : pred_large) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Large input should produce valid output");
        }
        for (double val : pred_neg) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Negative input should produce valid output");
        }
    }
};

} // namespace test
} // namespace MLLib
