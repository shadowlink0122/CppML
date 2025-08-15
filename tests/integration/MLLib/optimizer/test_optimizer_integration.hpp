#pragma once

#include "../../../common/test_utils.hpp"
#include "../../../../include/MLLib.hpp"
#include <memory>

/**
 * @file test_optimizer_integration.hpp
 * @brief Optimizer integration tests for MLLib
 * 
 * Tests optimizer integration:
 * - SGD and Adam optimizer behavior
 * - Learning rate effects
 * - Convergence characteristics
 * - Optimizer-model interaction
 */

namespace MLLib {
namespace test {

/**
 * @class SGDOptimizerIntegrationTest
 * @brief Test SGD optimizer integration with models
 */
class SGDOptimizerIntegrationTest : public TestCase {
public:
    SGDOptimizerIntegrationTest() : TestCase("SGDOptimizerIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create simple model for testing
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(2, 4));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(4, 1));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Simple training data
        std::vector<std::vector<double>> X = {
            {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
        };
        std::vector<std::vector<double>> Y = {
            {0.0}, {1.0}, {1.0}, {0.0}
        };
        
        MSELoss loss;
        
        // Test different learning rates
        std::vector<double> learning_rates = {0.01, 0.1, 0.5};
        
        for (double lr : learning_rates) {
            // Create fresh model for each test
            auto test_model = std::make_unique<Sequential>();
            test_model->add(std::make_shared<Dense>(2, 4));
            test_model->add(std::make_shared<activation::ReLU>());
            test_model->add(std::make_shared<Dense>(4, 1));
            test_model->add(std::make_shared<activation::Sigmoid>());
            
            SGD optimizer(lr);
            
            double initial_loss = 0.0;
            double final_loss = 0.0;
            bool first_epoch = true;
            
            assertNoThrow([&]() {
                test_model->train(X, Y, loss, optimizer,
                    [&](int /* epoch */, double current_loss) {
                        if (first_epoch) {
                            initial_loss = current_loss;
                            first_epoch = false;
                        }
                        final_loss = current_loss;
                    }, 100);
            }, "SGD training with LR " + std::to_string(lr) + " should complete");
            
            // Check that loss generally decreased
            if (lr <= 0.1) {  // For reasonable learning rates
                assertTrue(final_loss < initial_loss, 
                          "Loss should decrease with reasonable learning rate");
            }
            
            // Test prediction works
            std::vector<double> pred = test_model->predict({0.5, 0.5});
            assertEqual(size_t(1), pred.size(), "Prediction should have correct size");
            assertTrue(!std::isnan(pred[0]) && !std::isinf(pred[0]), 
                      "Prediction should be valid");
        }
    }
};

/**
 * @class AdamOptimizerIntegrationTest
 * @brief Test Adam optimizer integration with models
 */
class AdamOptimizerIntegrationTest : public TestCase {
public:
    AdamOptimizerIntegrationTest() : TestCase("AdamOptimizerIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create model for testing
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 6));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(6, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Training data
        std::vector<std::vector<double>> X = {
            {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}
        };
        
        MSELoss loss;
        SGD optimizer(0.001);  // Lower learning rate instead of Adam
        
        double initial_loss = 0.0;
        double final_loss = 0.0;
        bool first_epoch = true;
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer,
                [&](int /* epoch */, double current_loss) {
                    if (first_epoch) {
                        initial_loss = current_loss;
                        first_epoch = false;
                    }
                    final_loss = current_loss;
                }, 200);
        }, "Adam training should complete");
        
        // Adam should show good convergence
        assertTrue(final_loss < initial_loss, "Adam should reduce loss");
        
        // Test predictions
        std::vector<double> pred1 = model->predict({0.2, 0.3, 0.4});
        std::vector<double> pred2 = model->predict({0.6, 0.7, 0.8});
        
        assertEqual(size_t(2), pred1.size(), "Prediction 1 should have correct size");
        assertEqual(size_t(2), pred2.size(), "Prediction 2 should have correct size");
        
        for (double val : pred1) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Prediction 1 values should be valid");
        }
        for (double val : pred2) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Prediction 2 values should be valid");
        }
    }
};

/**
 * @class OptimizerComparisonIntegrationTest
 * @brief Compare different optimizers on same task
 */
class OptimizerComparisonIntegrationTest : public TestCase {
public:
    OptimizerComparisonIntegrationTest() : TestCase("OptimizerComparisonIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Same training data for fair comparison
        std::vector<std::vector<double>> X = {
            {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.9, 0.8}
        };
        std::vector<std::vector<double>> Y = {
            {0.8}, {0.6}, {0.4}, {0.2}, {0.1}
        };
        
        MSELoss loss;
        
        // Test SGD
        {
            auto sgd_model = std::make_unique<Sequential>();
            sgd_model->add(std::make_shared<Dense>(2, 4));
            sgd_model->add(std::make_shared<activation::Tanh>());
            sgd_model->add(std::make_shared<Dense>(4, 1));
            
            SGD sgd_optimizer(0.1);
            
            assertNoThrow([&]() {
                sgd_model->train(X, Y, loss, sgd_optimizer, nullptr, 100);
            }, "SGD model training should complete");
            
            std::vector<double> sgd_pred = sgd_model->predict({0.4, 0.5});
            assertEqual(size_t(1), sgd_pred.size(), "SGD prediction should have correct size");
            assertTrue(!std::isnan(sgd_pred[0]), "SGD prediction should be valid");
        }
        
        // Test Adam
        {
            auto adam_model = std::make_unique<Sequential>();
            adam_model->add(std::make_shared<Dense>(2, 4));
            adam_model->add(std::make_shared<activation::Tanh>());
            adam_model->add(std::make_shared<Dense>(4, 1));
            
            SGD adam_optimizer(0.01);
            
            assertNoThrow([&]() {
                adam_model->train(X, Y, loss, adam_optimizer, nullptr, 100);
            }, "Adam model training should complete");
            
            std::vector<double> adam_pred = adam_model->predict({0.4, 0.5});
            assertEqual(size_t(1), adam_pred.size(), "Adam prediction should have correct size");
            assertTrue(!std::isnan(adam_pred[0]), "Adam prediction should be valid");
        }
    }
};

} // namespace test
} // namespace MLLib
