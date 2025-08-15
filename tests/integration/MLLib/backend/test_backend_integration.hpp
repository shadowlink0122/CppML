#pragma once

#include "../../../common/test_utils.hpp"
#include "../../../../include/MLLib.hpp"
#include <memory>

/**
 * @file test_backend_integration.hpp
 * @brief Backend integration tests for MLLib
 * 
 * Tests backend integration:
 * - CPU backend functionality
 * - Backend-model interaction
 * - Performance characteristics
 * - Memory management
 */

namespace MLLib {
namespace test {

/**
 * @class CPUBackendIntegrationTest
 * @brief Test CPU backend integration with models
 */
class CPUBackendIntegrationTest : public TestCase {
public:
    CPUBackendIntegrationTest() : TestCase("CPUBackendIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create model that will use CPU backend
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 6));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(6, 4));
        model->add(std::make_shared<activation::Tanh>());
        model->add(std::make_shared<Dense>(4, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Training data
        std::vector<std::vector<double>> X = {
            {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9},
            {0.2, 0.3, 0.4}, {0.5, 0.6, 0.7}, {0.8, 0.9, 1.0}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}
        };
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        // Test training on CPU backend
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 50);
        }, "CPU backend training should complete");
        
        // Test single prediction
        std::vector<double> single_pred = model->predict({0.3, 0.4, 0.5});
        assertEqual(size_t(2), single_pred.size(), "Single prediction should have correct size");
        
        for (double val : single_pred) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Prediction values should be valid");
            assertTrue(val >= 0.0 && val <= 1.0, "Sigmoid output should be in [0,1]");
        }
        
        // Test batch prediction
        std::vector<std::vector<double>> batch_inputs = {
            {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}
        };
        
        for (const auto& input : batch_inputs) {
            std::vector<double> batch_pred = model->predict(input);
            assertEqual(size_t(2), batch_pred.size(), "Batch prediction should have correct size");
            
            for (double val : batch_pred) {
                assertTrue(!std::isnan(val) && !std::isinf(val), "Batch prediction values should be valid");
            }
        }
    }
};

/**
 * @class BackendMemoryIntegrationTest
 * @brief Test backend memory management integration
 */
class BackendMemoryIntegrationTest : public TestCase {
public:
    BackendMemoryIntegrationTest() : TestCase("BackendMemoryIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test memory management with multiple model creations
        for (int trial = 0; trial < 5; ++trial) {
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(10, 15));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(15, 5));
            model->add(std::make_shared<activation::Sigmoid>());
            
            // Generate data for this trial
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> Y;
            
            for (int i = 0; i < 20; ++i) {
                std::vector<double> x(10);
                std::vector<double> y(5);
                
                for (int j = 0; j < 10; ++j) {
                    x[j] = (trial * 20 + i + j) * 0.01;
                }
                for (int j = 0; j < 5; ++j) {
                    y[j] = static_cast<double>((trial + i + j) % 2);
                }
                
                X.push_back(x);
                Y.push_back(y);
            }
            
            MSELoss loss;
            SGD optimizer(0.05);
            
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer, nullptr, 20);
            }, "Memory management test " + std::to_string(trial) + " should complete");
            
            // Test that model still works
            std::vector<double> test_input(10, 0.5);
            std::vector<double> output = model->predict(test_input);
            assertEqual(size_t(5), output.size(), "Output should have correct size");
            
            for (double val : output) {
                assertTrue(!std::isnan(val) && !std::isinf(val), "Output should be valid");
            }
        }
    }
};

/**
 * @class BackendPerformanceIntegrationTest
 * @brief Test backend performance characteristics
 */
class BackendPerformanceIntegrationTest : public TestCase {
public:
    BackendPerformanceIntegrationTest() : TestCase("BackendPerformanceIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create larger model to test performance
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(20, 40));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(40, 20));
        model->add(std::make_shared<activation::Tanh>());
        model->add(std::make_shared<Dense>(20, 10));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Generate larger dataset
        std::vector<std::vector<double>> X;
        std::vector<std::vector<double>> Y;
        
        for (int i = 0; i < 100; ++i) {
            std::vector<double> x(20);
            std::vector<double> y(10);
            
            for (int j = 0; j < 20; ++j) {
                x[j] = (i + j) * 0.005;
            }
            for (int j = 0; j < 10; ++j) {
                y[j] = static_cast<double>((i + j) % 3 == 0);
            }
            
            X.push_back(x);
            Y.push_back(y);
        }
        
        MSELoss loss;
        SGD optimizer(0.01);
        
        // Test training performance
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 30);
        }, "Performance test training should complete");
        
        // Test inference performance with multiple predictions
        for (int i = 0; i < 50; ++i) {
            std::vector<double> test_input(20);
            for (int j = 0; j < 20; ++j) {
                test_input[j] = i * 0.01 + j * 0.002;
            }
            
            std::vector<double> output = model->predict(test_input);
            assertEqual(size_t(10), output.size(), "Performance test output should have correct size");
            
            for (double val : output) {
                assertTrue(!std::isnan(val) && !std::isinf(val), "Performance test output should be valid");
            }
        }
    }
};

} // namespace test
} // namespace MLLib
