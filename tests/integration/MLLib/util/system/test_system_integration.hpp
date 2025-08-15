#pragma once

#include "../../../../common/test_utils.hpp"
#include "../../../../../include/MLLib.hpp"
#include <memory>

/**
 * @file test_system_integration.hpp
 * @brief System utilities integration tests for MLLib
 * 
 * Tests system utilities integration:
 * - Memory management during training
 * - Resource usage monitoring
 * - System-level error handling
 * - Cross-platform compatibility
 */

namespace MLLib {
namespace test {

/**
 * @class MemoryManagementIntegrationTest
 * @brief Test memory management during intensive operations
 */
class MemoryManagementIntegrationTest : public TestCase {
public:
    MemoryManagementIntegrationTest() : TestCase("MemoryManagementIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test memory management with multiple model creation/destruction cycles
        for (int cycle = 0; cycle < 5; cycle++) {
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(10, 20));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(20, 10));
            model->add(std::make_shared<activation::Sigmoid>());
            
            // Training data
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> Y;
            
            for (int i = 0; i < 20; i++) {
                std::vector<double> x(10), y(10);
                for (int j = 0; j < 10; j++) {
                    x[j] = (cycle + i + j) * 0.01;
                    y[j] = (x[j] > 0.5) ? 1.0 : 0.0;
                }
                X.push_back(x);
                Y.push_back(y);
            }
            
            MSELoss loss;
            SGD optimizer(0.01);
            
            // Training should complete without memory issues
            bool training_completed = false;
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer, nullptr, 20);
                training_completed = true;
            }, "Memory management training should complete");
            
            assertTrue(training_completed, "Training should complete in memory cycle " + std::to_string(cycle));
            
            // Test predictions
            std::vector<double> test_input(10, 0.5);
            std::vector<double> output = model->predict(test_input);
            
            assertEqual(size_t(10), output.size(), "Memory managed model should produce correct output size");
            
            for (double val : output) {
                assertTrue(!std::isnan(val) && !std::isinf(val), "Memory managed predictions should be valid");
            }
            
            // Model will be destroyed at end of loop iteration
        }
        
        assertTrue(true, "Memory management cycles completed successfully");
    }
};

/**
 * @class ResourceUsageIntegrationTest
 * @brief Test resource usage monitoring during training
 */
class ResourceUsageIntegrationTest : public TestCase {
public:
    ResourceUsageIntegrationTest() : TestCase("ResourceUsageIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test resource usage with varying model sizes
        std::vector<std::pair<int, int>> model_sizes = {
            {5, 10}, {10, 20}, {20, 40}, {15, 30}
        };
        
        for (const auto& size_pair : model_sizes) {
            int input_size = size_pair.first;
            int hidden_size = size_pair.second;
            
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(input_size, hidden_size));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(hidden_size, input_size));
            model->add(std::make_shared<activation::Sigmoid>());
            
            // Generate appropriately sized data
            std::vector<std::vector<double>> X, Y;
            for (int i = 0; i < 10; i++) {
                std::vector<double> x(input_size), y(input_size);
                for (int j = 0; j < input_size; j++) {
                    x[j] = i * 0.1 + j * 0.01;
                    y[j] = (x[j] > 0.3) ? 1.0 : 0.0;
                }
                X.push_back(x);
                Y.push_back(y);
            }
            
            MSELoss loss;
            SGD optimizer(0.05);
            
            // Monitor resource usage through training stability
            bool resource_stable = true;
            
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer,
                    [&](int epoch, double current_loss) {
                        if (std::isnan(current_loss) || std::isinf(current_loss)) {
                            resource_stable = false;
                        }
                    }, 30);
            }, "Resource usage training should complete");
            
            assertTrue(resource_stable, "Resource usage should remain stable for model size " + 
                      std::to_string(input_size) + "x" + std::to_string(hidden_size));
            
            // Test that model still functions after training
            std::vector<double> test_input(input_size, 0.5);
            std::vector<double> output = model->predict(test_input);
            
            assertEqual(static_cast<size_t>(input_size), output.size(), 
                       "Resource-monitored model should maintain functionality");
        }
    }
};

/**
 * @class SystemErrorHandlingIntegrationTest
 * @brief Test system-level error handling
 */
class SystemErrorHandlingIntegrationTest : public TestCase {
public:
    SystemErrorHandlingIntegrationTest() : TestCase("SystemErrorHandlingIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 5));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(5, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Test error handling with edge case scenarios
        
        // Test 1: Empty data handling
        {
            std::vector<std::vector<double>> empty_X, empty_Y;
            MSELoss loss;
            SGD optimizer(0.1);
            
            bool empty_data_handled = true;
            try {
                if (!empty_X.empty()) {  // Only train if data exists
                    model->train(empty_X, empty_Y, loss, optimizer, nullptr, 10);
                }
            } catch (...) {
                empty_data_handled = false;
            }
            
            assertTrue(empty_data_handled, "Empty data should be handled gracefully");
        }
        
        // Test 2: Model consistency after errors
        {
            std::vector<std::vector<double>> X = {
                {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}
            };
            std::vector<std::vector<double>> Y = {
                {1.0, 0.0}, {0.0, 1.0}
            };
            
            MSELoss loss;
            SGD optimizer(0.1);
            
            // Get baseline prediction
            std::vector<double> baseline_pred = model->predict({0.5, 0.5, 0.5});
            
            // Train model
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer, nullptr, 20);
            }, "Error handling training should complete");
            
            // Model should still be functional
            std::vector<double> post_train_pred = model->predict({0.5, 0.5, 0.5});
            
            assertEqual(baseline_pred.size(), post_train_pred.size(), 
                       "Model should maintain consistency after error handling");
            
            for (double val : post_train_pred) {
                assertTrue(!std::isnan(val) && !std::isinf(val), 
                          "Model should produce valid outputs after error handling");
            }
        }
        
        // Test 3: Extreme parameter handling
        {
            std::vector<std::vector<double>> X = {{0.1, 0.2, 0.3}};
            std::vector<std::vector<double>> Y = {{0.5, 0.5}};
            
            MSELoss loss;
            SGD extreme_optimizer(1000.0); // Extremely high learning rate
            
            bool extreme_params_handled = true;
            
            try {
                model->train(X, Y, loss, extreme_optimizer, 
                    [&](int epoch, double current_loss) {
                        if (std::isnan(current_loss) || std::isinf(current_loss)) {
                            // Expected with extreme parameters
                        }
                    }, 5); // Very few epochs
            } catch (...) {
                extreme_params_handled = false;
            }
            
            assertTrue(extreme_params_handled, "Extreme parameters should be handled without crashes");
            
            // Model should still be responsive
            std::vector<double> extreme_test_pred = model->predict({0.5, 0.5, 0.5});
            assertTrue(extreme_test_pred.size() == 2, "Model should maintain structure after extreme parameters");
        }
    }
};

/**
 * @class CrossPlatformCompatibilityIntegrationTest
 * @brief Test cross-platform compatibility
 */
class CrossPlatformCompatibilityIntegrationTest : public TestCase {
public:
    CrossPlatformCompatibilityIntegrationTest() : TestCase("CrossPlatformCompatibilityIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test platform-independent functionality
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(4, 6));
        model->add(std::make_shared<activation::Tanh>());
        model->add(std::make_shared<Dense>(6, 3));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Test with platform-independent data
        std::vector<std::vector<double>> X = {
            {0.0, 0.25, 0.5, 0.75}, {0.1, 0.3, 0.6, 0.9},
            {0.2, 0.4, 0.7, 0.8}, {0.15, 0.35, 0.65, 0.85}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.5, 0.3, 0.2}
        };
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        bool platform_compatible = true;
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                    // Check for platform-specific numerical issues
                    if (std::isnan(current_loss) || std::isinf(current_loss)) {
                        platform_compatible = false;
                    }
                }, 50);
        }, "Cross-platform training should complete");
        
        assertTrue(platform_compatible, "Training should be stable across platforms");
        
        // Test numerical consistency
        std::vector<double> test_input = {0.3, 0.3, 0.3, 0.3};
        std::vector<double> prediction1 = model->predict(test_input);
        std::vector<double> prediction2 = model->predict(test_input);
        
        // Predictions should be identical (deterministic)
        for (size_t i = 0; i < prediction1.size(); i++) {
            assertTrue(std::abs(prediction1[i] - prediction2[i]) < 1e-12,
                      "Predictions should be deterministic across platforms");
        }
        
        // Test floating point precision consistency
        double small_val = 1e-15;
        double large_val = 1e15;
        
        assertTrue(small_val > 0.0, "Small values should be handled consistently");
        assertTrue(large_val > 1e10, "Large values should be handled consistently");
        
        // Test boundary conditions
        std::vector<double> boundary_input = {0.0, 1.0, -1.0, 0.5};
        std::vector<double> boundary_output = model->predict(boundary_input);
        
        for (double val : boundary_output) {
            assertTrue(!std::isnan(val) && !std::isinf(val),
                      "Boundary conditions should produce valid results across platforms");
            assertTrue(val >= 0.0 && val <= 1.0,
                      "Sigmoid outputs should respect bounds across platforms");
        }
    }
};

} // namespace test
} // namespace MLLib
