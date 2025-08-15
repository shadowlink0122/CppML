#pragma once

#include "../../../../common/test_utils.hpp"
#include "../../../../../include/MLLib.hpp"
#include <memory>
#include <chrono>

/**
 * @file test_time_integration.hpp
 * @brief Time utilities integration tests for MLLib
 * 
 * Tests time utilities integration:
 * - Training time measurement
 * - Performance benchmarking
 * - Timeout handling
 * - Time-based operations
 */

namespace MLLib {
namespace test {

/**
 * @class TrainingTimeIntegrationTest
 * @brief Test training time measurement in real scenarios
 */
class TrainingTimeIntegrationTest : public TestCase {
public:
    TrainingTimeIntegrationTest() : TestCase("TrainingTimeIntegrationTest") {}

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
        
        std::vector<std::vector<double>> X = {
            {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9},
            {0.2, 0.4, 0.6}, {0.3, 0.6, 0.9}, {0.5, 0.5, 0.5}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {0.3, 0.7}
        };
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        // Measure training time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 100);
        }, "Timed training should complete");
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Training should take some measurable time but not too long
        assertTrue(duration.count() >= 0, "Training should take non-negative time");
        assertTrue(duration.count() < 10000, "Training should complete in reasonable time (<10s)");
        
        // Test that model trained properly
        for (const auto& input : X) {
            std::vector<double> pred = model->predict(input);
            assertEqual(size_t(2), pred.size(), "Timed training should produce correct output size");
            
            for (double val : pred) {
                assertTrue(!std::isnan(val) && !std::isinf(val), "Timed training should produce valid outputs");
            }
        }
    }
};

/**
 * @class PerformanceBenchmarkIntegrationTest
 * @brief Test performance benchmarking utilities
 */
class PerformanceBenchmarkIntegrationTest : public TestCase {
public:
    PerformanceBenchmarkIntegrationTest() : TestCase("PerformanceBenchmarkIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        
        OutputCapture capture;
        
        // Test prediction performance
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(10, 15));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(15, 5));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Generate test data
        std::vector<std::vector<double>> test_inputs;
        for (int i = 0; i < 100; i++) {
            std::vector<double> input(10);
            for (int j = 0; j < 10; j++) {
                input[j] = (i + j) * 0.01;
            }
            test_inputs.push_back(input);
        }
        
        // Benchmark single predictions
        auto single_start = std::chrono::high_resolution_clock::now();
        for (const auto& input : test_inputs) {
            std::vector<double> output = model->predict(input);
            assertTrue(output.size() == 5, "Benchmark prediction should maintain correctness");
        }
        auto single_end = std::chrono::high_resolution_clock::now();
        auto single_duration = std::chrono::duration_cast<std::chrono::microseconds>(single_end - single_start);
        
        // Test that predictions are reasonably fast
        double avg_prediction_time = single_duration.count() / static_cast<double>(test_inputs.size());
        assertTrue(avg_prediction_time < 10000, "Average prediction should be fast (<10ms)");
        
        // Benchmark batch processing concept
        auto batch_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> batch_outputs;
        for (const auto& input : test_inputs) {
            batch_outputs.push_back(model->predict(input));
        }
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
        
        assertEqual(test_inputs.size(), batch_outputs.size(), "Batch processing should handle all inputs");
        
        // Performance should be consistent
        double time_ratio = static_cast<double>(batch_duration.count()) / static_cast<double>(single_duration.count());
        assertTrue(time_ratio >= 0.5 && time_ratio <= 2.0, "Batch performance should be consistent");
    }
};

/**
 * @class TimeoutHandlingIntegrationTest
 * @brief Test timeout handling in long-running operations
 */
class TimeoutHandlingIntegrationTest : public TestCase {
public:
    TimeoutHandlingIntegrationTest() : TestCase("TimeoutHandlingIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(5, 8));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(8, 3));
        model->add(std::make_shared<activation::Sigmoid>());
        
        std::vector<std::vector<double>> X = {
            {0.1, 0.2, 0.3, 0.4, 0.5}, {0.6, 0.7, 0.8, 0.9, 1.0},
            {0.2, 0.4, 0.6, 0.8, 1.0}, {0.1, 0.3, 0.5, 0.7, 0.9}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.5, 0.3, 0.2}
        };
        
        MSELoss loss;
        SGD optimizer(0.01); // Slow learning rate
        
        // Test with reasonable epoch limit (simulating timeout)
        auto start_time = std::chrono::high_resolution_clock::now();
        int epochs_completed = 0;
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                    epochs_completed = epoch + 1;
                    
                    // Simulate timeout check
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                    
                    if (elapsed.count() > 1000) { // 1 second timeout simulation
                        // In real implementation, this might stop training
                        return;
                    }
                }, 1000); // Many epochs, but should timeout first
        }, "Training with timeout simulation should complete");
        
        assertTrue(epochs_completed > 0, "Should complete some epochs before timeout");
        
        // Model should still be functional after timeout
        std::vector<double> test_input = {0.3, 0.3, 0.3, 0.3, 0.3};
        std::vector<double> test_output = model->predict(test_input);
        assertEqual(size_t(3), test_output.size(), "Model should work after timeout");
        
        for (double val : test_output) {
            assertTrue(!std::isnan(val) && !std::isinf(val), "Model should produce valid outputs after timeout");
        }
    }
};

/**
 * @class TimeBasedOperationsIntegrationTest
 * @brief Test time-based operations in training
 */
class TimeBasedOperationsIntegrationTest : public TestCase {
public:
    TimeBasedOperationsIntegrationTest() : TestCase("TimeBasedOperationsIntegrationTest") {}

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
        
        std::vector<std::vector<double>> X = {
            {0.1, 0.9}, {0.9, 0.1}, {0.3, 0.7}, {0.7, 0.3}, {0.5, 0.5}
        };
        std::vector<std::vector<double>> Y = {
            {1.0}, {0.0}, {1.0}, {0.0}, {0.5}
        };
        
        MSELoss loss;
        SGD optimizer(0.2);
        
        // Test time-based learning rate decay simulation
        std::vector<double> loss_history;
        std::vector<std::chrono::milliseconds> time_history;
        auto training_start = std::chrono::high_resolution_clock::now();
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                    loss_history.push_back(current_loss);
                    
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - training_start);
                    time_history.push_back(elapsed);
                    
                    // Simulate time-based adjustments
                    if (elapsed.count() > 100) { // After 100ms
                        // In real implementation, might adjust learning rate
                    }
                }, 50);
        }, "Time-based training should complete");
        
        assertTrue(loss_history.size() > 0, "Should record training progress");
        assertTrue(time_history.size() == loss_history.size(), "Should record time for each epoch");
        
        // Verify time progression is monotonic
        for (size_t i = 1; i < time_history.size(); i++) {
            assertTrue(time_history[i].count() >= time_history[i-1].count(),
                      "Time should progress monotonically");
        }
        
        // Test that training was time-aware
        auto final_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(final_time - training_start);
        
        assertTrue(total_duration.count() >= 0, "Total training time should be positive");
        assertTrue(total_duration.count() < 5000, "Training should complete in reasonable time");
        
        // Final model should work
        std::vector<double> final_input = {0.4, 0.6};
        std::vector<double> final_pred = model->predict(final_input);
        assertTrue(!std::isnan(final_pred[0]) && !std::isinf(final_pred[0]),
                  "Time-based training should produce valid final model");
    }
};

} // namespace test
} // namespace MLLib
