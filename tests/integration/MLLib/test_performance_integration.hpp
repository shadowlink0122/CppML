#pragma once

#include "../../common/test_utils.hpp"
#include "../../../include/MLLib.hpp"
#include <chrono>
#include <memory>

/**
 * @file test_performance_integration.hpp
 * @brief Performance integration tests for MLLib
 * 
 * Tests performance characteristics:
 * - Training speed with different model sizes
 * - Inference latency and throughput
 * - Memory usage patterns
 * - Scalability with data size
 * - Optimization efficiency
 */

namespace MLLib {
namespace test {

/**
 * @class TrainingPerformanceIntegrationTest
 * @brief Test training performance across different scenarios
 */
class TrainingPerformanceIntegrationTest : public TestCase {
public:
    TrainingPerformanceIntegrationTest() : TestCase("TrainingPerformanceIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test 1: Small model training speed
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(5, 8));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(8, 3));
            model->add(std::make_shared<activation::Sigmoid>());
            
            // Small dataset
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> Y;
            
            for (int i = 0; i < 50; ++i) {
                std::vector<double> x = {
                    i * 0.02, (i + 1) * 0.02, (i + 2) * 0.02, 
                    (i + 3) * 0.02, (i + 4) * 0.02
                };
                std::vector<double> y = {
                    static_cast<double>(i % 3 == 0),
                    static_cast<double>(i % 3 == 1),
                    static_cast<double>(i % 3 == 2)
                };
                X.push_back(x);
                Y.push_back(y);
            }
            
            MSE loss;
            SGD optimizer(0.1);
            
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer, nullptr, 100);
            }, "Small model training should complete");
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // Small model should train quickly (less than 10 seconds)
            assertTrue(duration.count() < 10000,
                      "Small model training should complete in reasonable time");
        }
        
        // Test 2: Medium model training speed
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(20, 50));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(50, 30));
            model->add(std::make_shared<activation::Tanh>());
            model->add(std::make_shared<Dense>(30, 10));
            model->add(std::make_shared<activation::Sigmoid>());
            
            // Medium dataset
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> Y;
            
            for (int i = 0; i < 200; ++i) {
                std::vector<double> x(20);
                std::vector<double> y(10);
                
                for (int j = 0; j < 20; ++j) {
                    x[j] = (i + j) * 0.005;
                }
                for (int j = 0; j < 10; ++j) {
                    y[j] = static_cast<double>((i + j) % 2);
                }
                
                X.push_back(x);
                Y.push_back(y);
            }
            
            MSE loss;
            SGD optimizer(0.01);
            
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer, nullptr, 50);
            }, "Medium model training should complete");
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // Medium model should still train in reasonable time (less than 30 seconds)
            assertTrue(duration.count() < 30000,
                      "Medium model training should complete in reasonable time");
        }
        
        // Test 3: Training convergence speed comparison
        {
            // Test different optimizers for convergence speed
            std::vector<std::vector<double>> X = {
                {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}
            };
            std::vector<std::vector<double>> Y = {
                {0.8}, {0.6}, {0.4}, {0.2}
            };
            
            MSE loss;
            
            // Test SGD with different learning rates
            std::vector<double> learning_rates = {0.001, 0.01, 0.1};
            
            for (double lr : learning_rates) {
                auto model = std::make_unique<Sequential>();
                model->add(std::make_shared<Dense>(2, 4));
                model->add(std::make_shared<activation::Sigmoid>());
                model->add(std::make_shared<Dense>(4, 1));
                
                SGD optimizer(lr);
                
                double final_loss = std::numeric_limits<double>::max();
                
                assertNoThrow([&]() {
                    model->train(X, Y, loss, optimizer,
                        [&](int epoch, double current_loss) {
                            final_loss = current_loss;
                        }, 100);
                }, "Training with different learning rates should work");
                
                assertTrue(final_loss < 1.0, "Model should converge to reasonable loss");
            }
        }
    }
};

/**
 * @class InferencePerformanceIntegrationTest
 * @brief Test inference performance and latency
 */
class InferencePerformanceIntegrationTest : public TestCase {
public:
    InferencePerformanceIntegrationTest() : TestCase("InferencePerformanceIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Prepare trained models of different sizes
        auto small_model = std::make_unique<Sequential>();
        small_model->add(std::make_shared<Dense>(3, 5));
        small_model->add(std::make_shared<activation::ReLU>());
        small_model->add(std::make_shared<Dense>(5, 2));
        
        auto medium_model = std::make_unique<Sequential>();
        medium_model->add(std::make_shared<Dense>(10, 20));
        medium_model->add(std::make_shared<activation::ReLU>());
        medium_model->add(std::make_shared<Dense>(20, 15));
        medium_model->add(std::make_shared<activation::Tanh>());
        medium_model->add(std::make_shared<Dense>(15, 5));
        
        auto large_model = std::make_unique<Sequential>();
        large_model->add(std::make_shared<Dense>(50, 100));
        large_model->add(std::make_shared<activation::ReLU>());
        large_model->add(std::make_shared<Dense>(100, 75));
        large_model->add(std::make_shared<activation::Sigmoid>());
        large_model->add(std::make_shared<Dense>(75, 50));
        large_model->add(std::make_shared<activation::Tanh>());
        large_model->add(std::make_shared<Dense>(50, 10));
        
        // Quick training to have working models
        {
            std::vector<std::vector<double>> small_X = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
            std::vector<std::vector<double>> small_Y = {{0.8, 0.2}, {0.3, 0.7}};
            
            std::vector<std::vector<double>> medium_X;
            std::vector<std::vector<double>> medium_Y;
            for (int i = 0; i < 10; ++i) {
                std::vector<double> x(10), y(5);
                for (int j = 0; j < 10; ++j) x[j] = (i + j) * 0.01;
                for (int j = 0; j < 5; ++j) y[j] = (i + j) % 2;
                medium_X.push_back(x); medium_Y.push_back(y);
            }
            
            std::vector<std::vector<double>> large_X;
            std::vector<std::vector<double>> large_Y;
            for (int i = 0; i < 20; ++i) {
                std::vector<double> x(50), y(10);
                for (int j = 0; j < 50; ++j) x[j] = (i + j) * 0.001;
                for (int j = 0; j < 10; ++j) y[j] = (i + j) % 2;
                large_X.push_back(x); large_Y.push_back(y);
            }
            
            MSE loss;
            SGD optimizer(0.1);
            
            small_model->train(small_X, small_Y, loss, optimizer, nullptr, 10);
            medium_model->train(medium_X, medium_Y, loss, optimizer, nullptr, 10);
            large_model->train(large_X, large_Y, loss, optimizer, nullptr, 10);
        }
        
        // Test 1: Single inference latency
        {
            std::vector<double> small_input = {0.5, 0.5, 0.5};
            std::vector<double> medium_input(10, 0.5);
            std::vector<double> large_input(50, 0.5);
            
            // Small model inference
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> small_output = small_model->predict(small_input);
            auto end = std::chrono::high_resolution_clock::now();
            auto small_latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            assertEqual(size_t(2), small_output.size(), "Small model should output 2 values");
            assertTrue(small_latency.count() < 10000, "Small model inference should be fast");
            
            // Medium model inference
            start = std::chrono::high_resolution_clock::now();
            std::vector<double> medium_output = medium_model->predict(medium_input);
            end = std::chrono::high_resolution_clock::now();
            auto medium_latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            assertEqual(size_t(5), medium_output.size(), "Medium model should output 5 values");
            assertTrue(medium_latency.count() < 50000, "Medium model inference should be reasonable");
            
            // Large model inference
            start = std::chrono::high_resolution_clock::now();
            std::vector<double> large_output = large_model->predict(large_input);
            end = std::chrono::high_resolution_clock::now();
            auto large_latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            assertEqual(size_t(10), large_output.size(), "Large model should output 10 values");
            assertTrue(large_latency.count() < 100000, "Large model inference should complete");
        }
        
        // Test 2: Batch inference throughput
        {
            std::vector<std::vector<double>> batch_inputs;
            for (int i = 0; i < 100; ++i) {
                batch_inputs.push_back({0.1 + i * 0.001, 0.2 + i * 0.001, 0.3 + i * 0.001});
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<std::vector<double>> batch_outputs;
            for (const auto& input : batch_inputs) {
                batch_outputs.push_back(small_model->predict(input));
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            assertEqual(size_t(100), batch_outputs.size(), "Should process all 100 inputs");
            assertTrue(duration.count() < 1000, "Batch processing should be efficient");
            
            // Check that all outputs are valid
            for (const auto& output : batch_outputs) {
                assertEqual(size_t(2), output.size(), "Each output should have 2 values");
                for (double val : output) {
                    assertTrue(!std::isnan(val) && !std::isinf(val), "Output values should be valid");
                }
            }
        }
        
        // Test 3: Repeated inference stability
        {
            std::vector<double> test_input = {0.25, 0.75, 0.5};
            std::vector<double> first_output = small_model->predict(test_input);
            
            // Run the same prediction 1000 times
            for (int i = 0; i < 1000; ++i) {
                std::vector<double> current_output = small_model->predict(test_input);
                
                assertEqual(first_output.size(), current_output.size(),
                           "Output size should be consistent");
                
                for (size_t j = 0; j < first_output.size(); ++j) {
                    assertTrue(std::abs(first_output[j] - current_output[j]) < 1e-10,
                              "Repeated inference should give identical results");
                }
            }
        }
    }
};

/**
 * @class ScalabilityIntegrationTest
 * @brief Test scalability with increasing data and model sizes
 */
class ScalabilityIntegrationTest : public TestCase {
public:
    ScalabilityIntegrationTest() : TestCase("ScalabilityIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test 1: Data size scalability
        {
            std::vector<int> data_sizes = {10, 50, 100, 200};
            
            for (int size : data_sizes) {
                auto model = std::make_unique<Sequential>();
                model->add(std::make_shared<Dense>(5, 8));
                model->add(std::make_shared<activation::ReLU>());
                model->add(std::make_shared<Dense>(8, 1));
                
                // Generate data of specified size
                std::vector<std::vector<double>> X;
                std::vector<std::vector<double>> Y;
                
                for (int i = 0; i < size; ++i) {
                    std::vector<double> x = {
                        i * 0.01, (i + 1) * 0.01, (i + 2) * 0.01,
                        (i + 3) * 0.01, (i + 4) * 0.01
                    };
                    std::vector<double> y = {static_cast<double>(i % 2)};
                    X.push_back(x);
                    Y.push_back(y);
                }
                
                MSE loss;
                SGD optimizer(0.1);
                
                auto start = std::chrono::high_resolution_clock::now();
                
                assertNoThrow([&]() {
                    model->train(X, Y, loss, optimizer, nullptr, 20);
                }, "Training should scale with data size");
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                // Training time should scale reasonably (not exponentially)
                assertTrue(duration.count() < size * 50,
                          "Training time should scale linearly with data size");
            }
        }
        
        // Test 2: Model complexity scalability
        {
            std::vector<int> layer_sizes = {5, 10, 20, 30};
            
            // Fixed dataset
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> Y;
            
            for (int i = 0; i < 50; ++i) {
                X.push_back({i * 0.02, (i + 1) * 0.02, (i + 2) * 0.02});
                Y.push_back({static_cast<double>(i % 2)});
            }
            
            for (int size : layer_sizes) {
                auto model = std::make_unique<Sequential>();
                model->add(std::make_shared<Dense>(3, size));
                model->add(std::make_shared<activation::ReLU>());
                model->add(std::make_shared<Dense>(size, size/2));
                model->add(std::make_shared<activation::Sigmoid>());
                model->add(std::make_shared<Dense>(size/2, 1));
                
                MSE loss;
                SGD optimizer(0.1);
                
                auto start = std::chrono::high_resolution_clock::now();
                
                assertNoThrow([&]() {
                    model->train(X, Y, loss, optimizer, nullptr, 20);
                }, "Training should scale with model complexity");
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                // More complex models should take longer but not excessively
                assertTrue(duration.count() < size * 100,
                          "Training time should scale reasonably with model complexity");
            }
        }
        
        // Test 3: Deep network scalability
        {
            std::vector<int> depths = {2, 4, 6, 8};
            
            std::vector<std::vector<double>> X = {
                {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}
            };
            std::vector<std::vector<double>> Y = {
                {0.8}, {0.5}, {0.2}
            };
            
            for (int depth : depths) {
                auto model = std::make_unique<Sequential>();
                
                // Add layers based on depth
                model->add(std::make_shared<Dense>(3, 6));
                model->add(std::make_shared<activation::ReLU>());
                
                for (int i = 1; i < depth; ++i) {
                    model->add(std::make_shared<Dense>(6, 6));
                    model->add(std::make_shared<activation::ReLU>());
                }
                
                model->add(std::make_shared<Dense>(6, 1));
                
                MSE loss;
                SGD optimizer(0.01);  // Lower learning rate for deeper networks
                
                bool training_stable = true;
                
                assertNoThrow([&]() {
                    model->train(X, Y, loss, optimizer,
                        [&](int epoch, double current_loss) {
                            if (std::isnan(current_loss) || std::isinf(current_loss)) {
                                training_stable = false;
                            }
                        }, 30);
                }, "Deep network training should complete");
                
                assertTrue(training_stable, "Deep network training should be numerically stable");
                
                // Test inference still works
                std::vector<double> test_output = model->predict(MLLib::NDArray({0.5, 0.5, 0.5});
                assertEqual(size_t(1), test_output.size(), "Deep network should produce correct output");
                assertTrue(!std::isnan(test_output[0]) && !std::isinf(test_output[0]),
                          "Deep network output should be valid");
            }
        }
    }
};

/**
 * @class MemoryEfficiencyIntegrationTest
 * @brief Test memory usage patterns and efficiency
 */
class MemoryEfficiencyIntegrationTest : public TestCase {
public:
    MemoryEfficiencyIntegrationTest() : TestCase("MemoryEfficiencyIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test 1: Multiple model creation and destruction
        {
            for (int trial = 0; trial < 20; ++trial) {
                auto model = std::make_unique<Sequential>();
                model->add(std::make_shared<Dense>(10, 15));
                model->add(std::make_shared<activation::ReLU>());
                model->add(std::make_shared<Dense>(15, 10));
                model->add(std::make_shared<activation::Sigmoid>());
                model->add(std::make_shared<Dense>(10, 1));
                
                std::vector<std::vector<double>> X;
                std::vector<std::vector<double>> Y;
                
                for (int i = 0; i < 20; ++i) {
                    std::vector<double> x(10);
                    for (int j = 0; j < 10; ++j) {
                        x[j] = (trial * 20 + i + j) * 0.001;
                    }
                    X.push_back(x);
                    Y.push_back({static_cast<double>((trial + i) % 2)});
                }
                
                MSE loss;
                SGD optimizer(0.1);
                
                assertNoThrow([&]() {
                    model->train(X, Y, loss, optimizer, nullptr, 10);
                }, "Repeated model creation should not cause memory issues");
                
                // Test that inference works
                std::vector<double> test_input(10, 0.5);
                std::vector<double> output = model->predict(test_input);
                assertEqual(size_t(1), output.size(), "Model should produce valid output");
            }
        }
        
        // Test 2: Large batch processing
        {
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(5, 8));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(8, 3));
            
            // Quick training
            std::vector<std::vector<double>> train_X = {
                {1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}
            };
            std::vector<std::vector<double>> train_Y = {
                {1, 0, 0}, {0, 1, 0}
            };
            
            MSE loss;
            SGD optimizer(0.1);
            model->train(train_X, train_Y, loss, optimizer, nullptr, 20);
            
            // Process large batches
            for (int batch = 0; batch < 10; ++batch) {
                std::vector<std::vector<double>> batch_results;
                
                for (int i = 0; i < 500; ++i) {
                    std::vector<double> input = {
                        batch * 0.1 + i * 0.001,
                        batch * 0.1 + (i + 1) * 0.001,
                        batch * 0.1 + (i + 2) * 0.001,
                        batch * 0.1 + (i + 3) * 0.001,
                        batch * 0.1 + (i + 4) * 0.001
                    };
                    
                    batch_results.push_back(model->predict(input));
                }
                
                assertEqual(size_t(500), batch_results.size(), "Batch processing should complete");
                
                // Verify all results are valid
                for (const auto& result : batch_results) {
                    assertEqual(size_t(3), result.size(), "Each result should have 3 outputs");
                    for (double val : result) {
                        assertTrue(!std::isnan(val) && !std::isinf(val),
                                  "All output values should be valid");
                    }
                }
            }
        }
        
        // Test 3: Model save/load memory management
        {
            std::string temp_dir = createTempDirectory();
            
            for (int i = 0; i < 5; ++i) {
                auto model = std::make_unique<Sequential>();
                model->add(std::make_shared<Dense>(8, 12));
                model->add(std::make_shared<activation::Tanh>());
                model->add(std::make_shared<Dense>(12, 6));
                model->add(std::make_shared<activation::Sigmoid>());
                model->add(std::make_shared<Dense>(6, 2));
                
                // Quick training
                std::vector<std::vector<double>> X = {
                    {1, 2, 3, 4, 5, 6, 7, 8}, {8, 7, 6, 5, 4, 3, 2, 1}
                };
                std::vector<std::vector<double>> Y = {
                    {1, 0}, {0, 1}
                };
                
                MSE loss;
                SGD optimizer(0.1);
                model->train(X, Y, loss, optimizer, nullptr, 15);
                
                // Save model
                std::string model_path = temp_dir + "/model_" + std::to_string(i) + ".bin";
                assertTrue(ModelIO::save_model(*model, model_path, ModelFormat::BINARY),
                          "Model save should succeed");
                
                // Clear original model
                model.reset();
                
                // Load model
                auto loaded_model = ModelIO::load_model(model_path, ModelFormat::BINARY);
                assertNotNull(loaded_model.get(), "Model load should succeed");
                
                // Test that loaded model works
                std::vector<double> test_output = loaded_model->predict(MLLib::NDArray({1, 2, 3, 4, 5, 6, 7, 8});
                assertEqual(size_t(2), test_output.size(), "Loaded model should work correctly");
            }
            
            removeTempDirectory(temp_dir);
        }
    }
};

} // namespace test
} // namespace MLLib
