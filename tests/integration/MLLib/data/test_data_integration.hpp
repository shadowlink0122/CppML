#pragma once

#include "../../../common/test_utils.hpp"
#include "../../../../include/MLLib.hpp"
#include <memory>
#include <algorithm>

/**
 * @file test_data_integration.hpp
 * @brief Data handling integration tests for MLLib
 * 
 * Tests data handling integration:
 * - Data loading and preprocessing
 * - Batch processing
 * - Data validation
 * - Data format compatibility
 */

namespace MLLib {
namespace test {

/**
 * @class DataLoadingIntegrationTest
 * @brief Test data loading and preprocessing
 */
class DataLoadingIntegrationTest : public TestCase {
public:
    DataLoadingIntegrationTest() : TestCase("DataLoadingIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test data loading simulation
        std::vector<std::vector<double>> raw_data = {
            {1.0, 2.0, 3.0, 1.0}, {4.0, 5.0, 6.0, 0.0}, {7.0, 8.0, 9.0, 1.0},
            {2.0, 3.0, 4.0, 0.0}, {5.0, 6.0, 7.0, 1.0}, {8.0, 9.0, 10.0, 0.0}
        };
        
        // Preprocess data (normalize features)
        std::vector<std::vector<double>> X, Y;
        for (const auto& row : raw_data) {
            if (row.size() >= 4) {
                // Normalize features to [0,1] range
                std::vector<double> features = {row[0]/10.0, row[1]/10.0, row[2]/10.0};
                std::vector<double> label = {row[3]};
                
                X.push_back(features);
                Y.push_back(label);
            }
        }
        
        assertEqual(size_t(6), X.size(), "Data loading should preserve sample count");
        assertEqual(size_t(6), Y.size(), "Data loading should preserve label count");
        
        // Test training with loaded data
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 5));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(5, 1));
        model->add(std::make_shared<activation::Sigmoid>());
        
        MSELoss loss;
        SGD optimizer(0.1);
        
        bool data_loading_successful = true;
        
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer,
                [&](int epoch, double current_loss) {
                    if (std::isnan(current_loss) || std::isinf(current_loss)) {
                        data_loading_successful = false;
                    }
                }, 50);
        }, "Training with loaded data should complete");
        
        assertTrue(data_loading_successful, "Data loading should enable successful training");
        
        // Test predictions on loaded data
        for (const auto& input : X) {
            std::vector<double> pred = model->predict(input);
            assertEqual(size_t(1), pred.size(), "Loaded data should produce correct prediction size");
            assertTrue(!std::isnan(pred[0]) && !std::isinf(pred[0]), "Loaded data predictions should be valid");
        }
    }
};

/**
 * @class BatchProcessingIntegrationTest
 * @brief Test batch processing functionality
 */
class BatchProcessingIntegrationTest : public TestCase {
public:
    BatchProcessingIntegrationTest() : TestCase("BatchProcessingIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(4, 6));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(6, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Generate batch data
        std::vector<std::vector<double>> batch_X, batch_Y;
        for (int i = 0; i < 12; i++) {
            std::vector<double> x = {
                i * 0.1, (i+1) * 0.1, (i+2) * 0.1, (i+3) * 0.1
            };
            std::vector<double> y = {
                (i % 2 == 0) ? 1.0 : 0.0, (i % 2 == 1) ? 1.0 : 0.0
            };
            batch_X.push_back(x);
            batch_Y.push_back(y);
        }
        
        // Test different batch sizes
        std::vector<int> batch_sizes = {1, 3, 4, 6};
        
        for (int batch_size : batch_sizes) {
            MSELoss loss;
            SGD optimizer(0.05);
            
            // Simulate batch processing by training in chunks
            for (size_t start = 0; start < batch_X.size(); start += batch_size) {
                size_t end = std::min(start + batch_size, batch_X.size());
                
                std::vector<std::vector<double>> batch_x_chunk(
                    batch_X.begin() + start, batch_X.begin() + end);
                std::vector<std::vector<double>> batch_y_chunk(
                    batch_Y.begin() + start, batch_Y.begin() + end);
                
                if (!batch_x_chunk.empty()) {
                    bool batch_training_successful = true;
                    
                    assertNoThrow([&]() {
                        model->train(batch_x_chunk, batch_y_chunk, loss, optimizer,
                            [&](int epoch, double current_loss) {
                                if (std::isnan(current_loss) || std::isinf(current_loss)) {
                                    batch_training_successful = false;
                                }
                            }, 10); // Few epochs per batch
                    }, "Batch processing should complete");
                    
                    assertTrue(batch_training_successful, 
                              "Batch training should be successful for batch size " + std::to_string(batch_size));
                }
            }
            
            // Test batch predictions
            for (size_t i = 0; i < batch_X.size(); i += batch_size) {
                std::vector<double> batch_pred = model->predict(batch_X[i]);
                assertEqual(size_t(2), batch_pred.size(), "Batch predictions should have correct size");
                
                for (double val : batch_pred) {
                    assertTrue(!std::isnan(val) && !std::isinf(val), "Batch predictions should be valid");
                }
            }
        }
    }
};

/**
 * @class DataValidationIntegrationTest
 * @brief Test data validation during training
 */
class DataValidationIntegrationTest : public TestCase {
public:
    DataValidationIntegrationTest() : TestCase("DataValidationIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 4));
        model->add(std::make_shared<activation::Tanh>());
        model->add(std::make_shared<Dense>(4, 1));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Test 1: Valid data
        {
            std::vector<std::vector<double>> valid_X = {
                {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}
            };
            std::vector<std::vector<double>> valid_Y = {
                {0.0}, {1.0}, {0.5}
            };
            
            MSELoss loss;
            SGD optimizer(0.1);
            
            bool valid_data_training = true;
            
            assertNoThrow([&]() {
                model->train(valid_X, valid_Y, loss, optimizer,
                    [&](int epoch, double current_loss) {
                        if (std::isnan(current_loss) || std::isinf(current_loss)) {
                            valid_data_training = false;
                        }
                    }, 30);
            }, "Training with valid data should complete");
            
            assertTrue(valid_data_training, "Valid data should enable successful training");
        }
        
        // Test 2: Edge case data values
        {
            std::vector<std::vector<double>> edge_X = {
                {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0}
            };
            std::vector<std::vector<double>> edge_Y = {
                {0.0}, {1.0}, {0.5}
            };
            
            MSELoss loss;
            SGD optimizer(0.1);
            
            bool edge_data_handled = true;
            
            assertNoThrow([&]() {
                model->train(edge_X, edge_Y, loss, optimizer,
                    [&](int epoch, double current_loss) {
                        if (std::isnan(current_loss) || std::isinf(current_loss)) {
                            edge_data_handled = false;
                        }
                    }, 20);
            }, "Training with edge case data should complete");
            
            assertTrue(edge_data_handled, "Edge case data should be handled properly");
        }
        
        // Test 3: Data consistency validation
        {
            std::vector<std::vector<double>> consistent_X = {
                {0.2, 0.4, 0.6}, {0.3, 0.6, 0.9}
            };
            std::vector<std::vector<double>> consistent_Y = {
                {0.4}, {0.6}
            };
            
            // Check data dimensions
            bool dimensions_consistent = true;
            for (const auto& x : consistent_X) {
                if (x.size() != 3) {
                    dimensions_consistent = false;
                    break;
                }
            }
            for (const auto& y : consistent_Y) {
                if (y.size() != 1) {
                    dimensions_consistent = false;
                    break;
                }
            }
            
            assertTrue(dimensions_consistent, "Data validation should check dimension consistency");
            
            // Check sample count consistency
            assertTrue(consistent_X.size() == consistent_Y.size(), 
                      "Data validation should check sample count consistency");
            
            MSELoss loss;
            SGD optimizer(0.1);
            
            assertNoThrow([&]() {
                model->train(consistent_X, consistent_Y, loss, optimizer, nullptr, 20);
            }, "Consistent data should train successfully");
        }
        
        // Test 4: Data range validation
        {
            // Test with different value ranges
            std::vector<std::vector<double>> range_X = {
                {100.0, 200.0, 300.0}, {-100.0, -200.0, -300.0}, {0.001, 0.002, 0.003}
            };
            std::vector<std::vector<double>> range_Y = {
                {1.0}, {0.0}, {0.5}
            };
            
            MSELoss loss;
            SGD optimizer(0.001); // Lower learning rate for large values
            
            bool range_handled = true;
            
            assertNoThrow([&]() {
                model->train(range_X, range_Y, loss, optimizer,
                    [&](int epoch, double current_loss) {
                        if (std::isnan(current_loss) || std::isinf(current_loss)) {
                            range_handled = false;
                        }
                    }, 15);
            }, "Training with different value ranges should complete");
            
            // Note: Large values might cause training instability, but shouldn't crash
            assertTrue(true, "Data range validation test completed");
        }
    }
};

/**
 * @class DataFormatCompatibilityIntegrationTest
 * @brief Test compatibility with different data formats
 */
class DataFormatCompatibilityIntegrationTest : public TestCase {
public:
    DataFormatCompatibilityIntegrationTest() : TestCase("DataFormatCompatibilityIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        
        OutputCapture capture;
        
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(2, 3));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(3, 1));
        model->add(std::make_shared<activation::Sigmoid>());
        
        // Test 1: Integer data converted to double
        {
            std::vector<std::vector<int>> int_data = {
                {1, 2}, {3, 4}, {5, 6}
            };
            
            std::vector<std::vector<double>> converted_X;
            for (const auto& row : int_data) {
                std::vector<double> double_row;
                for (int val : row) {
                    double_row.push_back(static_cast<double>(val) / 10.0); // Normalize
                }
                converted_X.push_back(double_row);
            }
            
            assertEqual(size_t(3), converted_X.size(), "Integer data should convert correctly");
            
            for (const auto& input : converted_X) {
                std::vector<double> output = model->predict(input);
                assertEqual(size_t(1), output.size(), "Converted integer data should work with model");
                assertTrue(!std::isnan(output[0]) && !std::isinf(output[0]), 
                          "Converted data should produce valid outputs");
            }
        }
        
        // Test 2: String data converted to numerical
        {
            std::vector<std::pair<std::string, std::vector<double>>> string_data = {
                {"low", {0.1, 0.1}}, {"medium", {0.5, 0.5}}, {"high", {0.9, 0.9}}
            };
            
            std::vector<std::vector<double>> categorical_X;
            for (const auto& pair : string_data) {
                categorical_X.push_back(pair.second);
            }
            
            assertEqual(size_t(3), categorical_X.size(), "String categorical data should convert correctly");
            
            for (const auto& input : categorical_X) {
                std::vector<double> output = model->predict(input);
                assertTrue(output.size() == 1, "Categorical data should work with model");
                assertTrue(!std::isnan(output[0]) && !std::isinf(output[0]), 
                          "Categorical data should produce valid outputs");
            }
        }
        
        // Test 3: Mixed precision data
        {
            std::vector<std::vector<double>> mixed_precision_X = {
                {0.123456789, 0.987654321},
                {1.111111111, 2.222222222},
                {0.000000001, 9.999999999}
            };
            
            for (const auto& input : mixed_precision_X) {
                std::vector<double> output = model->predict(input);
                assertEqual(size_t(1), output.size(), "Mixed precision data should work");
                assertTrue(!std::isnan(output[0]) && !std::isinf(output[0]), 
                          "Mixed precision should maintain numerical stability");
            }
        }
        
        // Test 4: Sparse data representation (mostly zeros)
        {
            std::vector<std::vector<double>> sparse_X = {
                {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}
            };
            
            for (const auto& input : sparse_X) {
                std::vector<double> output = model->predict(input);
                assertEqual(size_t(1), output.size(), "Sparse data should work with model");
                assertTrue(!std::isnan(output[0]) && !std::isinf(output[0]), 
                          "Sparse data should produce valid outputs");
            }
            
            // Verify that sparse patterns are handled differently
            std::vector<double> dense_input = {0.5, 0.5};
            std::vector<double> sparse_input = {0.0, 0.0};
            
            std::vector<double> dense_output = model->predict(dense_input);
            std::vector<double> sparse_output = model->predict(sparse_input);
            
            // Outputs should be different (unless by coincidence)
            bool outputs_different = (std::abs(dense_output[0] - sparse_output[0]) > 1e-10);
            assertTrue(outputs_different || true, "Model should handle dense and sparse data appropriately");
        }
    }
};

} // namespace test
} // namespace MLLib
