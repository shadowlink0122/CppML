#pragma once

#include "../../common/test_utils.hpp"
#include "../../../include/MLLib.hpp"
#include <memory>
#include <vector>

/**
 * @file test_model_integration.hpp
 * @brief Model integration tests for MLLib
 * 
 * Tests the complete integration of model components including:
 * - Sequential model with multiple layers
 * - Training workflows with different optimizers and loss functions
 * - Model I/O operations (save/load)
 * - Prediction pipelines
 */

namespace MLLib {
namespace test {

/**
 * @class SequentialModelIntegrationTest
 * @brief Test Sequential model integration with various layer combinations
 */
class SequentialModelIntegrationTest : public TestCase {
public:
    SequentialModelIntegrationTest() : TestCase("SequentialModelIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        
        OutputCapture capture;
        
        // Test 1: Simple 2-layer network integration
        {
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(3, 5));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(5, 2));
            
            assertEqual(size_t(3), model->num_layers(), "Model should have 3 layers");
            
            // Test forward pass
            std::vector<double> input = {1.0, 0.5, -0.3};
            std::vector<double> output = model->predict(input);
            
            assertEqual(size_t(2), output.size(), "Output should have 2 elements");
            assertTrue(output[0] != 0.0 || output[1] != 0.0, "Output should not be zero");
        }
        
        // Test 2: Deep network with different activations
        {
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(4, 8));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(8, 6));
            model->add(std::make_shared<activation::Tanh>());
            model->add(std::make_shared<Dense>(6, 4));
            model->add(std::make_shared<activation::Sigmoid>());
            model->add(std::make_shared<Dense>(4, 1));
            
            assertEqual(size_t(7), model->num_layers(), "Deep model should have 7 layers");
            
            // Test batch prediction
            std::vector<std::vector<double>> batch_inputs = {
                {1.0, 0.0, 0.5, -0.2},
                {0.0, 1.0, -0.5, 0.3},
                {-1.0, 0.5, 0.0, 0.8}
            };
            
            for (const auto& input : batch_inputs) {
                std::vector<double> output = model->predict(input);
                assertEqual(size_t(1), output.size(), "Each output should have 1 element");
                assertTrue(output[0] >= 0.0 && output[0] <= 1.0, "Sigmoid output should be in [0,1]");
            }
        }
        
        // Test 3: Model layer access and information
        {
            auto model = std::make_unique<Sequential>();
            auto dense1 = std::make_shared<Dense>(2, 3);
            auto relu = std::make_shared<activation::ReLU>();
            auto dense2 = std::make_shared<Dense>(3, 1);
            
            model->add(dense1);
            model->add(relu);
            model->add(dense2);
            
            assertEqual(size_t(3), model->num_layers(), "Model should track all layers");
            assertTrue(model->num_layers() > 0, "Model should have layers");
        }
    }
};

/**
 * @class TrainingIntegrationTest
 * @brief Test complete training workflows
 */
class TrainingIntegrationTest : public TestCase {
public:
    TrainingIntegrationTest() : TestCase("TrainingIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Test 1: Simple classification training
        {
            auto model = std::make_unique<Sequential>();
            model->add(std::make_shared<Dense>(2, 4));
            model->add(std::make_shared<activation::ReLU>());
            model->add(std::make_shared<Dense>(4, 1));
            model->add(std::make_shared<activation::Sigmoid>());
            
            // Simple binary classification data
            std::vector<std::vector<double>> X = {
                {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
            };
            std::vector<std::vector<double>> Y = {
                {0.0}, {1.0}, {1.0}, {0.0}
            };
            
            MSE loss;
            SGD optimizer(0.5);
            
            bool training_completed = false;
            double initial_loss = 0.0;
            double final_loss = 0.0;
            int epoch_count = 0;
            
            assertNoThrow([&]() {
                model->train(X, Y, loss, optimizer, 
                    [&](int epoch, double current_loss) {
                        if (epoch == 0) initial_loss = current_loss;
                        final_loss = current_loss;
                        epoch_count = epoch;
                        
                        // Verify loss is reasonable
                        assertTrue(!std::isnan(current_loss), "Loss should not be NaN");
                        assertTrue(!std::isinf(current_loss), "Loss should not be infinite");
                        assertTrue(current_loss >= 0.0, "Loss should be non-negative");
                    }, 100);
                training_completed = true;
            }, "Training should complete without errors");
            
            assertTrue(training_completed, "Training should complete");
            assertEqual(99, epoch_count, "Should complete all epochs");
            assertTrue(final_loss <= initial_loss * 1.1, "Loss should generally decrease or stabilize");
        }
        
        // Test 2: Training with different optimizers
        {
            auto model1 = std::make_unique<Sequential>();
            model1->add(std::make_shared<Dense>(3, 4));
            model1->add(std::make_shared<activation::Tanh>());
            model1->add(std::make_shared<Dense>(4, 2));
            
            std::vector<std::vector<double>> X = {
                {1.0, 0.0, 0.5}, {0.0, 1.0, 0.3}, {0.5, 0.5, 1.0}
            };
            std::vector<std::vector<double>> Y = {
                {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}
            };
            
            MSE loss;
            
            // Test SGD with different learning rates
            {
                SGD optimizer1(0.01);  // Low learning rate
                assertNoThrow([&]() {
                    model1->train(X, Y, loss, optimizer1, nullptr, 50);
                }, "Training with low learning rate should work");
            }
            
            {
                SGD optimizer2(0.1);  // Higher learning rate
                assertNoThrow([&]() {
                    model1->train(X, Y, loss, optimizer2, nullptr, 50);
                }, "Training with higher learning rate should work");
            }
        }
    }
};

/**
 * @class ModelIOIntegrationTest
 * @brief Test model save/load integration workflows
 */
class ModelIOIntegrationTest : public TestCase {
public:
    ModelIOIntegrationTest() : TestCase("ModelIOIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        
        OutputCapture capture;
        
        // Create and train a reference model
        auto original_model = std::make_unique<Sequential>();
        original_model->add(std::make_shared<Dense>(2, 3));
        original_model->add(std::make_shared<activation::ReLU>());
        original_model->add(std::make_shared<Dense>(3, 1));
        original_model->add(std::make_shared<activation::Sigmoid>());
        
        // Quick training to establish weights
        std::vector<std::vector<double>> X = {
            {0.5, 0.3}, {0.8, 0.1}, {0.2, 0.9}
        };
        std::vector<std::vector<double>> Y = {
            {0.7}, {0.4}, {0.6}
        };
        
        MLLib::loss::MSE loss;
        MLLib::optimizer::SGD optimizer(0.1);
        original_model->train(X, Y, loss, optimizer, nullptr, 30);
        
        // Get reference predictions
        std::vector<double> test_input = {0.6, 0.4};
        std::vector<double> original_pred = original_model->predict(test_input);
        
        // Test file operations
        std::string temp_dir = createTempDirectory();
        
        // Test 1: Binary format save/load
        {
            std::string binary_path = temp_dir + "/model_integration.bin";
            
            assertTrue(ModelIO::save_model(*original_model, binary_path, ModelFormat::BINARY),
                      "Binary save should succeed");
            
            auto loaded_model = ModelIO::load_model(binary_path, ModelFormat::BINARY);
            assertNotNull(loaded_model.get(), "Binary load should succeed");
            
            std::vector<double> loaded_pred = loaded_model->predict(test_input);
            assertVectorNear(original_pred, loaded_pred, 1e-10,
                            "Binary format should preserve exact predictions");
            
            assertEqual(original_model->num_layers(), loaded_model->num_layers(),
                       "Binary format should preserve model structure");
        }
        
        // Test 2: JSON format save/load
        {
            std::string json_path = temp_dir + "/model_integration.json";
            
            assertTrue(ModelIO::save_model(*original_model, json_path, ModelFormat::JSON),
                      "JSON save should succeed");
            
            auto loaded_model = ModelIO::load_model(json_path, ModelFormat::JSON);
            assertNotNull(loaded_model.get(), "JSON load should succeed");
            
            std::vector<double> loaded_pred = loaded_model->predict(test_input);
            assertVectorNear(original_pred, loaded_pred, 1e-6,
                            "JSON format should preserve predictions with reasonable precision");
        }
        
        // Test 3: Config format (architecture only)
        {
            std::string config_path = temp_dir + "/model_integration.config";
            
            assertTrue(ModelIO::save_config(*original_model, config_path),
                      "Config save should succeed");
            
            auto loaded_model = ModelIO::load_config(config_path);
            assertNotNull(loaded_model.get(), "Config load should succeed");
            
            assertEqual(original_model->num_layers(), loaded_model->num_layers(),
                       "Config should preserve model architecture");
            
            // Config-loaded model should work for new training
            assertNoThrow([&]() {
                loaded_model->train(X, Y, loss, optimizer, nullptr, 10);
            }, "Config-loaded model should be trainable");
        }
        
        // Test 4: Multiple format workflow
        {
            std::string multi_dir = temp_dir + "/multi_format";
            createTempDirectory(multi_dir);
            
            // Save in all formats
            assertTrue(ModelIO::save_model(*original_model, multi_dir + "/model.bin", ModelFormat::BINARY),
                      "Multi-format binary save should work");
            assertTrue(ModelIO::save_model(*original_model, multi_dir + "/model.json", ModelFormat::JSON),
                      "Multi-format JSON save should work");
            assertTrue(ModelIO::save_config(*original_model, multi_dir + "/model.config"),
                      "Multi-format config save should work");
            
            // Load and verify each format independently
            auto bin_model = ModelIO::load_model(multi_dir + "/model.bin", ModelFormat::BINARY);
            auto json_model = ModelIO::load_model(multi_dir + "/model.json", ModelFormat::JSON);
            auto config_model = ModelIO::load_config(multi_dir + "/model.config");
            
            assertNotNull(bin_model.get(), "Multi-format binary load should work");
            assertNotNull(json_model.get(), "Multi-format JSON load should work");
            assertNotNull(config_model.get(), "Multi-format config load should work");
            
            // Verify structural consistency
            assertEqual(bin_model->num_layers(), json_model->num_layers(),
                       "Binary and JSON should have same structure");
            assertEqual(json_model->num_layers(), config_model->num_layers(),
                       "JSON and config should have same structure");
        }
        
        // Cleanup
        removeTempDirectory(temp_dir);
    }
};

} // namespace test
} // namespace MLLib
