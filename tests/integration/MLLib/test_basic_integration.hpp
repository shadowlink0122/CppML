#pragma once

#include "../../common/test_utils.hpp"
#include "../../../include/MLLib.hpp"
#include <memory>

/**
 * @file test_basic_integration.hpp
 * @brief Basic integration tests for MLLib
 * 
 * Tests basic functionality integration:
 * - Simple training workflows
 * - Model save/load operations
 * - End-to-end prediction pipeline
 */

namespace MLLib {
namespace test {

/**
 * @class BasicTrainingIntegrationTest
 * @brief Test basic training workflow
 */
class BasicTrainingIntegrationTest : public TestCase {
public:
    BasicTrainingIntegrationTest() : TestCase("BasicTrainingIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        
        // Create simple model
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
        
        MLLib::loss::MSELoss loss;
        MLLib::optimizer::SGD optimizer(0.5);
        
        bool training_completed = false;
        assertNoThrow([&]() {
            model->train(X, Y, loss, optimizer, nullptr, 100);
            training_completed = true;
        }, "Basic training should complete without errors");
        
        assertTrue(training_completed, "Training should complete successfully");
        
        // Test prediction
        std::vector<double> pred = model->predict({0.5, 0.5});
        assertEqual(size_t(1), pred.size(), "Prediction should have 1 output");
        assertTrue(!std::isnan(pred[0]) && !std::isinf(pred[0]), "Prediction should be valid");
    }
};

/**
 * @class ModelSaveLoadIntegrationTest
 * @brief Test model save/load workflow
 */
class ModelSaveLoadIntegrationTest : public TestCase {
public:
    ModelSaveLoadIntegrationTest() : TestCase("ModelSaveLoadIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        std::string temp_dir = createTempDirectory();
        
        // Create and train model
        auto model = std::make_unique<Sequential>();
        model->add(std::make_shared<Dense>(3, 5));
        model->add(std::make_shared<activation::ReLU>());
        model->add(std::make_shared<Dense>(5, 2));
        model->add(std::make_shared<activation::Sigmoid>());
        
        std::vector<std::vector<double>> X = {
            {1.0, 0.0, 0.5}, {0.0, 1.0, 0.3}, {0.5, 0.5, 1.0}
        };
        std::vector<std::vector<double>> Y = {
            {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}
        };
        
        MLLib::loss::MSELoss loss;
        MLLib::optimizer::SGD optimizer(0.1);
        model->train(X, Y, loss, optimizer, nullptr, 50);
        
        // Get original prediction
        std::vector<double> original_pred = model->predict({0.5, 0.5, 0.5});
        
        // Save model
        std::string model_path = temp_dir + "/test_model.bin";
        assertTrue(model::ModelIO::save_model(*model, model_path, model::ModelFormat::BINARY),
                  "Model save should succeed");
        
        // Load model
        auto loaded_model = model::ModelIO::load_model(model_path, model::ModelFormat::BINARY);
        assertNotNull(loaded_model.get(), "Model load should succeed");
        
        // Test that loaded model produces same result
        std::vector<double> loaded_pred = loaded_model->predict({0.5, 0.5, 0.5});
        assertEqual(original_pred.size(), loaded_pred.size(), "Predictions should have same size");
        
        for (size_t i = 0; i < original_pred.size(); ++i) {
            assertTrue(std::abs(original_pred[i] - loaded_pred[i]) < 1e-6,
                      "Loaded model should produce nearly identical results");
        }
        
        removeTempDirectory(temp_dir);
    }
};

/**
 * @class FullWorkflowIntegrationTest
 * @brief Test complete workflow from training to deployment
 */
class FullWorkflowIntegrationTest : public TestCase {
public:
    FullWorkflowIntegrationTest() : TestCase("FullWorkflowIntegrationTest") {}

protected:
    void test() override {
        using namespace MLLib::model;
        using namespace MLLib::layer;
        using namespace MLLib::loss;
        using namespace MLLib::optimizer;
        
        OutputCapture capture;
        std::string temp_dir = createTempDirectory();
        
        // Phase 1: Model development
        auto dev_model = std::make_unique<Sequential>();
        dev_model->add(std::make_shared<Dense>(4, 8));
        dev_model->add(std::make_shared<activation::ReLU>());
        dev_model->add(std::make_shared<Dense>(8, 4));
        dev_model->add(std::make_shared<activation::Tanh>());
        dev_model->add(std::make_shared<Dense>(4, 2));
        dev_model->add(std::make_shared<activation::Sigmoid>());
        
        // Training data
        std::vector<std::vector<double>> train_X;
        std::vector<std::vector<double>> train_Y;
        
        for (int i = 0; i < 20; ++i) {
            std::vector<double> x = {
                i * 0.05, (i + 1) * 0.05, (i + 2) * 0.05, (i + 3) * 0.05
            };
            std::vector<double> y = {
                static_cast<double>(i % 2), static_cast<double>((i + 1) % 2)
            };
            train_X.push_back(x);
            train_Y.push_back(y);
        }
        
        MLLib::loss::MSELoss loss;
        MLLib::optimizer::SGD optimizer(0.1);
        
        // Phase 2: Training
        assertNoThrow([&]() {
            dev_model->train(train_X, train_Y, loss, optimizer, nullptr, 100);
        }, "Training phase should complete");
        
        // Phase 3: Model export
        std::string model_path = temp_dir + "/production_model.bin";
        std::string config_path = temp_dir + "/production_model.config";
        
        assertTrue(ModelIO::save_model(*dev_model, model_path, ModelFormat::BINARY),
                  "Production model save should succeed");
        assertTrue(ModelIO::save_config(*dev_model, config_path),
                  "Production config save should succeed");
        
        // Phase 4: Production deployment simulation
        auto prod_model = ModelIO::load_model(model_path, ModelFormat::BINARY);
        assertNotNull(prod_model.get(), "Production model load should succeed");
        
        // Phase 5: Production inference
        std::vector<std::vector<double>> prod_inputs = {
            {0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 0.8, 0.7, 0.6}
        };
        
        for (const auto& input : prod_inputs) {
            std::vector<double> output = prod_model->predict(input);
            assertEqual(size_t(2), output.size(), "Production output should have 2 values");
            
            for (double val : output) {
                assertTrue(!std::isnan(val) && !std::isinf(val), "Output values should be valid");
                assertTrue(val >= 0.0 && val <= 1.0, "Sigmoid output should be in [0,1]");
            }
        }
        
        // Phase 6: Model versioning
        std::string v2_path = temp_dir + "/model_v2.bin";
        
        // Simulate continued training
        assertNoThrow([&]() {
            dev_model->train(train_X, train_Y, loss, optimizer, nullptr, 50);
        }, "Continued training should work");
        
        assertTrue(ModelIO::save_model(*dev_model, v2_path, ModelFormat::BINARY),
                  "Version 2 save should succeed");
        
        auto v2_model = ModelIO::load_model(v2_path, ModelFormat::BINARY);
        assertNotNull(v2_model.get(), "Version 2 load should succeed");
        
        removeTempDirectory(temp_dir);
    }
};

} // namespace test
} // namespace MLLib
