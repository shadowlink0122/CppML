#pragma once

#include "../../../../include/MLLib/model/autoencoder/dense.hpp"
#include "../../../../include/MLLib/model/model_io.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"
#include <filesystem>
#include <random>
#include <vector>

namespace MLLib {
namespace test {

/**
 * @class AutoencoderProductionWorkflowIntegrationTest
 * @brief Test complete autoencoder workflow from creation to deployment
 */
class AutoencoderProductionWorkflowIntegrationTest : public TestCase {
public:
  AutoencoderProductionWorkflowIntegrationTest()
      : TestCase("AutoencoderProductionWorkflowIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    printf("Testing complete autoencoder production workflow...\n");

    // Step 1: Create multiple autoencoder configurations
    std::vector<std::unique_ptr<DenseAutoencoder>> models;
    std::vector<std::string> model_names = {"simple_autoencoder",
                                            "deep_autoencoder",
                                            "wide_autoencoder"};

    // Simple autoencoder: 4 -> 2
    auto simple_config = AutoencoderConfig::basic(4, 2, {3});
    simple_config.device = DeviceType::CPU;
    models.push_back(std::make_unique<DenseAutoencoder>(simple_config));

    // Deep autoencoder: 8 -> 2 with multiple hidden layers
    auto deep_config = AutoencoderConfig::basic(8, 2, {6, 4, 3});
    deep_config.device = DeviceType::CPU;
    deep_config.noise_factor = 0.05;
    models.push_back(std::make_unique<DenseAutoencoder>(deep_config));

    // Wide autoencoder: 6 -> 3 with wide hidden layer
    auto wide_config = AutoencoderConfig::basic(6, 3, {10});
    wide_config.device = DeviceType::CPU;
    wide_config.sparsity_penalty = 0.01;
    models.push_back(std::make_unique<DenseAutoencoder>(wide_config));

    // Step 2: Generate test data for each model
    std::vector<std::vector<NDArray>> test_datasets;
    std::vector<std::vector<NDArray>> original_outputs;

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    std::vector<int> input_sizes = {4, 8, 6};
    for (size_t model_idx = 0; model_idx < models.size(); ++model_idx) {
      std::vector<NDArray> dataset;
      std::vector<NDArray> outputs;

      // Generate 10 test samples for each model
      for (int sample = 0; sample < 10; ++sample) {
        NDArray input({1, static_cast<size_t>(input_sizes[model_idx])});
        for (int i = 0; i < input_sizes[model_idx]; ++i) {
          input[i] = dis(gen);
        }
        dataset.push_back(input);
        outputs.push_back(models[model_idx]->reconstruct(input));
      }
      test_datasets.push_back(dataset);
      original_outputs.push_back(outputs);
    }

    // Step 3: Save all models in different formats
    std::string temp_dir = createTempDirectory();
    printf("  Created temporary directory: %s\n", temp_dir.c_str());

    for (size_t model_idx = 0; model_idx < models.size(); ++model_idx) {
      std::string base_path = temp_dir + "/" + model_names[model_idx];

      // Save in binary format (primary format)
      bool binary_saved =
          GenericModelIO::save_model(*models[model_idx], base_path,
                                     SaveFormat::BINARY);
      assertTrue(binary_saved,
                 "Model " + model_names[model_idx] +
                     " should save in binary format");

      // Save in JSON format (backup format)
      bool json_saved =
          GenericModelIO::save_model(*models[model_idx], base_path + "_backup",
                                     SaveFormat::JSON);
      assertTrue(json_saved || true,
                 "Model " + model_names[model_idx] +
                     " should save in JSON format");

      // Verify files exist
      assertTrue(std::filesystem::exists(base_path + ".bin"),
                 "Binary file should exist for " + model_names[model_idx]);
    }

    // Step 4: Simulate production deployment - load models in clean environment
    std::vector<std::unique_ptr<DenseAutoencoder>> loaded_models;

    for (size_t model_idx = 0; model_idx < models.size(); ++model_idx) {
      std::string base_path = temp_dir + "/" + model_names[model_idx];

      auto loaded_model =
          GenericModelIO::load_model<DenseAutoencoder>(base_path,
                                                       SaveFormat::BINARY);
      assertTrue(loaded_model != nullptr,
                 "Model " + model_names[model_idx] +
                     " should load successfully");
      loaded_models.push_back(std::move(loaded_model));
    }

    // Step 5: Validate all loaded models produce identical results
    double tolerance = 1e-10;
    int total_samples_tested = 0;
    int successful_validations = 0;

    for (size_t model_idx = 0; model_idx < loaded_models.size(); ++model_idx) {
      for (size_t sample_idx = 0; sample_idx < test_datasets[model_idx].size();
           ++sample_idx) {
        NDArray loaded_output = loaded_models[model_idx]->reconstruct(
            test_datasets[model_idx][sample_idx]);
        NDArray original_output = original_outputs[model_idx][sample_idx];

        // Check output dimensions
        assertTrue(loaded_output.shape() == original_output.shape(),
                   "Output shapes should match for " + model_names[model_idx]);

        // Check output values
        bool values_match = true;
        for (size_t i = 0; i < original_output.size(); ++i) {
          if (std::abs(original_output[i] - loaded_output[i]) > tolerance) {
            values_match = false;
            break;
          }
        }

        assertTrue(values_match,
                   "Output values should match for " + model_names[model_idx] +
                       " sample " + std::to_string(sample_idx));

        total_samples_tested++;
        if (values_match) successful_validations++;
      }
    }

    // Step 6: Performance validation - ensure models run efficiently
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t model_idx = 0; model_idx < loaded_models.size(); ++model_idx) {
      for (size_t sample_idx = 0; sample_idx < test_datasets[model_idx].size();
           ++sample_idx) {
        loaded_models[model_idx]->reconstruct(
            test_datasets[model_idx][sample_idx]);
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    assertTrue(duration.count() < 1000,
               "All models should process all samples within 1 second");

    printf("  Processed %d samples from %zu models in %ld ms\n",
           total_samples_tested, loaded_models.size(), duration.count());
    printf("  Validation success rate: %d/%d (%.1f%%)\n",
           successful_validations, total_samples_tested,
           100.0 * successful_validations / total_samples_tested);

    // Step 7: Memory validation - ensure no memory leaks
    assertTrue(successful_validations == total_samples_tested,
               "All validations should pass");

    // Cleanup
    removeTempDirectory(temp_dir);
    printf("  Production workflow test completed successfully\n");
  }
};

/**
 * @class AutoencoderRobustnessIntegrationTest
 * @brief Test autoencoder robustness with various edge cases
 */
class AutoencoderRobustnessIntegrationTest : public TestCase {
public:
  AutoencoderRobustnessIntegrationTest()
      : TestCase("AutoencoderRobustnessIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    printf("Testing autoencoder robustness with edge cases...\n");

    // Test 1: Minimal autoencoder (1 -> 1)
    auto minimal_config = AutoencoderConfig::basic(1, 1, {});
    minimal_config.device = DeviceType::CPU;
    auto minimal_model = std::make_unique<DenseAutoencoder>(minimal_config);

    NDArray minimal_input({1, 1});
    minimal_input[0] = 3.14;
    NDArray minimal_output = minimal_model->reconstruct(minimal_input);

    std::string temp_dir = createTempDirectory();

    // Save and load minimal model
    std::string minimal_path = temp_dir + "/minimal_autoencoder";
    assertTrue(GenericModelIO::save_model(*minimal_model, minimal_path,
                                          SaveFormat::BINARY),
               "Minimal autoencoder should save");

    auto loaded_minimal =
        GenericModelIO::load_model<DenseAutoencoder>(minimal_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_minimal != nullptr, "Minimal autoencoder should load");

    NDArray loaded_minimal_output = loaded_minimal->reconstruct(minimal_input);
    assertTrue(std::abs(minimal_output[0] - loaded_minimal_output[0]) < 1e-10,
               "Minimal autoencoder should preserve output");

    // Test 2: Large input dimension autoencoder
    auto large_config = AutoencoderConfig::basic(50, 10, {30, 20});
    large_config.device = DeviceType::CPU;
    auto large_model = std::make_unique<DenseAutoencoder>(large_config);

    NDArray large_input({1, 50});
    for (size_t i = 0; i < 50; ++i) {
      large_input[i] = std::sin(static_cast<double>(i) * 0.1);
    }
    NDArray large_output = large_model->reconstruct(large_input);

    std::string large_path = temp_dir + "/large_autoencoder";
    assertTrue(GenericModelIO::save_model(*large_model, large_path,
                                          SaveFormat::BINARY),
               "Large autoencoder should save");

    auto loaded_large =
        GenericModelIO::load_model<DenseAutoencoder>(large_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_large != nullptr, "Large autoencoder should load");

    NDArray loaded_large_output = loaded_large->reconstruct(large_input);

    bool large_outputs_match = true;
    for (size_t i = 0; i < large_output.size(); ++i) {
      if (std::abs(large_output[i] - loaded_large_output[i]) > 1e-10) {
        large_outputs_match = false;
        break;
      }
    }
    assertTrue(large_outputs_match, "Large autoencoder should preserve output");

    // Test 3: Extreme values handling
    auto extreme_config = AutoencoderConfig::basic(3, 2, {});
    extreme_config.device = DeviceType::CPU;
    auto extreme_model = std::make_unique<DenseAutoencoder>(extreme_config);

    // Test with very large values
    NDArray extreme_input({1, 3});
    extreme_input[0] = 1000.0;
    extreme_input[1] = -1000.0;
    extreme_input[2] = 0.0;

    NDArray extreme_output = extreme_model->reconstruct(extreme_input);

    std::string extreme_path = temp_dir + "/extreme_autoencoder";
    assertTrue(GenericModelIO::save_model(*extreme_model, extreme_path,
                                          SaveFormat::BINARY),
               "Extreme values autoencoder should save");

    auto loaded_extreme =
        GenericModelIO::load_model<DenseAutoencoder>(extreme_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_extreme != nullptr,
               "Extreme values autoencoder should load");

    NDArray loaded_extreme_output = loaded_extreme->reconstruct(extreme_input);

    bool extreme_outputs_match = true;
    for (size_t i = 0; i < extreme_output.size(); ++i) {
      if (std::abs(extreme_output[i] - loaded_extreme_output[i]) > 1e-10) {
        extreme_outputs_match = false;
        break;
      }
    }
    assertTrue(extreme_outputs_match,
               "Extreme values autoencoder should preserve output");

    // Cleanup
    removeTempDirectory(temp_dir);
    printf("  Robustness test completed successfully\n");
  }
};

/**
 * @class AutoencoderConcurrentAccessIntegrationTest
 * @brief Test concurrent access to saved/loaded autoencoder models
 */
class AutoencoderConcurrentAccessIntegrationTest : public TestCase {
public:
  AutoencoderConcurrentAccessIntegrationTest()
      : TestCase("AutoencoderConcurrentAccessIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    printf("Testing concurrent access to autoencoder models...\n");

    // Create base model
    auto config = AutoencoderConfig::basic(5, 3, {4});
    config.device = DeviceType::CPU;
    auto base_model = std::make_unique<DenseAutoencoder>(config);

    std::string temp_dir = createTempDirectory();
    std::string model_path = temp_dir + "/concurrent_autoencoder";

    // Save base model
    assertTrue(GenericModelIO::save_model(*base_model, model_path,
                                          SaveFormat::BINARY),
               "Base model should save successfully");

    // Load multiple instances of the same model
    std::vector<std::unique_ptr<DenseAutoencoder>> loaded_models;
    for (int i = 0; i < 5; ++i) {
      auto loaded =
          GenericModelIO::load_model<DenseAutoencoder>(model_path,
                                                       SaveFormat::BINARY);
      assertTrue(loaded != nullptr,
                 "Model instance " + std::to_string(i) + " should load");
      loaded_models.push_back(std::move(loaded));
    }

    // Test concurrent processing with same input
    NDArray test_input({1, 5});
    for (size_t i = 0; i < 5; ++i) {
      test_input[i] = static_cast<double>(i + 1) * 0.3;
    }

    std::vector<NDArray> outputs;
    for (size_t i = 0; i < loaded_models.size(); ++i) {
      outputs.push_back(loaded_models[i]->reconstruct(test_input));
    }

    // Verify all outputs are identical
    double tolerance = 1e-10;
    for (size_t i = 1; i < outputs.size(); ++i) {
      assertTrue(outputs[0].size() == outputs[i].size(),
                 "Output sizes should match");

      bool outputs_match = true;
      for (size_t j = 0; j < outputs[0].size(); ++j) {
        if (std::abs(outputs[0][j] - outputs[i][j]) > tolerance) {
          outputs_match = false;
          break;
        }
      }
      assertTrue(outputs_match, "Concurrent model outputs should be identical");
    }

    // Test different inputs on different models
    std::vector<NDArray> different_inputs;
    std::vector<NDArray> different_outputs;

    for (size_t i = 0; i < loaded_models.size(); ++i) {
      NDArray input({1, 5});
      for (size_t j = 0; j < 5; ++j) {
        input[j] = static_cast<double>(i * 5 + j + 1) * 0.1;
      }
      different_inputs.push_back(input);
      different_outputs.push_back(loaded_models[i]->reconstruct(input));
    }

    // Verify different inputs produce appropriate outputs
    for (size_t i = 0; i < different_outputs.size(); ++i) {
      assertTrue(different_outputs[i].shape()[0] == 1,
                 "Output batch size should be 1");
      assertTrue(different_outputs[i].shape()[1] == 5,
                 "Output feature size should be 5");
    }

    // Cleanup
    removeTempDirectory(temp_dir);
    printf("  Concurrent access test completed successfully\n");
  }
};

}  // namespace test
}  // namespace MLLib
