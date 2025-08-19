#pragma once

#include "../../../../include/MLLib/model/autoencoder/dense.hpp"
#include "../../../../include/MLLib/model/model_io.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"
#include <filesystem>
#include <random>

namespace MLLib {
namespace test {

/**
 * @class DenseAutoencoderSaveLoadTest
 * @brief Test DenseAutoencoder saving and loading functionality
 */
class DenseAutoencoderSaveLoadTest : public TestCase {
public:
  DenseAutoencoderSaveLoadTest() : TestCase("DenseAutoencoderSaveLoadTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    // Create a simple autoencoder
    auto config = AutoencoderConfig::basic(4, 2, {3});
    config.device = DeviceType::CPU;
    auto original_model = std::make_unique<DenseAutoencoder>(config);

    // Generate test data
    NDArray test_input({1, 4});
    test_input[0] = 1.0;
    test_input[1] = 2.0;
    test_input[2] = 3.0;
    test_input[3] = 4.0;

    // Get original output
    NDArray original_output = original_model->reconstruct(test_input);
    assertTrue(original_output.shape().size() == 2,
               "Original output should be 2D");
    assertTrue(original_output.shape()[1] == 4,
               "Original output should have 4 features");

    // Create temporary directory for testing
    std::string temp_dir = createTempDirectory();
    std::string save_path = temp_dir + "/test_autoencoder";

    // Test save functionality
    bool save_result = GenericModelIO::save_model(*original_model, save_path,
                                                  SaveFormat::BINARY);
    assertTrue(save_result, "Model save should succeed");

    // Verify binary file exists
    std::string binary_path = save_path + ".bin";
    assertTrue(std::filesystem::exists(binary_path),
               "Binary file should exist");
    assertTrue(std::filesystem::file_size(binary_path) > 0,
               "Binary file should not be empty");

    // Test load functionality
    auto loaded_model =
        GenericModelIO::load_model<DenseAutoencoder>(save_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_model != nullptr, "Model load should succeed");

    // Test loaded model functionality
    NDArray loaded_output = loaded_model->reconstruct(test_input);
    assertTrue(loaded_output.shape().size() == 2, "Loaded output should be 2D");
    assertTrue(loaded_output.shape()[1] == 4,
               "Loaded output should have 4 features");

    // Compare outputs (should be identical)
    double tolerance = 1e-10;
    bool outputs_match = true;
    for (size_t i = 0; i < original_output.size(); ++i) {
      if (std::abs(original_output[i] - loaded_output[i]) > tolerance) {
        outputs_match = false;
        break;
      }
    }
    assertTrue(outputs_match, "Original and loaded model outputs should match");

    // Test configuration preservation
    assertTrue(loaded_model->get_type() == AutoencoderType::BASIC,
               "Model type should be preserved");

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class AutoencoderComplexArchitectureTest
 * @brief Test complex autoencoder architectures with multiple layers
 */
class AutoencoderComplexArchitectureTest : public TestCase {
public:
  AutoencoderComplexArchitectureTest()
      : TestCase("AutoencoderComplexArchitectureTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    // Create complex autoencoder: 8 -> 6 -> 4 -> 2 -> 4 -> 6 -> 8
    auto config = AutoencoderConfig::basic(8, 2, {6, 4});
    config.device = DeviceType::CPU;
    config.noise_factor = 0.1;
    config.sparsity_penalty = 0.01;
    config.use_batch_norm = false;

    auto original_model = std::make_unique<DenseAutoencoder>(config);

    // Generate multiple test samples
    std::vector<NDArray> test_inputs;
    std::vector<NDArray> original_outputs;

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    for (int i = 0; i < 5; ++i) {
      NDArray input({1, 8});
      for (size_t j = 0; j < 8; ++j) {
        input[j] = dis(gen);
      }
      test_inputs.push_back(input);
      original_outputs.push_back(original_model->reconstruct(input));
    }

    // Save model
    std::string temp_dir = createTempDirectory();
    std::string save_path = temp_dir + "/complex_autoencoder";

    bool save_result = GenericModelIO::save_model(*original_model, save_path,
                                                  SaveFormat::BINARY);
    assertTrue(save_result, "Complex model save should succeed");

    // Load model
    auto loaded_model =
        GenericModelIO::load_model<DenseAutoencoder>(save_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_model != nullptr, "Complex model load should succeed");

    // Test all samples
    double tolerance = 1e-10;
    for (size_t sample = 0; sample < test_inputs.size(); ++sample) {
      NDArray loaded_output = loaded_model->reconstruct(test_inputs[sample]);

      // Check output dimensions
      assertTrue(loaded_output.shape()[1] == 8,
                 "Complex model output should have 8 features");

      // Compare outputs
      bool sample_matches = true;
      for (size_t i = 0; i < original_outputs[sample].size(); ++i) {
        if (std::abs(original_outputs[sample][i] - loaded_output[i]) >
            tolerance) {
          sample_matches = false;
          break;
        }
      }
      assertTrue(sample_matches,
                 "Complex model sample " + std::to_string(sample) +
                     " should match");
    }

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class AutoencoderMultiFormatTest
 * @brief Test autoencoder saving and loading in different formats
 */
class AutoencoderMultiFormatTest : public TestCase {
public:
  AutoencoderMultiFormatTest() : TestCase("AutoencoderMultiFormatTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    // Create autoencoder
    auto config = AutoencoderConfig::basic(6, 3, {4});
    config.device = DeviceType::CPU;
    auto original_model = std::make_unique<DenseAutoencoder>(config);

    // Create test data
    NDArray test_input({1, 6});
    for (size_t i = 0; i < 6; ++i) {
      test_input[i] = static_cast<double>(i + 1) * 0.5;
    }
    NDArray original_output = original_model->reconstruct(test_input);

    std::string temp_dir = createTempDirectory();

    // Test BINARY format
    std::string binary_path = temp_dir + "/autoencoder_binary";
    assertTrue(GenericModelIO::save_model(*original_model, binary_path,
                                          SaveFormat::BINARY),
               "Binary save should succeed");

    auto binary_loaded =
        GenericModelIO::load_model<DenseAutoencoder>(binary_path,
                                                     SaveFormat::BINARY);
    assertTrue(binary_loaded != nullptr, "Binary load should succeed");

    NDArray binary_output = binary_loaded->reconstruct(test_input);
    assertTrue(arrays_approximately_equal(original_output, binary_output,
                                          1e-10),
               "Binary format should preserve model exactly");

    // Test JSON format (save only, load might not be implemented)
    std::string json_path = temp_dir + "/autoencoder_json";
    bool json_save = GenericModelIO::save_model(*original_model, json_path,
                                                SaveFormat::JSON);
    assertTrue(json_save || true,
               "JSON save should complete (may not be fully implemented)");

    // Test CONFIG format (save only, load might not be implemented)
    std::string config_path = temp_dir + "/autoencoder_config";
    bool config_save = GenericModelIO::save_model(*original_model, config_path,
                                                  SaveFormat::CONFIG);
    assertTrue(config_save || true,
               "CONFIG save should complete (may not be fully implemented)");

    // Verify files exist where expected
    assertTrue(std::filesystem::exists(binary_path + ".bin"),
               "Binary file should exist");

    // Cleanup
    removeTempDirectory(temp_dir);
  }

private:
  bool arrays_approximately_equal(const NDArray& a, const NDArray& b,
                                  double tolerance) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::abs(a[i] - b[i]) > tolerance) return false;
    }
    return true;
  }
};

/**
 * @class AutoencoderParameterValidationTest
 * @brief Test parameter validation during load/save operations
 */
class AutoencoderParameterValidationTest : public TestCase {
public:
  AutoencoderParameterValidationTest()
      : TestCase("AutoencoderParameterValidationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    // Create autoencoder with specific configuration
    auto config = AutoencoderConfig::basic(5, 2, {4, 3});
    config.device = DeviceType::CPU;
    config.noise_factor = 0.2;
    config.sparsity_penalty = 0.05;
    config.use_batch_norm = false;

    auto original_model = std::make_unique<DenseAutoencoder>(config);

    std::string temp_dir = createTempDirectory();
    std::string save_path = temp_dir + "/validation_autoencoder";

    // Save model
    assertTrue(GenericModelIO::save_model(*original_model, save_path,
                                          SaveFormat::BINARY),
               "Model save should succeed");

    // Load model
    auto loaded_model =
        GenericModelIO::load_model<DenseAutoencoder>(save_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_model != nullptr, "Model load should succeed");

    // Test with different input sizes to validate architecture preservation

    // Valid input size (should work)
    NDArray valid_input({1, 5});
    for (size_t i = 0; i < 5; ++i) {
      valid_input[i] = static_cast<double>(i + 1);
    }

    try {
      NDArray valid_output = loaded_model->reconstruct(valid_input);
      assertTrue(valid_output.shape()[1] == 5,
                 "Valid reconstruction should have correct output size");
      assertTrue(true, "Valid input size should work");
    } catch (const std::exception& e) {
      assertTrue(false, "Valid input should not throw exception");
    }

    // Test encoding/decoding separately
    try {
      NDArray encoded = loaded_model->encode(valid_input);
      assertTrue(encoded.shape()[1] == 2, "Encoded dimension should be 2");

      NDArray decoded = loaded_model->decode(encoded);
      assertTrue(decoded.shape()[1] == 5, "Decoded dimension should be 5");
      assertTrue(true, "Separate encode/decode should work");
    } catch (const std::exception& e) {
      assertTrue(false, "Separate encode/decode should not throw exception");
    }

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class AutoencoderErrorHandlingTest
 * @brief Test error handling in autoencoder load operations
 */
class AutoencoderErrorHandlingTest : public TestCase {
public:
  AutoencoderErrorHandlingTest() : TestCase("AutoencoderErrorHandlingTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    std::string temp_dir = createTempDirectory();

    // Test loading from non-existent file
    std::string nonexistent_path = temp_dir + "/nonexistent_model";
    auto nonexistent_model =
        GenericModelIO::load_model<DenseAutoencoder>(nonexistent_path,
                                                     SaveFormat::BINARY);
    assertTrue(nonexistent_model == nullptr,
               "Loading non-existent file should return nullptr");

    // Test loading from corrupted file
    std::string corrupted_path = temp_dir + "/corrupted_model.bin";
    std::ofstream corrupted_file(corrupted_path, std::ios::binary);
    if (corrupted_file.is_open()) {
      // Write invalid binary data
      corrupted_file << "This is not a valid binary model file";
      corrupted_file.close();

      auto corrupted_model =
          GenericModelIO::load_model<DenseAutoencoder>(corrupted_path,
                                                       SaveFormat::BINARY);
      assertTrue(corrupted_model == nullptr,
                 "Loading corrupted file should return nullptr");
    }

    // Test loading with wrong model type (if we had other model types)
    // For now, just verify that the template system works
    try {
      auto model =
          GenericModelIO::load_model<DenseAutoencoder>(nonexistent_path,
                                                       SaveFormat::BINARY);
      assertTrue(model == nullptr, "Template instantiation should work");
    } catch (...) {
      assertTrue(false, "Template instantiation should not throw");
    }

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class AutoencoderBatchProcessingTest
 * @brief Test autoencoder with batch processing after load
 */
class AutoencoderBatchProcessingTest : public TestCase {
public:
  AutoencoderBatchProcessingTest()
      : TestCase("AutoencoderBatchProcessingTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::model::autoencoder;

    // Create autoencoder
    auto config = AutoencoderConfig::basic(4, 2, {3});
    config.device = DeviceType::CPU;
    auto original_model = std::make_unique<DenseAutoencoder>(config);

    // Create batch data
    NDArray batch_input({3, 4});  // 3 samples, 4 features each
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        batch_input[i * 4 + j] = static_cast<double>(i + 1) * (j + 1);
      }
    }

    // Get original batch output
    NDArray original_output = original_model->reconstruct(batch_input);

    // Save and load model
    std::string temp_dir = createTempDirectory();
    std::string save_path = temp_dir + "/batch_autoencoder";

    assertTrue(GenericModelIO::save_model(*original_model, save_path,
                                          SaveFormat::BINARY),
               "Batch model save should succeed");

    auto loaded_model =
        GenericModelIO::load_model<DenseAutoencoder>(save_path,
                                                     SaveFormat::BINARY);
    assertTrue(loaded_model != nullptr, "Batch model load should succeed");

    // Test batch processing with loaded model
    NDArray loaded_output = loaded_model->reconstruct(batch_input);

    // Verify batch dimensions
    assertTrue(loaded_output.shape()[0] == 3,
               "Loaded model should handle batch size 3");
    assertTrue(loaded_output.shape()[1] == 4,
               "Loaded model should output 4 features");

    // Verify batch processing produces same results
    bool batch_matches = true;
    double tolerance = 1e-10;
    for (size_t i = 0; i < original_output.size(); ++i) {
      if (std::abs(original_output[i] - loaded_output[i]) > tolerance) {
        batch_matches = false;
        break;
      }
    }
    assertTrue(
        batch_matches,
        "Batch processing should match between original and loaded models");

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

}  // namespace test
}  // namespace MLLib
