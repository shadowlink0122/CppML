#pragma once

#include "../../../../include/MLLib/layer/activation/relu.hpp"
#include "../../../../include/MLLib/layer/activation/sigmoid.hpp"
#include "../../../../include/MLLib/layer/dense.hpp"
#include "../../../../include/MLLib/model/model_io.hpp"
#include "../../../../include/MLLib/model/sequential.hpp"
#include "../../../common/test_utils.hpp"

namespace MLLib {
namespace test {

/**
 * @class ModelFormatTest
 * @brief Test ModelFormat enum and string conversions
 */
class ModelFormatTest : public TestCase {
public:
  ModelFormatTest() : TestCase("ModelFormatTest") {}

protected:
  void test() override {
    using namespace MLLib::model;

    // Test string to format conversion
    assertEqual(ModelFormat::BINARY, ModelIO::string_to_format("binary"),
                "binary string should convert to BINARY");
    assertEqual(ModelFormat::JSON, ModelIO::string_to_format("json"),
                "json string should convert to JSON");
    assertEqual(ModelFormat::CONFIG, ModelIO::string_to_format("config"),
                "config string should convert to CONFIG");

    // Test case insensitive conversion
    assertEqual(ModelFormat::BINARY, ModelIO::string_to_format("BINARY"),
                "BINARY string should convert to BINARY");
    assertEqual(ModelFormat::JSON, ModelIO::string_to_format("JSON"),
                "JSON string should convert to JSON");
    assertEqual(ModelFormat::CONFIG, ModelIO::string_to_format("CONFIG"),
                "CONFIG string should convert to CONFIG");

    // Test format to string conversion
    assertEqual(std::string("binary"),
                ModelIO::format_to_string(ModelFormat::BINARY),
                "BINARY should convert to binary");
    assertEqual(std::string("json"),
                ModelIO::format_to_string(ModelFormat::JSON),
                "JSON should convert to json");
    assertEqual(std::string("config"),
                ModelIO::format_to_string(ModelFormat::CONFIG),
                "CONFIG should convert to config");

    // Test invalid string conversion
    assertThrows<std::invalid_argument>(
        [&]() { ModelIO::string_to_format("invalid"); },
        "Invalid format string should throw exception");
  }
};

/**
 * @class ModelSaveLoadTest
 * @brief Test model saving and loading functionality
 */
class ModelSaveLoadTest : public TestCase {
public:
  ModelSaveLoadTest() : TestCase("ModelSaveLoadTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    // Create a simple model for testing
    auto model = std::make_unique<Sequential>();
    model->add(std::make_unique<Dense>(2, 4));
    model->add(std::make_unique<activation::ReLU>());
    model->add(std::make_unique<Dense>(4, 3));
    model->add(std::make_unique<activation::Sigmoid>());

    // Test directory creation and cleanup
    std::string temp_dir = createTempDirectory();

    // Test binary format save/load
    std::string binary_path = temp_dir + "/model.bin";
    assertTrue(ModelIO::save_model(*model, binary_path, ModelFormat::BINARY),
               "Binary save should succeed");
    assertTrue(fileExists(binary_path), "Binary file should exist after save");

    auto loaded_binary = ModelIO::load_model(binary_path, ModelFormat::BINARY);
    assertNotNull(loaded_binary.get(), "Binary load should return valid model");

    // Test JSON format save/load
    std::string json_path = temp_dir + "/model.json";
    assertTrue(ModelIO::save_model(*model, json_path, ModelFormat::JSON),
               "JSON save should succeed");
    assertTrue(fileExists(json_path), "JSON file should exist after save");

    auto loaded_json = ModelIO::load_model(json_path, ModelFormat::JSON);
    assertNotNull(loaded_json.get(), "JSON load should return valid model");

    // Test config format save/load
    std::string config_path = temp_dir + "/model.config";
    assertTrue(ModelIO::save_config(*model, config_path),
               "Config save should succeed");
    assertTrue(fileExists(config_path), "Config file should exist after save");

    auto loaded_config = ModelIO::load_config(config_path);
    assertNotNull(loaded_config.get(), "Config load should return valid model");

    // Cleanup
    removeTempDirectory(temp_dir);
  }
};

/**
 * @class ModelParameterTest
 * @brief Test model parameter saving and loading
 */
class ModelParameterTest : public TestCase {
public:
  ModelParameterTest() : TestCase("ModelParameterTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    // Create model with known parameters
    auto original_model = std::make_unique<Sequential>();
    original_model->add(std::make_unique<Dense>(2, 3));
    original_model->add(std::make_unique<Dense>(3, 1));

    // Initialize with some test values (if model supports parameter access)
    // Note: This test assumes the model has been trained or initialized

    std::string temp_dir = createTempDirectory();
    std::string param_path = temp_dir + "/parameters.bin";

    // Test parameter save
    assertTrue(ModelIO::save_parameters(*original_model, param_path),
               "Parameter save should succeed");
    assertTrue(fileExists(param_path),
               "Parameter file should exist after save");

    // Create new model with same architecture
    auto new_model = std::make_unique<Sequential>();
    new_model->add(std::make_unique<Dense>(2, 3));
    new_model->add(std::make_unique<Dense>(3, 1));

    // Test parameter load
    assertTrue(ModelIO::load_parameters(*new_model, param_path),
               "Parameter load should succeed");

    // Note: Parameter comparison would require access to model internals
    // This test verifies the save/load operations complete without error

    removeTempDirectory(temp_dir);
  }
};

/**
 * @class ModelIOErrorTest
 * @brief Test ModelIO error conditions
 */
class ModelIOErrorTest : public TestCase {
public:
  ModelIOErrorTest() : TestCase("ModelIOErrorTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    // Create a simple model
    auto model = std::make_unique<Sequential>();
    model->add(std::make_unique<Dense>(2, 3));

    // Test save to invalid path
    std::string invalid_path = "/nonexistent/directory/model.bin";
    assertFalse(ModelIO::save_model(*model, invalid_path, ModelFormat::BINARY),
                "Save to invalid path should fail");

    // Test load from nonexistent file
    std::string nonexistent_path = "/nonexistent/file.bin";
    auto loaded = ModelIO::load_model(nonexistent_path, ModelFormat::BINARY);
    assertNull(loaded.get(), "Load from nonexistent file should return null");

    // Test load from corrupted file
    std::string temp_dir = createTempDirectory();
    std::string corrupted_path = temp_dir + "/corrupted.bin";

    // Create a file with invalid content
    std::ofstream corrupted_file(corrupted_path);
    corrupted_file << "This is not a valid model file";
    corrupted_file.close();

    auto corrupted_loaded =
        ModelIO::load_model(corrupted_path, ModelFormat::BINARY);
    assertNull(corrupted_loaded.get(),
               "Load from corrupted file should return null");

    // Test parameter operations with incompatible models
    std::string param_path = temp_dir + "/test_params.bin";

    // Save parameters from one model
    ModelIO::save_parameters(*model, param_path);

    // Try to load into incompatible model
    auto incompatible_model = std::make_unique<Sequential>();
    incompatible_model->add(
        std::make_unique<Dense>(3, 4));  // Different architecture

    assertFalse(ModelIO::load_parameters(*incompatible_model, param_path),
                "Loading parameters into incompatible model should fail");

    removeTempDirectory(temp_dir);
  }
};

/**
 * @class ModelIOFileHandlingTest
 * @brief Test ModelIO file handling capabilities
 */
class ModelIOFileHandlingTest : public TestCase {
public:
  ModelIOFileHandlingTest() : TestCase("ModelIOFileHandlingTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    auto model = std::make_unique<Sequential>();
    model->add(std::make_unique<Dense>(2, 3));

    // Test nested directory creation
    std::string temp_dir = createTempDirectory();
    std::string nested_path = temp_dir + "/deep/nested/directory/model.bin";

    assertTrue(ModelIO::save_model(*model, nested_path, ModelFormat::BINARY),
               "Save to nested directory should succeed");
    assertTrue(fileExists(nested_path),
               "File in nested directory should exist");

    // Test overwriting existing file
    assertTrue(ModelIO::save_model(*model, nested_path, ModelFormat::BINARY),
               "Overwriting existing file should succeed");

    // Test loading from nested path
    auto loaded = ModelIO::load_model(nested_path, ModelFormat::BINARY);
    assertNotNull(loaded.get(), "Load from nested directory should succeed");

    // Test different file extensions
    std::string bin_path = temp_dir + "/model.bin";
    std::string json_path = temp_dir + "/model.json";
    std::string custom_path = temp_dir + "/model.custom";

    assertTrue(ModelIO::save_model(*model, bin_path, ModelFormat::BINARY),
               "Save with .bin extension should succeed");
    assertTrue(ModelIO::save_model(*model, json_path, ModelFormat::JSON),
               "Save with .json extension should succeed");
    assertTrue(ModelIO::save_model(*model, custom_path, ModelFormat::BINARY),
               "Save with custom extension should succeed");

    removeTempDirectory(temp_dir);
  }
};

}  // namespace test
}  // namespace MLLib
