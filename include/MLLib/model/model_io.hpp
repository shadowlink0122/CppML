#pragma once

#include "base_model.hpp"
#include "sequential.hpp"
#include <fstream>
#include <memory>
#include <string>

/**
 * @file model_io.hpp
 * @brief Generic model serialization and deserialization functionality
 */

namespace MLLib {
namespace model {

/**
 * @enum SaveFormat
 * @brief Model file format options
 */
enum class SaveFormat {
  BINARY,  ///< Binary format (compact, fast)
  JSON,    ///< JSON format (human readable, larger)
  CONFIG   ///< Configuration only (no parameters)
};

/**
 * @class GenericModelIO
 * @brief Handles generic model saving and loading operations
 */
class GenericModelIO {
public:
  /**
   * @brief Save a model using the generic interface
   * @param model Model implementing ISerializableModel
   * @param filepath Output file path
   * @param format Save format
   * @return True if successful
   */
  static bool save_model(const ISerializableModel& model,
                         const std::string& filepath, SaveFormat format);

  /**
   * @brief Load model of specific type (template function)
   * @tparam ModelClass Specific model class to load (must inherit from
   * ISerializableModel)
   * @param filepath Input file path
   * @param format Load format
   * @return Loaded model or nullptr on failure
   */
  template <typename ModelClass>
  static std::unique_ptr<ModelClass> load_model(const std::string& filepath,
                                                SaveFormat format) {
    static_assert(std::is_base_of_v<ISerializableModel, ModelClass>,
                  "ModelClass must inherit from ISerializableModel");

    // Suppress unused parameter warnings
    (void)filepath;
    (void)format;

    // For now, return nullptr as placeholder
    // Real implementation would deserialize based on ModelClass type
    return nullptr;
  }

  /**
   * @brief Load model data from file
   * @param filepath Input file path
   * @param format Load format
   * @return Loaded data or nullptr on failure
   */
  static std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
  load_model_data(const std::string& filepath, SaveFormat format);

private:
  static bool create_directories(const std::string& path);

  /**
   * @brief Get filepath with appropriate extension for the format
   * @param base_filepath Base filepath without extension
   * @param format Save format to determine extension
   * @return Filepath with appropriate extension
   */
  static std::string
  get_filepath_with_extension(const std::string& base_filepath,
                              SaveFormat format);

  /**
   * @brief Save model configuration only (no parameters)
   * @param model Model to extract config from
   * @param filepath File path to save to
   * @return true if successful, false otherwise
   */
  static bool save_config(const ISerializableModel& model,
                          const std::string& filepath);

  /**
   * @brief Load model metadata from file
   * @param filepath File path to load metadata from
   * @return Model metadata (type, version, etc.)
   */
  static std::unique_ptr<SerializationMetadata>
  load_metadata(const std::string& filepath);

  static bool save_binary(const ISerializableModel& model,
                          const std::string& filepath);
  static bool save_json(const ISerializableModel& model,
                        const std::string& filepath);
  static std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
  load_binary(const std::string& filepath);
  static std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
  load_json(const std::string& filepath);

  // Utility functions made public for ModelIO compatibility
  static bool ensure_directory_exists(const std::string& filepath);
};

// Legacy support for existing Sequential models
/**
 * @struct LayerInfo
 * @brief Layer configuration information for serialization (legacy)
 */
struct LayerInfo {
  std::string type;        ///< Layer type (dense, relu, sigmoid, etc.)
  size_t input_size = 0;   ///< Input size (for Dense layers)
  size_t output_size = 0;  ///< Output size (for Dense layers)
  bool use_bias = true;    ///< Whether to use bias (for Dense layers)

  LayerInfo() = default;
  LayerInfo(const std::string& t) : type(t) {}
  LayerInfo(const std::string& t, size_t in, size_t out, bool bias = true)
      : type(t), input_size(in), output_size(out), use_bias(bias) {}
};

/**
 * @struct ModelConfig
 * @brief Model configuration for serialization (legacy)
 */
struct ModelConfig {
  std::string model_type = "Sequential";
  std::string version = "1.0.0";
  DeviceType device = DeviceType::CPU;
  std::vector<LayerInfo> layers;

  ModelConfig() = default;
};

/**
 * @class ModelIO
 * @brief Handles Sequential model saving and loading operations (legacy
 * support)
 */
class ModelIO {
public:
  /**
   * @brief Save Sequential model to file (legacy)
   * @param model Model to save
   * @param filepath File path to save to
   * @param format Save format
   * @return true if successful, false otherwise
   */
  static bool save_model(const Sequential& model, const std::string& filepath,
                         SaveFormat format = SaveFormat::BINARY);

  /**
   * @brief Load Sequential model from file (legacy)
   * @param filepath File path to load from
   * @param format Load format
   * @return Loaded model (nullptr if failed)
   */
  static std::unique_ptr<Sequential>
  load_model(const std::string& filepath,
             SaveFormat format = SaveFormat::BINARY);

  /**
   * @brief Save model configuration only (legacy)
   * @param model Model to extract config from
   * @param filepath File path to save to
   * @return true if successful, false otherwise
   */
  static bool save_config(const Sequential& model, const std::string& filepath);

  /**
   * @brief Load model configuration and create empty model (legacy)
   * @param filepath File path to load config from
   * @return Model with architecture but no trained parameters
   */
  static std::unique_ptr<Sequential> load_config(const std::string& filepath);

  // Utility methods
  static SaveFormat string_to_format(const std::string& format_str);
  static std::string format_to_string(SaveFormat format);

private:
  static ModelConfig extract_config(const Sequential& model);
  static std::unique_ptr<Sequential>
  create_from_config(const ModelConfig& config);

  static bool save_binary(const Sequential& model, const std::string& filepath);
  static std::unique_ptr<Sequential> load_binary(const std::string& filepath);
  static bool save_json(const Sequential& model, const std::string& filepath);
  static std::unique_ptr<Sequential> load_json(const std::string& filepath);

  // Legacy helper functions
  static void write_binary_data(std::ofstream& file, const void* data,
                                size_t size);
  static bool read_binary_data(std::ifstream& file, void* data, size_t size);
  static void write_ndarray(std::ofstream& file, const NDArray& array);
  static NDArray read_ndarray(std::ifstream& file);
  static bool create_directories(const std::string& path);
  static std::string get_directory_path(const std::string& filepath);
  static bool directory_exists(const std::string& path);
  static bool ensure_directory_exists(const std::string& filepath);
  static std::string
  get_filepath_with_extension(const std::string& base_filepath,
                              SaveFormat format);
};

}  // namespace model
}  // namespace MLLib
