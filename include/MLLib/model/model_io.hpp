#pragma once

#include "sequential.hpp"
#include <fstream>
#include <memory>
#include <string>

/**
 * @file model_io.hpp
 * @brief Model serialization and deserialization functionality
 */

namespace MLLib {
namespace model {

/**
 * @enum ModelFormat
 * @brief Model file format options
 */
enum class ModelFormat {
  BINARY,  ///< Binary format (compact, fast)
  JSON,    ///< JSON format (human readable, larger)
  CONFIG   ///< Configuration only (no parameters)
};

/**
 * @struct LayerInfo
 * @brief Layer configuration information for serialization
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
 * @brief Model configuration for serialization
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
 * @brief Handles model saving and loading operations
 */
class ModelIO {
public:
  /**
   * @brief Save model to file
   * @param model Model to save
   * @param filepath File path to save to
   * @param format Save format (binary, json, or config)
   * @return true if successful, false otherwise
   */
  static bool save_model(const Sequential& model, const std::string& filepath,
                         ModelFormat format = ModelFormat::BINARY);

  /**
   * @brief Load model from file
   * @param filepath File path to load from
   * @param format Load format (binary, json, or config)
   * @return Loaded model (nullptr if failed)
   */
  static std::unique_ptr<Sequential>
  load_model(const std::string& filepath,
             ModelFormat format = ModelFormat::BINARY);

  /**
   * @brief Save model configuration only (no parameters)
   * @param model Model to extract config from
   * @param filepath File path to save to
   * @return true if successful, false otherwise
   */
  static bool save_config(const Sequential& model, const std::string& filepath);

  /**
   * @brief Load model configuration and create empty model
   * @param filepath File path to load config from
   * @return Model with architecture but no trained parameters
   */
  static std::unique_ptr<Sequential> load_config(const std::string& filepath);

  /**
   * @brief Save only model parameters
   * @param model Model to save parameters from
   * @param filepath File path to save to
   * @return true if successful, false otherwise
   */
  static bool save_parameters(const Sequential& model,
                              const std::string& filepath);

  /**
   * @brief Load parameters into existing model
   * @param model Model to load parameters into
   * @param filepath File path to load parameters from
   * @return true if successful, false otherwise
   */
  static bool load_parameters(Sequential& model, const std::string& filepath);

  /**
   * @brief Convert string format to ModelFormat enum
   * @param format_str Format string ("binary", "json", "config")
   * @return ModelFormat enum value
   */
  static ModelFormat string_to_format(const std::string& format_str);

  /**
   * @brief Convert ModelFormat enum to string
   * @param format ModelFormat enum value
   * @return Format string
   */
  static std::string format_to_string(ModelFormat format);

private:
  /**
   * @brief Extract model configuration from Sequential model
   * @param model Model to extract from
   * @return Model configuration
   */
  static ModelConfig extract_config(const Sequential& model);

  /**
   * @brief Create model from configuration
   * @param config Model configuration
   * @return Created model
   */
  static std::unique_ptr<Sequential>
  create_from_config(const ModelConfig& config);

  /**
   * @brief Save in binary format
   */
  static bool save_binary(const Sequential& model, const std::string& filepath);

  /**
   * @brief Load from binary format
   */
  static std::unique_ptr<Sequential> load_binary(const std::string& filepath);

  /**
   * @brief Save in JSON format
   */
  static bool save_json(const Sequential& model, const std::string& filepath);

  /**
   * @brief Load from JSON format
   */
  static std::unique_ptr<Sequential> load_json(const std::string& filepath);

  /**
   * @brief Write binary data
   */
  static void write_binary_data(std::ofstream& file, const void* data,
                                size_t size);

  /**
   * @brief Read binary data
   */
  static bool read_binary_data(std::ifstream& file, void* data, size_t size);

  /**
   * @brief Write NDArray to binary file
   */
  static void write_ndarray(std::ofstream& file, const NDArray& array);

  /**
   * @brief Read NDArray from binary file
   */
  static NDArray read_ndarray(std::ifstream& file);

  /**
   * @brief Create directory recursively (like mkdir -p)
   * @param path Directory path to create
   * @return true if successful or already exists, false otherwise
   */
  static bool create_directories(const std::string& path);

  /**
   * @brief Get directory path from file path
   * @param filepath Full file path
   * @return Directory path (empty if no directory)
   */
  static std::string get_directory_path(const std::string& filepath);

  /**
   * @brief Check if directory exists
   * @param path Directory path to check
   * @return true if directory exists, false otherwise
   */
  static bool directory_exists(const std::string& path);

  /**
   * @brief Ensure directory exists for file path (create if necessary)
   * @param filepath Full file path
   * @return true if directory exists or was created successfully
   */
  static bool ensure_directory_exists(const std::string& filepath);
};

}  // namespace model
}  // namespace MLLib
