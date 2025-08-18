#pragma once

#include "../device/device.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @file base_model.hpp
 * @brief Base interface for all models to enable generic serialization
 */

namespace MLLib {
namespace model {

/**
 * @enum ModelType
 * @brief Enumeration of supported model types
 */
enum class ModelType {
  SEQUENTIAL,             ///< Sequential neural network
  AUTOENCODER_BASIC,      ///< Basic autoencoder
  AUTOENCODER_DENSE,      ///< Dense autoencoder
  AUTOENCODER_VAE,        ///< Variational autoencoder
  AUTOENCODER_DENOISING,  ///< Denoising autoencoder
  AUTOENCODER_ANOMALY,    ///< Anomaly detection autoencoder
  CUSTOM                  ///< Custom model type
};

/**
 * @struct SerializationMetadata
 * @brief Metadata for model serialization
 */
struct SerializationMetadata {
  ModelType model_type;
  std::string version;
  DeviceType device;
  std::unordered_map<std::string, std::string> custom_properties;

  SerializationMetadata()
      : model_type(ModelType::CUSTOM), version("1.0.0"),
        device(DeviceType::CPU) {}
};

/**
 * @interface ISerializableModel
 * @brief Interface for models that can be serialized/deserialized
 */
class ISerializableModel {
public:
  virtual ~ISerializableModel() = default;

  /**
   * @brief Get model type for serialization
   * @return Model type enum
   */
  virtual ModelType get_model_type() const = 0;

  /**
   * @brief Get serialization metadata
   * @return Metadata structure
   */
  virtual SerializationMetadata get_serialization_metadata() const = 0;

  /**
   * @brief Serialize model to key-value pairs
   * @return Map of serializable data
   */
  virtual std::unordered_map<std::string, std::vector<uint8_t>>
  serialize() const = 0;

  /**
   * @brief Deserialize model from key-value pairs
   * @param data Serialized data map
   * @return true if successful
   */
  virtual bool deserialize(
      const std::unordered_map<std::string, std::vector<uint8_t>>& data) = 0;

  /**
   * @brief Get human-readable model configuration
   * @return JSON-like string representation
   */
  virtual std::string get_config_string() const = 0;

  /**
   * @brief Set model configuration from string
   * @param config_str Configuration string
   * @return true if successful
   */
  virtual bool set_config_from_string(const std::string& config_str) = 0;
};

/**
 * @class BaseModel
 * @brief Base implementation for all models
 */
class BaseModel : public ISerializableModel {
protected:
  ModelType model_type_;
  DeviceType device_ = DeviceType::CPU;

public:
  BaseModel(ModelType type) : model_type_(type) {}
  virtual ~BaseModel() = default;

  // ISerializableModel interface implementation
  ModelType get_model_type() const override { return model_type_; }
  SerializationMetadata get_serialization_metadata() const override = 0;
  std::unordered_map<std::string, std::vector<uint8_t>>
  serialize() const override = 0;
  bool deserialize(const std::unordered_map<std::string, std::vector<uint8_t>>&
                       data) override = 0;
  std::string get_config_string() const override = 0;
  bool set_config_from_string(const std::string& config_str) override = 0;

  /**
   * @brief Set training mode
   * @param training True for training, false for inference
   */
  virtual void set_training(bool training) = 0;

  /**
   * @brief Get device type
   * @return Current device type
   */
  DeviceType get_device_type() const { return device_; }

  /**
   * @brief Set device type
   * @param device New device type
   */
  virtual void set_device_type(DeviceType device) { device_ = device; }
};

/**
 * @brief Convert ModelType enum to string
 * @param type Model type enum
 * @return String representation
 */
std::string model_type_to_string(ModelType type);

/**
 * @brief Convert string to ModelType enum
 * @param type_str String representation
 * @return Model type enum
 */
ModelType string_to_model_type(const std::string& type_str);

}  // namespace model
}  // namespace MLLib
