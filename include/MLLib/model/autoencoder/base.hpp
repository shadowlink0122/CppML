#pragma once

#include "../../device/device.hpp"
#include "../../layer/base.hpp"
#include "../../loss/base.hpp"
#include "../../optimizer/base.hpp"
#include "../base_model.hpp"
#include "../model_io.hpp"
#include "../sequential.hpp"
#include <memory>
#include <vector>

/**
 * @file base.hpp
 * @brief Base autoencoder implementation
 */

namespace MLLib {
namespace model {
namespace autoencoder {

/**
 * @enum AutoencoderType
 * @brief Types of autoencoders
 */
enum class AutoencoderType {
  BASIC,         ///< Basic autoencoder
  DENOISING,     ///< Denoising autoencoder
  VARIATIONAL,   ///< Variational autoencoder
  SPARSE,        ///< Sparse autoencoder
  CONVOLUTIONAL  ///< Convolutional autoencoder
};

/**
 * @struct AutoencoderConfig
 * @brief Configuration for autoencoder models
 */
struct AutoencoderConfig {
  std::vector<int> encoder_dims;  ///< Encoder layer dimensions
  std::vector<int> decoder_dims;  ///< Decoder layer dimensions
  int latent_dim = 1;             ///< Latent space dimension
  double noise_factor = 0.0;  ///< Noise factor for denoising (0.0 = no noise)
  double sparsity_penalty = 0.0;        ///< Sparsity penalty coefficient
  bool use_batch_norm = false;          ///< Use batch normalization
  DeviceType device = DeviceType::CPU;  ///< Computation device

  /// Default constructor
  AutoencoderConfig() = default;

  /**
   * @brief Create a basic autoencoder config
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param hidden_dims Hidden layer dimensions (symmetric)
   * @return AutoencoderConfig
   */
  static AutoencoderConfig basic(int input_dim, int latent_dim,
                                 const std::vector<int>& hidden_dims = {});

  /**
   * @brief Create a denoising autoencoder config
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param noise_factor Noise level (0.0-1.0)
   * @param hidden_dims Hidden layer dimensions
   * @return AutoencoderConfig
   */
  static AutoencoderConfig denoising(int input_dim, int latent_dim,
                                     double noise_factor = 0.1,
                                     const std::vector<int>& hidden_dims = {});
};

/**
 * @class BaseAutoencoder
 * @brief Base class for all autoencoder implementations
 */
class BaseAutoencoder : public BaseModel {
public:
  /**
   * @brief Default constructor (for deserialization)
   */
  BaseAutoencoder();

  /**
   * @brief Constructor
   * @param config Autoencoder configuration
   */
  explicit BaseAutoencoder(const AutoencoderConfig& config);

  /**
   * @brief Virtual destructor
   */
  virtual ~BaseAutoencoder() = default;

  /**
   * @brief Encode input to latent representation
   * @param input Input data
   * @return Latent representation
   */
  virtual NDArray encode(const NDArray& input);

  /**
   * @brief Decode latent representation to output
   * @param latent Latent representation
   * @return Reconstructed output
   */
  virtual NDArray decode(const NDArray& latent);

  /**
   * @brief Reconstruct input (encode + decode)
   * @param input Input data
   * @return Reconstructed output
   */
  virtual NDArray reconstruct(const NDArray& input);

  /**
   * @brief Train the autoencoder
   * @param training_data Training dataset
   * @param loss Loss function
   * @param optimizer Optimizer
   * @param epochs Number of epochs
   * @param batch_size Batch size
   * @param validation_data Optional validation data
   * @param callback Optional training callback
   */
  virtual void
  train(const std::vector<NDArray>& training_data, loss::BaseLoss& loss,
        optimizer::BaseOptimizer& optimizer, int epochs = 100,
        int batch_size = 32,
        const std::vector<NDArray>* validation_data = nullptr,
        std::function<void(int, double, double)> callback = nullptr);

  /**
   * @brief Calculate reconstruction error
   * @param input Input data
   * @param metric Error metric ("mse", "mae", "rmse")
   * @return Reconstruction error
   */
  virtual double reconstruction_error(const NDArray& input,
                                      const std::string& metric = "mse");

  /**
   * @brief Get latent dimension
   * @return Latent space dimension
   */
  int get_latent_dim() const { return config_.latent_dim; }

  /**
   * @brief Get input dimension
   * @return Input dimension
   */
  int get_input_dim() const { return config_.encoder_dims.front(); }

  /**
   * @brief Get autoencoder type
   * @return Autoencoder type
   */
  virtual AutoencoderType get_type() const = 0;

  /**
   * @brief Set training mode
   * @param training True for training, false for inference
   */
  virtual void set_training(bool training) override;

  /**
   * @brief Get encoder model
   * @return Const reference to encoder
   */
  const Sequential& get_encoder() const { return *encoder_; }

  /**
   * @brief Get decoder model
   * @return Const reference to decoder
   */
  const Sequential& get_decoder() const { return *decoder_; }

  // ISerializableModel interface implementation
  SerializationMetadata get_serialization_metadata() const override;
  std::unordered_map<std::string, std::vector<uint8_t>>
  serialize() const override;
  bool deserialize(const std::unordered_map<std::string, std::vector<uint8_t>>&
                       data) override;
  std::string get_config_string() const override;
  bool set_config_from_string(const std::string& config_str) override;

  /**
   * @brief Save model using generic model I/O
   * @param filepath Path to save file
   * @param format Save format
   * @return True if successful
   */
  bool save(const std::string& filepath) const;

  /**
   * @brief Load model using generic model I/O
   * @param filepath Path to load file
   * @param format Load format
   * @return True if successful
   */
  bool load(const std::string& filepath);

  /**
   * @brief Save model to files (legacy method)
   * @param base_path Base path for saving (without extension)
   * @param save_json Save JSON metadata
   * @param save_binary Save binary weights
   */
  virtual void save_legacy(const std::string& base_path, bool save_json = true,
                           bool save_binary = true);

  /**
   * @brief Load model from files (legacy method)
   * @param base_path Base path for loading (without extension)
   * @return True if successful
   */
  virtual bool load_legacy(const std::string& base_path);

protected:
  AutoencoderConfig config_;             ///< Model configuration
  std::unique_ptr<Sequential> encoder_;  ///< Encoder network
  std::unique_ptr<Sequential> decoder_;  ///< Decoder network

  /**
   * @brief Build encoder network
   */
  virtual void build_encoder();

  /**
   * @brief Build decoder network
   */
  virtual void build_decoder();

  /**
   * @brief Apply noise to input (for denoising autoencoders)
   * @param input Clean input
   * @return Noisy input
   */
  virtual NDArray add_noise(const NDArray& input);

  /**
   * @brief Get all trainable parameters
   * @return Vector of parameter pointers
   */
  std::vector<NDArray*> get_parameters();

  /**
   * @brief Get all gradients
   * @return Vector of gradient pointers
   */
  std::vector<NDArray*> get_gradients();

protected:
  /**
   * @brief Initialize the autoencoder models
   */
  void initialize();

  /**
   * @brief Serialize autoencoder-specific data
   * @return Serialized data map
   */
  virtual std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
  serialize_impl() const;

  /**
   * @brief Deserialize autoencoder-specific data
   * @param data Serialized data map
   * @return True if successful
   */
  virtual bool deserialize_impl(
      const std::unordered_map<std::string, std::vector<uint8_t>>& data);
};

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
