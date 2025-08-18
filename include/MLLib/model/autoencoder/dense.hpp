#pragma once

#include "base.hpp"

/**
 * @file dense.hpp
 * @brief Dense (fully-connected) autoencoder implementation
 */

namespace MLLib {
namespace model {
namespace autoencoder {

/**
 * @class DenseAutoencoder
 * @brief Dense autoencoder with fully connected layers
 */
class DenseAutoencoder : public BaseAutoencoder {
public:
  /**
   * @brief Default constructor (for deserialization)
   */
  DenseAutoencoder();

  /**
   * @brief Constructor with configuration
   * @param config Autoencoder configuration
   */
  explicit DenseAutoencoder(const AutoencoderConfig& config);

  /**
   * @brief Constructor with explicit dimensions
   * @param input_dim Input dimension
   * @param latent_dim Latent space dimension
   * @param hidden_dims Hidden layer dimensions (for encoder)
   * @param device Computation device
   */
  DenseAutoencoder(int input_dim, int latent_dim,
                   const std::vector<int>& hidden_dims = {},
                   DeviceType device = DeviceType::CPU);

  /**
   * @brief Get autoencoder type
   * @return BASIC type
   */
  AutoencoderType get_type() const override { return AutoencoderType::BASIC; }

  /**
   * @brief Create a simple dense autoencoder
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param compression_ratio Compression ratio (input_dim / latent_dim)
   * @param device Computation device
   * @return DenseAutoencoder instance
   */
  static std::unique_ptr<DenseAutoencoder>
  create_simple(int input_dim, int latent_dim, double compression_ratio = 4.0,
                DeviceType device = DeviceType::CPU);

  /**
   * @brief Create a deep dense autoencoder
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param num_layers Number of hidden layers in encoder
   * @param device Computation device
   * @return DenseAutoencoder instance
   */
  static std::unique_ptr<DenseAutoencoder>
  create_deep(int input_dim, int latent_dim, int num_layers = 3,
              DeviceType device = DeviceType::CPU);

private:
  /**
   * @brief Build encoder network for dense autoencoder
   */
  void build_encoder() override;

  /**
   * @brief Build decoder network for dense autoencoder
   */
  void build_decoder() override;

  /**
   * @brief Calculate symmetric decoder dimensions
   */
  void calculate_decoder_dims();
};

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
