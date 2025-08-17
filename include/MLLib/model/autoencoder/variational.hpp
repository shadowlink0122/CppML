#pragma once

#include "base.hpp"

/**
 * @file variational.hpp
 * @brief Variational Autoencoder (VAE) implementation
 */

namespace MLLib {
namespace model {
namespace autoencoder {

/**
 * @struct VAEConfig
 * @brief Configuration for variational autoencoder
 */
struct VAEConfig {
  double kl_weight = 1.0;              ///< KL divergence weight (beta in beta-VAE)
  double kl_anneal_start = 0.0;        ///< KL annealing start value
  double kl_anneal_rate = 0.0001;      ///< KL annealing rate
  bool use_kl_annealing = false;       ///< Use KL annealing
  std::string prior_type = "gaussian"; ///< Prior distribution type
  bool reparameterize = true;          ///< Use reparameterization trick
};

/**
 * @struct VAEOutput
 * @brief Output from VAE encoding
 */
struct VAEOutput {
  NDArray mean;        ///< Mean of latent distribution
  NDArray log_var;     ///< Log variance of latent distribution  
  NDArray sample;      ///< Sampled latent vector
  double kl_loss;      ///< KL divergence loss
};

/**
 * @class VariationalAutoencoder
 * @brief Variational Autoencoder implementation
 */
class VariationalAutoencoder : public BaseAutoencoder {
public:
  /**
   * @brief Constructor with configuration
   * @param config Base autoencoder configuration
   * @param vae_config VAE-specific configuration
   */
  VariationalAutoencoder(const AutoencoderConfig& config,
                        const VAEConfig& vae_config = {});

  /**
   * @brief Constructor with explicit parameters
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param hidden_dims Hidden layer dimensions
   * @param kl_weight KL divergence weight
   * @param device Computation device
   */
  VariationalAutoencoder(int input_dim, int latent_dim,
                        const std::vector<int>& hidden_dims = {},
                        double kl_weight = 1.0,
                        DeviceType device = DeviceType::CPU);

  /**
   * @brief Get autoencoder type
   * @return VARIATIONAL type
   */
  AutoencoderType get_type() const override { return AutoencoderType::VARIATIONAL; }

  /**
   * @brief Encode input to latent distribution parameters
   * @param input Input data
   * @return VAE encoding output with mean, log_var, and sample
   */
  VAEOutput encode_variational(const NDArray& input);

  /**
   * @brief Sample from latent space
   * @param num_samples Number of samples to generate
   * @return Sampled latent vectors
   */
  std::vector<NDArray> sample_latent(int num_samples = 1);

  /**
   * @brief Generate new data by sampling from latent space
   * @param num_samples Number of samples to generate
   * @return Generated data samples
   */
  std::vector<NDArray> generate(int num_samples = 1);

  /**
   * @brief Interpolate between two data points in latent space
   * @param start_point Starting data point
   * @param end_point Ending data point
   * @param num_steps Number of interpolation steps
   * @return Interpolated data points
   */
  std::vector<NDArray> interpolate(const NDArray& start_point,
                                  const NDArray& end_point,
                                  int num_steps = 10);

  /**
   * @brief Train the VAE
   * @param training_data Training dataset
   * @param loss Base loss function (reconstruction loss)
   * @param optimizer Optimizer
   * @param epochs Number of epochs
   * @param batch_size Batch size
   * @param validation_data Optional validation data
   * @param callback Optional training callback (epoch, recon_loss, kl_loss)
   */
  void train(const std::vector<NDArray>& training_data,
            loss::BaseLoss& loss,
            optimizer::BaseOptimizer& optimizer,
            int epochs = 100,
            int batch_size = 32,
            const std::vector<NDArray>* validation_data = nullptr,
            std::function<void(int, double, double)> callback = nullptr) override;

  /**
   * @brief Calculate VAE loss (reconstruction + KL divergence)
   * @param input Input data
   * @param reconstruction Reconstructed data
   * @param mean Latent mean
   * @param log_var Latent log variance
   * @param recon_loss Base reconstruction loss
   * @return Total VAE loss
   */
  double calculate_vae_loss(const NDArray& input,
                           const NDArray& reconstruction,
                           const NDArray& mean,
                           const NDArray& log_var,
                           loss::BaseLoss& recon_loss);

  /**
   * @brief Get current KL weight (with annealing)
   * @param epoch Current epoch
   * @return Current KL weight
   */
  double get_current_kl_weight(int epoch = 0);

  /**
   * @brief Set VAE configuration
   * @param config New VAE configuration
   */
  void set_vae_config(const VAEConfig& config);

  /**
   * @brief Get current VAE configuration
   * @return Current VAE configuration
   */
  const VAEConfig& get_vae_config() const { return vae_config_; }

  /**
   * @brief Create VAE for image generation
   * @param height Image height
   * @param width Image width
   * @param channels Number of channels
   * @param latent_dim Latent dimension
   * @param kl_weight KL divergence weight
   * @param device Computation device
   * @return VariationalAutoencoder instance for images
   */
  static std::unique_ptr<VariationalAutoencoder> create_for_images(
    int height, int width, int channels = 1,
    int latent_dim = 64,
    double kl_weight = 1.0,
    DeviceType device = DeviceType::CPU);

  /**
   * @brief Create beta-VAE with controllable disentanglement
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param beta Beta parameter for disentanglement
   * @param hidden_dims Hidden layer dimensions
   * @param device Computation device
   * @return VariationalAutoencoder instance (beta-VAE)
   */
  static std::unique_ptr<VariationalAutoencoder> create_beta_vae(
    int input_dim, int latent_dim,
    double beta = 4.0,
    const std::vector<int>& hidden_dims = {},
    DeviceType device = DeviceType::CPU);

protected:
  /**
   * @brief Build encoder network (outputs mean and log_var)
   */
  void build_encoder() override;

  /**
   * @brief Build decoder network
   */
  void build_decoder() override;

private:
  VAEConfig vae_config_;
  std::unique_ptr<Sequential> mean_encoder_;    ///< Encoder for mean
  std::unique_ptr<Sequential> logvar_encoder_;  ///< Encoder for log variance

  /**
   * @brief Reparameterization trick: sample = mean + std * epsilon
   * @param mean Latent mean
   * @param log_var Latent log variance
   * @return Sampled latent vector
   */
  NDArray reparameterize_sample(const NDArray& mean, const NDArray& log_var);

  /**
   * @brief Calculate KL divergence loss
   * @param mean Latent mean
   * @param log_var Latent log variance
   * @return KL divergence loss
   */
  double calculate_kl_loss(const NDArray& mean, const NDArray& log_var);

  /**
   * @brief Sample from standard normal distribution
   * @param shape Shape of the tensor to sample
   * @return Random tensor from N(0,1)
   */
  NDArray sample_standard_normal(const std::vector<int>& shape);
};

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
