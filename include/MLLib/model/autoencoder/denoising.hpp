#pragma once

#include "base.hpp"
#include <map>

/**
 * @file denoising.hpp
 * @brief Denoising autoencoder implementation
 */

namespace MLLib {
namespace model {
namespace autoencoder {

/**
 * @enum NoiseType
 * @brief Types of noise for denoising autoencoders
 */
enum class NoiseType {
  GAUSSIAN,     ///< Gaussian noise
  SALT_PEPPER,  ///< Salt and pepper noise
  DROPOUT,      ///< Dropout noise
  UNIFORM       ///< Uniform noise
};

/**
 * @struct DenoisingConfig
 * @brief Configuration for denoising autoencoder
 */
struct DenoisingConfig {
  NoiseType noise_type = NoiseType::GAUSSIAN;  ///< Type of noise to add
  double noise_factor = 0.1;                   ///< Noise intensity
  double dropout_rate = 0.2;      ///< Dropout rate for dropout noise
  bool validate_on_clean = true;  ///< Validate on clean data
};

/**
 * @class DenoisingAutoencoder
 * @brief Autoencoder designed for noise removal
 */
class DenoisingAutoencoder : public BaseAutoencoder {
public:
  /**
   * @brief Constructor with configuration
   * @param config Base autoencoder configuration
   * @param denoising_config Denoising-specific configuration
   */
  DenoisingAutoencoder(const AutoencoderConfig& config,
                       const DenoisingConfig& denoising_config = {});

  /**
   * @brief Constructor with explicit parameters
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param hidden_dims Hidden layer dimensions
   * @param noise_factor Noise intensity (0.0-1.0)
   * @param noise_type Type of noise
   * @param device Computation device
   */
  DenoisingAutoencoder(int input_dim, int latent_dim,
                       const std::vector<int>& hidden_dims = {},
                       double noise_factor = 0.1,
                       NoiseType noise_type = NoiseType::GAUSSIAN,
                       DeviceType device = DeviceType::CPU);

  /**
   * @brief Get autoencoder type
   * @return DENOISING type
   */
  AutoencoderType get_type() const override {
    return AutoencoderType::DENOISING;
  }

  /**
   * @brief Train the denoising autoencoder
   * @param clean_data Clean training data
   * @param loss Loss function
   * @param optimizer Optimizer
   * @param epochs Number of epochs
   * @param batch_size Batch size
   * @param validation_data Optional clean validation data
   * @param callback Optional training callback
   */
  void
  train(const std::vector<NDArray>& clean_data, loss::BaseLoss& loss,
        optimizer::BaseOptimizer& optimizer, int epochs = 100,
        int batch_size = 32,
        const std::vector<NDArray>* validation_data = nullptr,
        std::function<void(int, double, double)> callback = nullptr) override;

  /**
   * @brief Denoise input data
   * @param noisy_input Noisy input data
   * @return Denoised output
   */
  NDArray denoise(const NDArray& noisy_input);

  /**
   * @brief Calculate denoising performance metrics
   * @param clean_data Clean reference data
   * @param noisy_data Noisy input data
   * @return Metrics (PSNR, SSIM, MSE)
   */
  std::map<std::string, double>
  evaluate_denoising(const std::vector<NDArray>& clean_data,
                     const std::vector<NDArray>& noisy_data);

  /**
   * @brief Set noise configuration
   * @param config New denoising configuration
   */
  void set_denoising_config(const DenoisingConfig& config);

  /**
   * @brief Get current noise configuration
   * @return Current denoising configuration
   */
  const DenoisingConfig& get_denoising_config() const {
    return denoising_config_;
  }

  /**
   * @brief Create denoising autoencoder for images
   * @param height Image height
   * @param width Image width
   * @param channels Number of channels
   * @param latent_dim Latent dimension
   * @param noise_factor Noise intensity
   * @param device Computation device
   * @return DenoisingAutoencoder instance for images
   */
  static std::unique_ptr<DenoisingAutoencoder>
  create_for_images(int height, int width, int channels = 1,
                    int latent_dim = 64, double noise_factor = 0.1,
                    DeviceType device = DeviceType::CPU);

protected:
  /**
   * @brief Apply noise to input data
   * @param input Clean input
   * @return Noisy input
   */
  NDArray add_noise(const NDArray& input) override;

private:
  DenoisingConfig denoising_config_;

  /**
   * @brief Apply Gaussian noise
   * @param input Input data
   * @return Noisy data
   */
  NDArray add_gaussian_noise(const NDArray& input);

  /**
   * @brief Apply salt and pepper noise
   * @param input Input data
   * @return Noisy data
   */
  NDArray add_salt_pepper_noise(const NDArray& input);

  /**
   * @brief Apply dropout noise
   * @param input Input data
   * @return Noisy data
   */
  NDArray add_dropout_noise(const NDArray& input);

  /**
   * @brief Apply uniform noise
   * @param input Input data
   * @return Noisy data
   */
  NDArray add_uniform_noise(const NDArray& input);

  /**
   * @brief Calculate PSNR (Peak Signal-to-Noise Ratio)
   * @param clean Clean image
   * @param noisy Noisy/reconstructed image
   * @return PSNR value in dB
   */
  double calculate_psnr(const NDArray& clean, const NDArray& noisy);

  /**
   * @brief Calculate SSIM (Structural Similarity Index)
   * @param clean Clean image
   * @param reconstructed Reconstructed image
   * @return SSIM value (0-1)
   */
  double calculate_ssim(const NDArray& clean, const NDArray& reconstructed);
};

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
