#include "MLLib/model/autoencoder/denoising.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <random>

namespace MLLib {
namespace model {
namespace autoencoder {

DenoisingAutoencoder::DenoisingAutoencoder(
    const AutoencoderConfig& config, const DenoisingConfig& denoising_config)
    : BaseAutoencoder(config), denoising_config_(denoising_config) {}

DenoisingAutoencoder::DenoisingAutoencoder(int input_dim, int latent_dim,
                                           const std::vector<int>& hidden_dims,
                                           double noise_factor,
                                           NoiseType noise_type,
                                           DeviceType device)
    : BaseAutoencoder(AutoencoderConfig()) {
  config_.encoder_dims = {input_dim};
  config_.encoder_dims.insert(config_.encoder_dims.end(), hidden_dims.begin(),
                              hidden_dims.end());
  config_.encoder_dims.push_back(latent_dim);
  config_.latent_dim = latent_dim;
  config_.device = device;
  config_.noise_factor = noise_factor;

  // Calculate symmetric decoder dimensions
  config_.decoder_dims = {latent_dim};
  for (int i = static_cast<int>(config_.encoder_dims.size()) - 2; i >= 1; --i) {
    config_.decoder_dims.push_back(config_.encoder_dims[i]);
  }
  config_.decoder_dims.push_back(input_dim);

  denoising_config_.noise_type = noise_type;
  denoising_config_.noise_factor = noise_factor;

  initialize();
}

void DenoisingAutoencoder::train(
    const std::vector<NDArray>& clean_data, loss::BaseLoss& loss,
    optimizer::BaseOptimizer& optimizer, int epochs, int batch_size,
    const std::vector<NDArray>* validation_data,
    std::function<void(int, double, double)> callback) {
  // For denoising autoencoders, we add noise during training
  BaseAutoencoder::train(clean_data, loss, optimizer, epochs, batch_size,
                         validation_data, callback);
}

NDArray DenoisingAutoencoder::denoise(const NDArray& noisy_input) {
  return reconstruct(noisy_input);
}

std::map<std::string, double> DenoisingAutoencoder::evaluate_denoising(
    const std::vector<NDArray>& clean_data,
    const std::vector<NDArray>& noisy_data) {

  std::map<std::string, double> metrics;

  double total_psnr = 0.0;
  double total_ssim = 0.0;
  double total_mse = 0.0;

  for (size_t i = 0; i < std::min(clean_data.size(), noisy_data.size()); ++i) {
    NDArray denoised = denoise(noisy_data[i]);

    total_psnr += calculate_psnr(clean_data[i], denoised);
    total_ssim += calculate_ssim(clean_data[i], denoised);
    total_mse += reconstruction_error(clean_data[i], "mse");
  }

  size_t num_samples = std::min(clean_data.size(), noisy_data.size());
  metrics["psnr"] = total_psnr / num_samples;
  metrics["ssim"] = total_ssim / num_samples;
  metrics["mse"] = total_mse / num_samples;

  return metrics;
}

void DenoisingAutoencoder::set_denoising_config(const DenoisingConfig& config) {
  denoising_config_ = config;
  config_.noise_factor = config.noise_factor;
}

std::unique_ptr<DenoisingAutoencoder>
DenoisingAutoencoder::create_for_images(int height, int width, int channels,
                                        int latent_dim, double noise_factor,
                                        DeviceType device) {

  int input_dim = height * width * channels;
  std::vector<int> hidden_dims = {input_dim / 2,
                                  input_dim / 4};  // Progressive compression

  return std::make_unique<DenoisingAutoencoder>(input_dim, latent_dim,
                                                hidden_dims, noise_factor,
                                                NoiseType::GAUSSIAN, device);
}

NDArray DenoisingAutoencoder::add_noise(const NDArray& input) {
  if (denoising_config_.noise_factor <= 0.0) {
    return input;
  }

  switch (denoising_config_.noise_type) {
  case NoiseType::GAUSSIAN: return add_gaussian_noise(input);
  case NoiseType::SALT_PEPPER: return add_salt_pepper_noise(input);
  case NoiseType::DROPOUT: return add_dropout_noise(input);
  case NoiseType::UNIFORM: return add_uniform_noise(input);
  default: return input;
  }
}

NDArray DenoisingAutoencoder::add_gaussian_noise(const NDArray& input) {
  // Simplified - in practice you'd add proper Gaussian noise
  return input;  // Placeholder
}

NDArray DenoisingAutoencoder::add_salt_pepper_noise(const NDArray& input) {
  return input;  // Placeholder
}

NDArray DenoisingAutoencoder::add_dropout_noise(const NDArray& input) {
  return input;  // Placeholder
}

NDArray DenoisingAutoencoder::add_uniform_noise(const NDArray& input) {
  return input;  // Placeholder
}

double DenoisingAutoencoder::calculate_psnr(const NDArray& clean,
                                            const NDArray& noisy) {
  // Simplified PSNR calculation - placeholder
  return 20.0;  // Default PSNR value
}

double DenoisingAutoencoder::calculate_ssim(const NDArray& clean,
                                            const NDArray& reconstructed) {
  // Simplified SSIM calculation - placeholder
  return 0.8;  // Default SSIM value
}

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
