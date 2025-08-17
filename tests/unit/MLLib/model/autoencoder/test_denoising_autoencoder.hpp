#pragma once

#include "../../../../../include/MLLib/loss/mse.hpp"
#include "../../../../../include/MLLib/model/autoencoder/denoising.hpp"
#include "../../../../../include/MLLib/optimizer/adam.hpp"
#include "../../../../common/test_utils.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace MLLib {
namespace test {
namespace autoencoder {

/**
 * @brief Test denoising autoencoder configuration
 */
void test_denoising_config() {
  printf("Testing denoising autoencoder configuration...\n");

  // Test default config
  model::autoencoder::DenoisingConfig default_config;
  assert(default_config.noise_type == model::autoencoder::NoiseType::GAUSSIAN);
  assert(default_config.noise_factor == 0.1);
  assert(default_config.validate_on_clean == true);

  // Test custom config
  model::autoencoder::DenoisingConfig custom_config;
  custom_config.noise_type = model::autoencoder::NoiseType::SALT_PEPPER;
  custom_config.noise_factor = 0.2;
  custom_config.dropout_rate = 0.3;
  custom_config.validate_on_clean = false;

  assert(custom_config.noise_type ==
         model::autoencoder::NoiseType::SALT_PEPPER);
  assert(custom_config.noise_factor == 0.2);
  assert(custom_config.dropout_rate == 0.3);
  assert(custom_config.validate_on_clean == false);

  printf("✓ Denoising autoencoder configuration tests passed\n");
}

/**
 * @brief Test denoising autoencoder construction
 */
void test_denoising_construction() {
  printf("Testing denoising autoencoder construction...\n");

  // Test with explicit parameters
  model::autoencoder::DenoisingAutoencoder
      denoiser(10, 5, {8, 6}, 0.15, model::autoencoder::NoiseType::GAUSSIAN);
  assert(denoiser.get_input_dim() == 10);
  assert(denoiser.get_latent_dim() == 5);
  assert(denoiser.get_type() == model::autoencoder::AutoencoderType::DENOISING);
  assert(denoiser.get_denoising_config().noise_factor == 0.15);

  // Test with configuration
  auto base_config =
      model::autoencoder::AutoencoderConfig::basic(20, 8, {16, 12});
  model::autoencoder::DenoisingConfig denoising_config;
  denoising_config.noise_type = model::autoencoder::NoiseType::DROPOUT;
  denoising_config.noise_factor = 0.2;

  model::autoencoder::DenoisingAutoencoder config_denoiser(base_config,
                                                           denoising_config);
  assert(config_denoiser.get_input_dim() == 20);
  assert(config_denoiser.get_latent_dim() == 8);
  assert(config_denoiser.get_denoising_config().noise_type ==
         model::autoencoder::NoiseType::DROPOUT);

  printf("✓ Denoising autoencoder construction tests passed\n");
}

/**
 * @brief Test different noise types
 */
void test_noise_types() {
  printf("Testing different noise types...\n");

  // Test data
  NDArray clean_data({2, 8});
  for (int i = 0; i < 16; ++i) {
    clean_data.data()[i] = (i % 4) / 3.0;  // Values between 0 and 1
  }

  // Test Gaussian noise
  model::autoencoder::DenoisingAutoencoder
      gaussian_denoiser(8, 4, {}, 0.1, model::autoencoder::NoiseType::GAUSSIAN);
  NDArray gaussian_output = gaussian_denoiser.reconstruct(clean_data);
  assert(gaussian_output.shape()[0] == 2);
  assert(gaussian_output.shape()[1] == 8);

  // Test Salt & Pepper noise
  model::autoencoder::DenoisingAutoencoder
      sp_denoiser(8, 4, {}, 0.1, model::autoencoder::NoiseType::SALT_PEPPER);
  NDArray sp_output = sp_denoiser.reconstruct(clean_data);
  assert(sp_output.shape()[0] == 2);
  assert(sp_output.shape()[1] == 8);

  // Test Dropout noise
  model::autoencoder::DenoisingAutoencoder
      dropout_denoiser(8, 4, {}, 0.1, model::autoencoder::NoiseType::DROPOUT);
  NDArray dropout_output = dropout_denoiser.reconstruct(clean_data);
  assert(dropout_output.shape()[0] == 2);
  assert(dropout_output.shape()[1] == 8);

  // Test Uniform noise
  model::autoencoder::DenoisingAutoencoder
      uniform_denoiser(8, 4, {}, 0.1, model::autoencoder::NoiseType::UNIFORM);
  NDArray uniform_output = uniform_denoiser.reconstruct(clean_data);
  assert(uniform_output.shape()[0] == 2);
  assert(uniform_output.shape()[1] == 8);

  printf("✓ Different noise types tests passed\n");
}

/**
 * @brief Test denoising functionality
 */
void test_denoising() {
  printf("Testing denoising functionality...\n");

  model::autoencoder::DenoisingAutoencoder
      denoiser(6, 3, {}, 0.15, model::autoencoder::NoiseType::GAUSSIAN);

  // Create clean test data
  NDArray clean_data({1, 6});
  clean_data.data()[0] = 1.0;
  clean_data.data()[1] = 0.0;
  clean_data.data()[2] = 1.0;
  clean_data.data()[3] = 0.0;
  clean_data.data()[4] = 1.0;
  clean_data.data()[5] = 0.0;

  // Manually create noisy version (simulating external noise)
  NDArray noisy_data({1, 6});
  noisy_data.data()[0] = 0.9;
  noisy_data.data()[1] = 0.1;
  noisy_data.data()[2] = 0.85;
  noisy_data.data()[3] = 0.15;
  noisy_data.data()[4] = 0.95;
  noisy_data.data()[5] = 0.05;

  // Test denoising
  NDArray denoised = denoiser.denoise(noisy_data);
  assert(denoised.shape()[0] == 1);
  assert(denoised.shape()[1] == 6);

  // Calculate reconstruction error
  double error = denoiser.reconstruction_error(clean_data, "mse");
  assert(error >= 0.0);

  printf("Denoising error: %.4f\n", error);

  printf("✓ Denoising functionality tests passed\n");
}

/**
 * @brief Test denoising metrics evaluation
 */
void test_denoising_metrics() {
  printf("Testing denoising metrics evaluation...\n");

  model::autoencoder::DenoisingAutoencoder
      denoiser(4, 2, {}, 0.1, model::autoencoder::NoiseType::GAUSSIAN);

  // Create test datasets
  std::vector<NDArray> clean_data;
  std::vector<NDArray> noisy_data;

  for (int i = 0; i < 3; ++i) {
    NDArray clean({1, 4});
    NDArray noisy({1, 4});

    for (int j = 0; j < 4; ++j) {
      clean.data()[j] = (j % 2) ? 1.0 : 0.0;
      noisy.data()[j] =
          clean.data()[j] + 0.1 * (i + 1) * ((j % 2) ? -1.0 : 1.0);
    }

    clean_data.push_back(clean);
    noisy_data.push_back(noisy);
  }

  // Evaluate denoising performance
  auto metrics = denoiser.evaluate_denoising(clean_data, noisy_data);

  // Check that metrics exist and are reasonable
  assert(metrics.find("MSE") != metrics.end());
  assert(metrics.find("PSNR") != metrics.end());
  assert(metrics.find("SSIM") != metrics.end());

  assert(metrics["MSE"] >= 0.0);
  assert(metrics["PSNR"] >= 0.0);  // PSNR can be very low but not negative
  assert(metrics["SSIM"] >= -1.0 && metrics["SSIM"] <= 1.0);  // SSIM range

  printf("Denoising metrics - MSE: %.4f, PSNR: %.2f dB, SSIM: %.4f\n",
         metrics["MSE"], metrics["PSNR"], metrics["SSIM"]);

  printf("✓ Denoising metrics evaluation tests passed\n");
}

/**
 * @brief Test denoising training
 */
void test_denoising_training() {
  printf("Testing denoising training...\n");

  model::autoencoder::DenoisingAutoencoder
      denoiser(4, 2, {}, 0.1, model::autoencoder::NoiseType::GAUSSIAN);

  // Create clean training data
  std::vector<NDArray> clean_data;

  NDArray pattern1({1, 4});
  pattern1.data()[0] = 1.0;
  pattern1.data()[1] = 0.0;
  pattern1.data()[2] = 1.0;
  pattern1.data()[3] = 0.0;

  NDArray pattern2({1, 4});
  pattern2.data()[0] = 0.0;
  pattern2.data()[1] = 1.0;
  pattern2.data()[2] = 0.0;
  pattern2.data()[3] = 1.0;

  for (int i = 0; i < 8; ++i) {
    clean_data.push_back(i % 2 == 0 ? pattern1 : pattern2);
  }

  // Create loss and optimizer
  loss::MSELoss loss;
  optimizer::Adam optimizer(0.01);

  // Measure initial performance
  double initial_error = denoiser.reconstruction_error(pattern1, "mse");

  // Train denoising autoencoder (trains on noisy->clean mapping)
  denoiser.train(clean_data, loss, optimizer, 3, 4);

  // Measure final performance
  double final_error = denoiser.reconstruction_error(pattern1, "mse");

  printf("Denoising training - Initial error: %.4f, Final error: %.4f\n",
         initial_error, final_error);

  printf("✓ Denoising training tests passed\n");
}

/**
 * @brief Test denoising configuration updates
 */
void test_denoising_config_updates() {
  printf("Testing denoising configuration updates...\n");

  model::autoencoder::DenoisingAutoencoder denoiser(6, 3);

  // Test initial config
  auto initial_config = denoiser.get_denoising_config();
  assert(initial_config.noise_factor == 0.1);  // Default
  assert(initial_config.noise_type == model::autoencoder::NoiseType::GAUSSIAN);

  // Update config
  model::autoencoder::DenoisingConfig new_config;
  new_config.noise_factor = 0.2;
  new_config.noise_type = model::autoencoder::NoiseType::UNIFORM;
  new_config.dropout_rate = 0.3;
  new_config.validate_on_clean = false;

  denoiser.set_denoising_config(new_config);

  auto updated_config = denoiser.get_denoising_config();
  assert(updated_config.noise_factor == 0.2);
  assert(updated_config.noise_type == model::autoencoder::NoiseType::UNIFORM);
  assert(updated_config.dropout_rate == 0.3);
  assert(updated_config.validate_on_clean == false);

  printf("✓ Denoising configuration update tests passed\n");
}

/**
 * @brief Test factory method for images
 */
void test_image_denoising_factory() {
  printf("Testing image denoising factory method...\n");

  // Create denoising autoencoder for 8x8 grayscale images
  auto image_denoiser =
      model::autoencoder::DenoisingAutoencoder::create_for_images(8, 8, 1, 16,
                                                                  0.15);

  assert(image_denoiser != nullptr);
  assert(image_denoiser->get_input_dim() == 8 * 8 * 1);  // Flattened image
  assert(image_denoiser->get_latent_dim() == 16);
  assert(image_denoiser->get_type() ==
         model::autoencoder::AutoencoderType::DENOISING);

  // Test with color images
  auto color_denoiser =
      model::autoencoder::DenoisingAutoencoder::create_for_images(16, 16, 3, 32,
                                                                  0.1);

  assert(color_denoiser != nullptr);
  assert(color_denoiser->get_input_dim() == 16 * 16 * 3);
  assert(color_denoiser->get_latent_dim() == 32);

  printf("✓ Image denoising factory method tests passed\n");
}

/**
 * @brief Test different noise levels
 */
void test_noise_levels() {
  printf("Testing different noise levels...\n");

  NDArray clean_data({1, 6});
  clean_data.fill(0.5);

  // Test low noise
  model::autoencoder::DenoisingAutoencoder low_noise(6, 3, {}, 0.05);
  NDArray low_noise_output = low_noise.reconstruct(clean_data);
  double low_error = low_noise.reconstruction_error(clean_data, "mse");

  // Test medium noise
  model::autoencoder::DenoisingAutoencoder med_noise(6, 3, {}, 0.2);
  NDArray med_noise_output = med_noise.reconstruct(clean_data);
  double med_error = med_noise.reconstruction_error(clean_data, "mse");

  // Test high noise
  model::autoencoder::DenoisingAutoencoder high_noise(6, 3, {}, 0.5);
  NDArray high_noise_output = high_noise.reconstruct(clean_data);
  double high_error = high_noise.reconstruction_error(clean_data, "mse");

  printf(
      "Noise level effects - Low (0.05): %.4f, Med (0.2): %.4f, High (0.5): %.4f\n",
      low_error, med_error, high_error);

  // All should produce valid outputs
  assert(low_noise_output.shape()[1] == 6);
  assert(med_noise_output.shape()[1] == 6);
  assert(high_noise_output.shape()[1] == 6);

  printf("✓ Different noise level tests passed\n");
}

/**
 * @brief Run all denoising autoencoder tests
 */
void run_denoising_autoencoder_tests() {
  printf("=== Running Denoising Autoencoder Tests ===\n");
  test_denoising_config();
  test_denoising_construction();
  test_noise_types();
  test_denoising();
  test_denoising_metrics();
  test_denoising_training();
  test_denoising_config_updates();
  test_image_denoising_factory();
  test_noise_levels();
  printf("=== Denoising Autoencoder Tests Completed ===\n\n");
}

}  // namespace autoencoder
}  // namespace test
}  // namespace MLLib
