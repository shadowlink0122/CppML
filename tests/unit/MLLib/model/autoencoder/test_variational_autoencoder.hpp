#pragma once

#include "../../../../../include/MLLib/loss/mse.hpp"
#include "../../../../../include/MLLib/model/autoencoder/variational.hpp"
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
 * @brief Test VAE configuration
 */
void test_vae_config() {
  printf("Testing VAE configuration...\n");

  // Test default VAE config
  model::autoencoder::VAEConfig default_config;
  assert(default_config.kl_weight == 1.0);
  assert(!default_config.use_kl_annealing);
  assert(default_config.reparameterize == true);

  // Test custom VAE config
  model::autoencoder::VAEConfig custom_config;
  custom_config.kl_weight = 0.5;
  custom_config.use_kl_annealing = true;
  custom_config.kl_anneal_start = 0.0;
  custom_config.kl_anneal_rate = 0.001;

  printf("✅ VAE configuration tests passed\n");
}

/**
 * @brief Test VAE construction
 */
void test_vae_construction() {
  printf("Testing VAE construction...\n");

  // Test with explicit parameters
  model::autoencoder::VariationalAutoencoder vae(10, 5, {8, 6}, 1.0);
  assert(vae.get_input_dim() == 10);
  assert(vae.get_latent_dim() == 5);
  assert(vae.get_type() == model::autoencoder::AutoencoderType::VARIATIONAL);

  // Test with configuration
  auto base_config =
      model::autoencoder::AutoencoderConfig::basic(20, 8, {16, 12});
  model::autoencoder::VAEConfig vae_config;
  vae_config.kl_weight = 0.5;

  model::autoencoder::VariationalAutoencoder config_vae(base_config,
                                                        vae_config);
  assert(config_vae.get_input_dim() == 20);
  assert(config_vae.get_latent_dim() == 8);
  assert(config_vae.get_vae_config().kl_weight == 0.5);

  printf("✅ VAE construction tests passed\n");
}

/**
 * @brief Test VAE forward pass
 */
void test_vae_forward() {
  printf("Testing VAE forward pass...\n");

  model::autoencoder::VariationalAutoencoder vae(8, 4, {6});

  // Test batch processing
  NDArray input({2, 8});
  input.fill(1.0);

  // Test variational encoding
  auto vae_output = vae.encode_variational(input);
  assert(vae_output.mean.shape()[0] == 2);
  assert(vae_output.mean.shape()[1] == 4);
  assert(vae_output.log_var.shape()[0] == 2);
  assert(vae_output.log_var.shape()[1] == 4);
  assert(vae_output.sample.shape()[0] == 2);
  assert(vae_output.sample.shape()[1] == 4);
  assert(vae_output.kl_loss >= 0.0);

  // Test decoding
  NDArray decoded = vae.decode(vae_output.sample);
  assert(decoded.shape()[0] == 2);
  assert(decoded.shape()[1] == 8);

  // Test full reconstruction
  NDArray reconstructed = vae.reconstruct(input);
  assert(reconstructed.shape()[0] == 2);
  assert(reconstructed.shape()[1] == 8);

  printf("✅ VAE forward pass tests passed\n");
}

/**
 * @brief Test VAE sampling and generation
 */
void test_vae_sampling() {
  printf("Testing VAE sampling and generation...\n");

  model::autoencoder::VariationalAutoencoder vae(6, 3);

  // Test latent sampling
  auto latent_samples = vae.sample_latent(5);
  assert(latent_samples.size() == 5);
  for (const auto& sample : latent_samples) {
    assert(sample.shape()[0] == 1);
    assert(sample.shape()[1] == 3);
  }

  // Test data generation
  auto generated_data = vae.generate(3);
  assert(generated_data.size() == 3);
  for (const auto& data : generated_data) {
    assert(data.shape()[0] == 1);
    assert(data.shape()[1] == 6);
  }

  printf("✅ VAE sampling and generation tests passed\n");
}

/**
 * @brief Test VAE interpolation
 */
void test_vae_interpolation() {
  printf("Testing VAE interpolation...\n");

  model::autoencoder::VariationalAutoencoder vae(4, 2);

  // Create two different input points
  NDArray start_point({1, 4});
  start_point.data()[0] = 1.0;
  start_point.data()[1] = 0.0;
  start_point.data()[2] = 1.0;
  start_point.data()[3] = 0.0;

  NDArray end_point({1, 4});
  end_point.data()[0] = 0.0;
  end_point.data()[1] = 1.0;
  end_point.data()[2] = 0.0;
  end_point.data()[3] = 1.0;

  // Test interpolation
  auto interpolated = vae.interpolate(start_point, end_point, 5);
  assert(interpolated.size() == 5);

  for (const auto& point : interpolated) {
    assert(point.shape()[0] == 1);
    assert(point.shape()[1] == 4);
  }

  printf("✅ VAE interpolation tests passed\n");
}

/**
 * @brief Test VAE loss calculation
 */
void test_vae_loss() {
  printf("Testing VAE loss calculation...\n");

  model::autoencoder::VariationalAutoencoder vae(4, 2);

  // Create test data
  NDArray input({1, 4});
  input.fill(1.0);

  NDArray reconstruction({1, 4});
  reconstruction.fill(0.8);  // Some difference for loss

  NDArray mean({1, 2});
  mean.fill(0.1);  // Small mean

  NDArray log_var({1, 2});
  log_var.fill(-1.0);  // Log variance (variance = exp(-1) ≈ 0.37)

  // Test VAE loss calculation
  loss::MSELoss mse_loss;
  double total_loss =
      vae.calculate_vae_loss(input, reconstruction, mean, log_var, mse_loss);
  assert(total_loss > 0.0);

  printf("Total VAE loss: %.4f\n", total_loss);

  printf("✅ VAE loss calculation tests passed\n");
}

/**
 * @brief Test KL weight annealing
 */
void test_kl_annealing() {
  printf("Testing KL weight annealing...\n");

  // Create VAE with KL annealing
  auto base_config = model::autoencoder::AutoencoderConfig::basic(8, 4);
  model::autoencoder::VAEConfig vae_config;
  vae_config.use_kl_annealing = true;
  vae_config.kl_anneal_start = 0.0;
  vae_config.kl_anneal_rate = 0.01;
  vae_config.kl_weight = 1.0;

  model::autoencoder::VariationalAutoencoder vae(base_config, vae_config);

  // Test KL weight at different epochs
  double weight_epoch_0 = vae.get_current_kl_weight(0);
  double weight_epoch_50 = vae.get_current_kl_weight(50);
  double weight_epoch_100 = vae.get_current_kl_weight(100);

  printf("KL weights - Epoch 0: %.4f, Epoch 50: %.4f, Epoch 100: %.4f\n",
         weight_epoch_0, weight_epoch_50, weight_epoch_100);

  // With annealing, weight should increase over time
  assert(weight_epoch_0 <= weight_epoch_50);
  assert(weight_epoch_50 <= weight_epoch_100);

  printf("✅ KL annealing tests passed\n");
}

/**
 * @brief Test VAE factory methods
 */
void test_vae_factory() {
  printf("Testing VAE factory methods...\n");

  // Test beta-VAE creation
  auto beta_vae =
      model::autoencoder::VariationalAutoencoder::create_beta_vae(16, 8, 4.0,
                                                                  {12, 10});
  assert(beta_vae != nullptr);
  assert(beta_vae->get_input_dim() == 16);
  assert(beta_vae->get_latent_dim() == 8);
  assert(beta_vae->get_vae_config().kl_weight == 4.0);  // Beta parameter

  // Test image VAE creation
  auto image_vae =
      model::autoencoder::VariationalAutoencoder::create_for_images(28, 28, 1,
                                                                    64, 1.0);
  assert(image_vae != nullptr);
  assert(image_vae->get_input_dim() == 28 * 28 * 1);  // Flattened image
  assert(image_vae->get_latent_dim() == 64);

  printf("✅ VAE factory method tests passed\n");
}

/**
 * @brief Test VAE training (basic)
 */
void test_vae_training() {
  printf("Testing VAE training...\n");

  // Create simple VAE
  model::autoencoder::VariationalAutoencoder vae(4, 2);

  // Create simple training data
  std::vector<NDArray> training_data;

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
    training_data.push_back(i % 2 == 0 ? pattern1 : pattern2);
  }

  // Create loss and optimizer
  loss::MSELoss loss;
  optimizer::Adam optimizer(0.01);

  // Test training (very few epochs for quick test)
  double initial_error = vae.reconstruction_error(pattern1, "mse");

  // Train with callback to monitor both losses
  vae.train(training_data, loss, optimizer, 3, 2, nullptr,
            [](int epoch, double recon_loss, double kl_loss) {
              printf("Epoch %d - Reconstruction: %.4f, KL: %.4f\n", epoch,
                     recon_loss, kl_loss);
            });

  double final_error = vae.reconstruction_error(pattern1, "mse");

  printf("Training - Initial error: %.4f, Final error: %.4f\n", initial_error,
         final_error);

  printf("✅ VAE training tests passed\n");
}

/**
 * @brief Test VAE configuration updates
 */
void test_vae_config_updates() {
  printf("Testing VAE configuration updates...\n");

  model::autoencoder::VariationalAutoencoder vae(6, 3);

  // Test initial config
  auto initial_config = vae.get_vae_config();
  assert(initial_config.kl_weight == 1.0);

  // Update config
  model::autoencoder::VAEConfig new_config;
  new_config.kl_weight = 2.0;
  new_config.use_kl_annealing = true;
  new_config.kl_anneal_rate = 0.002;

  vae.set_vae_config(new_config);

  auto updated_config = vae.get_vae_config();
  assert(updated_config.kl_weight == 2.0);
  assert(updated_config.use_kl_annealing == true);
  assert(updated_config.kl_anneal_rate == 0.002);

  printf("✅ VAE configuration update tests passed\n");
}

/**
 * @brief Run all VAE tests
 */
void run_variational_autoencoder_tests() {
  printf("=== Running Variational Autoencoder Tests ===\n");
  test_vae_config();
  test_vae_construction();
  test_vae_forward();
  test_vae_sampling();
  test_vae_interpolation();
  test_vae_loss();
  test_kl_annealing();
  test_vae_factory();
  test_vae_training();
  test_vae_config_updates();
  printf("=== Variational Autoencoder Tests Completed ===\n\n");
}

}  // namespace autoencoder
}  // namespace test
}  // namespace MLLib
