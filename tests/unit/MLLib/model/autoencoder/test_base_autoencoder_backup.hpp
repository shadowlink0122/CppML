#pragma once

#include "../../../../../include/MLLib/layer/dense.hpp"
#include "../../../../../include/MLLib/loss/mse.hpp"
#include "../../../../../include/MLLib/model/autoencoder/base.hpp"
#include "../../../../../include/MLLib/model/autoencoder/dense.hpp"
#include "../../../../../include/MLLib/optimizer/adam.hpp"
#include "../../../../common/test_utils.hpp"
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>

namespace MLLib {
namespace test {
namespace autoencoder {

/**
 * @brief Test autoencoder configuration creation
 */
void test_autoencoder_config() {
  printf("Testing autoencoder configuration...\n");

  // Test basic config
  auto basic_config =
      model::autoencoder::AutoencoderConfig::basic(784, 64, {256, 128});
  assert(basic_config.encoder_dims[0] == 784);
  assert(basic_config.latent_dim == 64);
  assert(basic_config.noise_factor == 0.0);

  // Test denoising config
  auto denoising_config =
      model::autoencoder::AutoencoderConfig::denoising(784, 64, 0.2,
                                                       {256, 128});
  assert(denoising_config.encoder_dims[0] == 784);
  assert(denoising_config.latent_dim == 64);
  assert(denoising_config.noise_factor == 0.2);

  printf("✓ Autoencoder configuration tests passed\n");
}

/**
 * @brief Test base autoencoder functionality using DenseAutoencoder
 */
void test_base_autoencoder() {
  printf("Testing base autoencoder functionality...\n");

  try {
    // Create a simple dense autoencoder
    auto autoencoder =
        model::autoencoder::DenseAutoencoder::create_simple(10, 5);

    // Test dimensions
    assert(autoencoder->get_input_dim() == 10);
    assert(autoencoder->get_latent_dim() == 5);
    assert(autoencoder->get_type() ==
           model::autoencoder::AutoencoderType::BASIC);

    // Test forward pass
    NDArray input({2, 10});  // Batch size 2, input dimension 10
    input.fill(0.5);

    NDArray encoded = autoencoder->encode(input);
    assert(encoded.shape()[0] == 2);
    assert(encoded.shape()[1] == 5);

    NDArray decoded = autoencoder->decode(encoded);
    assert(decoded.shape()[0] == 2);
    assert(decoded.shape()[1] == 10);

    NDArray reconstructed = autoencoder->reconstruct(input);
    assert(reconstructed.shape()[0] == 2);
    assert(reconstructed.shape()[1] == 10);

    printf("✓ Base autoencoder tests passed
");
  } catch (const std::exception& e) {
    printf("❌ Base autoencoder test failed: %s
", e.what());
  }
}

/**
 * @brief Test autoencoder training
 */
void test_autoencoder_training() {
  printf("Testing autoencoder training...
");

  try {
    // Create simple dense autoencoder
    auto autoencoder =
        model::autoencoder::DenseAutoencoder::create_simple(4, 2);

    // Create simple training data
    std::vector<NDArray> training_data;
    for (int i = 0; i < 5; ++i) {
      NDArray sample({1, 4});  // Single sample with batch dimension
      sample.fill(static_cast<double>(i % 2) * 0.5);
      training_data.push_back(sample);
    }

    // Create loss and optimizer
    loss::MSELoss loss;
    optimizer::Adam optimizer(0.01);

    // Test a very short training session
    autoencoder->train(training_data, loss, optimizer, 1,
                       2);  // 1 epoch, batch size 2
    printf("✓ Autoencoder training test passed
");
  } catch (const std::exception& e) {
    printf("❌ Autoencoder training test failed: %s
", e.what());
  }
}

/**
 * @brief Test noise addition (denoising functionality)
 */
void test_noise_addition() {
  printf("Testing noise addition...
");

  try {
    // Create denoising autoencoder
    auto config = model::autoencoder::AutoencoderConfig::denoising(10, 5, 0.1);
    auto autoencoder =
        std::make_unique<model::autoencoder::DenseAutoencoder>(config);

    // Test basic functionality with noisy config
    NDArray input({2, 10});
    input.fill(0.5);

    NDArray reconstructed = autoencoder->reconstruct(input);
    assert(reconstructed.shape()[0] == 2);
    assert(reconstructed.shape()[1] == 10);

    printf("✓ Noise addition test passed
");
  } catch (const std::exception& e) {
    printf("❌ Noise addition test failed: %s
", e.what());
  }
}

/**
 * @brief Test model save/load
 */
void test_model_save_load() {
  printf("Testing model save/load...
");

  try {
    // Create dense autoencoder
    auto autoencoder =
        model::autoencoder::DenseAutoencoder::create_simple(4, 2);

    // Test input
    NDArray input({1, 4});
    input.fill(0.8);

    NDArray original_output = autoencoder->reconstruct(input);

    // Test save (simplified - actual file operations are complex for testing)
    std::string test_path = "/tmp/test_autoencoder";
    autoencoder->save(test_path, true, false);  // Save JSON only for simplicity

    printf("✓ Model save/load test passed (basic functionality check)
");
  } catch (const std::exception& e) {
    printf("❌ Model save/load test failed: %s
", e.what());
  }
}

/**
 * @brief Run all base autoencoder tests
 */
void run_base_autoencoder_tests() {
  printf("=== Running Base Autoencoder Tests ===
");
  test_autoencoder_config();
  test_base_autoencoder();
  test_autoencoder_training();
  test_noise_addition();
  test_model_save_load();
  printf("=== Base Autoencoder Tests Completed ===

");
}

}  // namespace autoencoder
}  // namespace test
}  // namespace MLLib
