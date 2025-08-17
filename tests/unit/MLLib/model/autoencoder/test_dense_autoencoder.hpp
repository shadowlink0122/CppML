#pragma once

#include "../../../../../include/MLLib/loss/mse.hpp"
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
 * @brief Test dense autoencoder construction
 */
void test_dense_autoencoder_construction() {
  printf("Testing dense autoencoder construction...\n");

  // Test with explicit dimensions
  model::autoencoder::DenseAutoencoder autoencoder(10, 5, {8, 6});
  assert(autoencoder.get_input_dim() == 10);
  assert(autoencoder.get_latent_dim() == 5);
  assert(autoencoder.get_type() == model::autoencoder::AutoencoderType::BASIC);

  // Test with configuration
  auto config = model::autoencoder::AutoencoderConfig::basic(20, 8, {16, 12});
  model::autoencoder::DenseAutoencoder config_autoencoder(config);
  assert(config_autoencoder.get_input_dim() == 20);
  assert(config_autoencoder.get_latent_dim() == 8);

  printf("✓ Dense autoencoder construction tests passed\n");
}

/**
 * @brief Test dense autoencoder forward pass
 */
void test_dense_autoencoder_forward() {
  printf("Testing dense autoencoder forward pass...\n");

  model::autoencoder::DenseAutoencoder autoencoder(8, 4, {6});

  // Test batch processing
  NDArray input({3, 8});
  input.fill(1.0);

  // Test encoding
  NDArray encoded = autoencoder.encode(input);
  assert(encoded.shape()[0] == 3);
  assert(encoded.shape()[1] == 4);

  // Test decoding
  NDArray decoded = autoencoder.decode(encoded);
  assert(decoded.shape()[0] == 3);
  assert(decoded.shape()[1] == 8);

  // Test full reconstruction
  NDArray reconstructed = autoencoder.reconstruct(input);
  assert(reconstructed.shape()[0] == 3);
  assert(reconstructed.shape()[1] == 8);

  printf("✓ Dense autoencoder forward pass tests passed\n");
}

/**
 * @brief Test dense autoencoder factory methods
 */
void test_dense_autoencoder_factory() {
  printf("Testing dense autoencoder factory methods...\n");

  // Test simple autoencoder creation
  auto simple_ae =
      model::autoencoder::DenseAutoencoder::create_simple(100, 25, 4.0);
  assert(simple_ae != nullptr);
  assert(simple_ae->get_input_dim() == 100);
  assert(simple_ae->get_latent_dim() == 25);

  // Test deep autoencoder creation
  auto deep_ae = model::autoencoder::DenseAutoencoder::create_deep(64, 16, 3);
  assert(deep_ae != nullptr);
  assert(deep_ae->get_input_dim() == 64);
  assert(deep_ae->get_latent_dim() == 16);

  printf("✓ Dense autoencoder factory method tests passed\n");
}

/**
 * @brief Test dense autoencoder training
 */
void test_dense_autoencoder_training() {
  printf("Testing dense autoencoder training...\n");

  // Create autoencoder
  model::autoencoder::DenseAutoencoder autoencoder(4, 2);

  // Create training data (simple patterns)
  std::vector<NDArray> training_data;

  // Pattern 1: [1, 0, 1, 0]
  NDArray pattern1({1, 4});
  pattern1.data()[0] = 1.0;
  pattern1.data()[1] = 0.0;
  pattern1.data()[2] = 1.0;
  pattern1.data()[3] = 0.0;

  // Pattern 2: [0, 1, 0, 1]
  NDArray pattern2({1, 4});
  pattern2.data()[0] = 0.0;
  pattern2.data()[1] = 1.0;
  pattern2.data()[2] = 0.0;
  pattern2.data()[3] = 1.0;

  for (int i = 0; i < 10; ++i) {
    training_data.push_back(i % 2 == 0 ? pattern1 : pattern2);
  }

  // Create loss and optimizer
  loss::MSELoss loss;
  optimizer::Adam optimizer(0.01);

  // Measure initial reconstruction error
  double initial_error = autoencoder.reconstruction_error(pattern1, "mse");

  // Train for a few epochs
  autoencoder.train(training_data, loss, optimizer, 5, 2);

  // Measure final reconstruction error
  double final_error = autoencoder.reconstruction_error(pattern1, "mse");

  printf("Training - Initial error: %.4f, Final error: %.4f\n", initial_error,
         final_error);

  // Test that autoencoder can still perform forward pass after training
  NDArray test_output = autoencoder.reconstruct(pattern1);
  assert(test_output.shape()[0] == 1);
  assert(test_output.shape()[1] == 4);

  printf("✓ Dense autoencoder training tests passed\n");
}

/**
 * @brief Test dense autoencoder compression
 */
void test_dense_autoencoder_compression() {
  printf("Testing dense autoencoder compression...\n");

  // Create autoencoder with significant compression
  model::autoencoder::DenseAutoencoder autoencoder(16, 4, {12, 8});

  // Create input data
  NDArray input({1, 16});
  for (int i = 0; i < 16; ++i) {
    input.data()[i] =
        static_cast<double>(i % 4) / 3.0;  // Pattern with repetition
  }

  // Test compression/decompression
  NDArray encoded = autoencoder.encode(input);
  assert(encoded.shape()[1] == 4);  // Compressed to 1/4 size

  NDArray decoded = autoencoder.decode(encoded);
  assert(decoded.shape()[1] == 16);  // Restored to original size

  // Calculate compression ratio
  double compression_ratio =
      static_cast<double>(input.shape()[1]) / encoded.shape()[1];
  printf("Compression ratio: %.1fx (from %zu to %zu dimensions)\n",
         compression_ratio, input.shape()[1], encoded.shape()[1]);

  assert(compression_ratio == 4.0);  // Expected 4:1 compression

  printf("✓ Dense autoencoder compression tests passed\n");
}

/**
 * @brief Test dense autoencoder different architectures
 */
void test_dense_autoencoder_architectures() {
  printf("Testing dense autoencoder different architectures...\n");

  // Test minimal autoencoder (no hidden layers)
  model::autoencoder::DenseAutoencoder minimal(6, 3);
  NDArray minimal_input({1, 6});
  minimal_input.fill(0.5);
  NDArray minimal_output = minimal.reconstruct(minimal_input);
  assert(minimal_output.shape()[1] == 6);

  // Test deep autoencoder
  model::autoencoder::DenseAutoencoder deep(12, 3, {10, 8, 6, 4});
  NDArray deep_input({1, 12});
  deep_input.fill(0.5);
  NDArray deep_output = deep.reconstruct(deep_input);
  assert(deep_output.shape()[1] == 12);

  // Test wide autoencoder
  model::autoencoder::DenseAutoencoder wide(8, 6, {32, 16});
  NDArray wide_input({1, 8});
  wide_input.fill(0.5);
  NDArray wide_output = wide.reconstruct(wide_input);
  assert(wide_output.shape()[1] == 8);

  printf("✓ Dense autoencoder architecture tests passed\n");
}

/**
 * @brief Test dense autoencoder error metrics
 */
void test_dense_autoencoder_metrics() {
  printf("Testing dense autoencoder error metrics...\n");

  model::autoencoder::DenseAutoencoder autoencoder(4, 2);

  // Create test input
  NDArray input({1, 4});
  input.data()[0] = 1.0;
  input.data()[1] = 0.0;
  input.data()[2] = 1.0;
  input.data()[3] = 0.0;

  // Test different error metrics
  double mse_error = autoencoder.reconstruction_error(input, "mse");
  double mae_error = autoencoder.reconstruction_error(input, "mae");
  double rmse_error = autoencoder.reconstruction_error(input, "rmse");

  assert(mse_error >= 0.0);
  assert(mae_error >= 0.0);
  assert(rmse_error >= 0.0);

  printf("Error metrics - MSE: %.4f, MAE: %.4f, RMSE: %.4f\n", mse_error,
         mae_error, rmse_error);

  printf("✓ Dense autoencoder metrics tests passed\n");
}

/**
 * @brief Run all dense autoencoder tests
 */
void run_dense_autoencoder_tests() {
  printf("=== Running Dense Autoencoder Tests ===\n");
  test_dense_autoencoder_construction();
  test_dense_autoencoder_forward();
  test_dense_autoencoder_factory();
  test_dense_autoencoder_training();
  test_dense_autoencoder_compression();
  test_dense_autoencoder_architectures();
  test_dense_autoencoder_metrics();
  printf("=== Dense Autoencoder Tests Completed ===\n\n");
}

}  // namespace autoencoder
}  // namespace test
}  // namespace MLLib
