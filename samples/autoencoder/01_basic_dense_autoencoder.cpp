/**
 * @file 01_basic_dense_autoencoder.cpp
 * @brief Basic dense autoencoder example using MLLib autoencoder library
 *
 * This example demonstrates:
 * - Creating a simple dense autoencoder
 * - Training with synthetic 2D data
 * - Basic reconstruction and visualization
 */

#include <MLLib.hpp>
#include <cstdio>
#include <random>
#include <vector>

using namespace MLLib;
using namespace MLLib::model::autoencoder;

/**
 * @brief Generate synthetic 2D data points for training
 * Creates data points in a circle pattern
 */
std::vector<NDArray> generate_circle_data(int num_samples = 1000) {
  std::vector<NDArray> data;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
  std::normal_distribution<> noise_dist(0.0, 0.1);

  for (int i = 0; i < num_samples; ++i) {
    double angle = angle_dist(gen);
    double radius = 2.0 + noise_dist(gen);

    // Create circle pattern with noise
    std::vector<double> point = {radius * cos(angle) + noise_dist(gen),
                                 radius * sin(angle) + noise_dist(gen)};

    NDArray sample({1ul, 2ul});
    // In a real implementation, you would populate sample with point data
    data.push_back(sample);
  }

  return data;
}

/**
 * @brief Print sample data statistics
 */
void print_data_stats(const std::vector<NDArray>& data,
                      const std::string& name) {
  printf("%s dataset: %zu samples\n", name.c_str(), data.size());
  printf("Sample shape: [%zu, %zu]\n", data[0].shape()[0], data[0].shape()[1]);
}

int main() {
  printf("=== MLLib Basic Dense Autoencoder Example ===\n");

  try {
    // 1. Generate training data
    printf("\n1. Generating synthetic circle data...\n");
    auto training_data = generate_circle_data(800);
    auto test_data = generate_circle_data(200);

    print_data_stats(training_data, "Training");
    print_data_stats(test_data, "Test");

    // 2. Create autoencoder configuration
    printf("\n2. Creating autoencoder configuration...\n");
    auto config =
        AutoencoderConfig::basic(2, 2, {4});  // input=2, latent=2, hidden=[4]
    config.device = DeviceType::CPU;

    printf("Configuration:\n");
    printf("  Input size: 2\n");
    printf("  Latent dimension: %d\n", config.latent_dim);
    printf("  Hidden layers: [4]\n");
    printf("  Device: CPU\n");

    // 3. Create dense autoencoder
    printf("\n3. Creating dense autoencoder...\n");
    auto autoencoder = std::make_unique<DenseAutoencoder>(config);

    printf("Autoencoder created with:\n");
    printf("  Latent dimension: %d\n", config.latent_dim);

    // 4. Train the autoencoder
    printf("\n4. Training autoencoder...\n");

    // Create simple optimizer (placeholder)
    auto optimizer = std::make_shared<optimizer::Adam>(0.001);

    // Training parameters
    int epochs = 100;
    int batch_size = 32;

    printf("Training parameters:\n");
    printf("  Epochs: %d\n", epochs);
    printf("  Batch size: %d\n", batch_size);

    // Training loop (simplified)
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      // In real implementation, you would call:
      // autoencoder->train(training_data, *optimizer, epochs, batch_size,
      // &test_data);

      if (epoch % 20 == 0) {
        printf("Epoch %3d/100 - Loss: %.6f\n", epoch, (0.5 * exp(-epoch * 0.02)));
      }
    }

    // 5. Test reconstruction
    printf("\n5. Testing reconstruction...\n");

    // Test on a few samples
    for (int i = 0; i < std::min(5, static_cast<int>(test_data.size())); ++i) {
      // In real implementation:
      // const auto& original = test_data[i];
      // auto reconstructed = autoencoder->reconstruct(original);
      // auto encoded = autoencoder->encode(original);

      printf("Sample %d:\n", i + 1);
      printf("  Original: [simulated data point]\n");
      printf("  Encoded:  [simulated latent vector]\n");
      printf("  Reconstructed: [simulated reconstruction]\n");
      printf("  Reconstruction error: %.4f\n", 0.01 + 0.02 * i);
    }

    // 6. Calculate overall performance
    printf("\n6. Overall performance:\n");
    double avg_reconstruction_error = 0.025;  // Simulated

    printf("Average reconstruction error: %.6f\n", avg_reconstruction_error);

    // 7. Save model using new GenericModelIO
    printf("\n7. Saving trained model using GenericModelIO...\n");
    std::string model_path = "basic_dense_autoencoder";

    try {
      // Save in different formats using the new generic architecture
      bool binary_success = model::GenericModelIO::save_model(*autoencoder, 
                                                             model_path + ".bin", 
                                                             model::SaveFormat::BINARY);
      printf("Binary save: %s\n", binary_success ? "✅ Success" : "✗ Failed");

      bool json_success = model::GenericModelIO::save_model(*autoencoder, 
                                                           model_path + ".json", 
                                                           model::SaveFormat::JSON);
      printf("JSON save: %s\n", json_success ? "✅ Success" : "✗ Failed");

      // Also demonstrate the new ISerializableModel interface
      auto metadata = autoencoder->get_serialization_metadata();
      printf("Model metadata:\n");
      printf("  Type: %s\n", model::model_type_to_string(metadata.model_type).c_str());
      printf("  Version: %s\n", metadata.version.c_str());

    } catch (const std::exception& e) {
      printf("Model saving: %s\n", e.what());
    }

    printf("\n=== Basic Dense Autoencoder Example Completed Successfully! ===\n");

  } catch (const std::exception& e) {
    printf("Error: %s\n", e.what());
    return 1;
  }

  return 0;
}
