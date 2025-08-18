#include "MLLib/layer/activation/relu.hpp"
#include "MLLib/layer/activation/sigmoid.hpp"
#include "MLLib/layer/dense.hpp"
#include "MLLib/model/autoencoder/dense.hpp"
#include <algorithm>
#include <cmath>

namespace MLLib {
namespace model {
namespace autoencoder {

DenseAutoencoder::DenseAutoencoder() : BaseAutoencoder(AutoencoderConfig()) {
  // Default configuration for deserialization
  config_.encoder_dims = {1};  // Will be updated during deserialization
  config_.latent_dim = 1;
  config_.device = DeviceType::CPU;
}

DenseAutoencoder::DenseAutoencoder(const AutoencoderConfig& config)
    : BaseAutoencoder(config) {
  calculate_decoder_dims();
}

DenseAutoencoder::DenseAutoencoder(int input_dim, int latent_dim,
                                   const std::vector<int>& hidden_dims,
                                   DeviceType device)
    : BaseAutoencoder(AutoencoderConfig()) {
  config_.encoder_dims = {input_dim};
  config_.encoder_dims.insert(config_.encoder_dims.end(), hidden_dims.begin(),
                              hidden_dims.end());
  config_.encoder_dims.push_back(latent_dim);

  config_.latent_dim = latent_dim;
  config_.device = device;

  calculate_decoder_dims();
  initialize();
}

std::unique_ptr<DenseAutoencoder>
DenseAutoencoder::create_simple(int input_dim, int latent_dim,
                                double compression_ratio, DeviceType device) {

  // Calculate intermediate dimension
  int intermediate_dim =
      static_cast<int>(std::sqrt(input_dim * latent_dim * compression_ratio));

  std::vector<int> hidden_dims = {intermediate_dim};

  return std::make_unique<DenseAutoencoder>(input_dim, latent_dim, hidden_dims,
                                            device);
}

std::unique_ptr<DenseAutoencoder>
DenseAutoencoder::create_deep(int input_dim, int latent_dim, int num_layers,
                              DeviceType device) {

  std::vector<int> hidden_dims;

  // Create decreasing layer sizes
  for (int i = 1; i <= num_layers; ++i) {
    double ratio = static_cast<double>(i) / (num_layers + 1);
    int dim = static_cast<int>(input_dim * (1.0 - ratio) + latent_dim * ratio);
    hidden_dims.push_back(dim);
  }

  return std::make_unique<DenseAutoencoder>(input_dim, latent_dim, hidden_dims,
                                            device);
}

void DenseAutoencoder::build_encoder() {
  encoder_ = std::make_unique<Sequential>(config_.device);

  for (size_t i = 0; i < config_.encoder_dims.size() - 1; ++i) {
    size_t input_dim = static_cast<size_t>(config_.encoder_dims[i]);
    size_t output_dim = static_cast<size_t>(config_.encoder_dims[i + 1]);

    // Add dense layer
    auto dense_layer = std::make_shared<layer::Dense>(input_dim, output_dim);
    encoder_->add(dense_layer);

    // Add activation function
    if (i < config_.encoder_dims.size() - 2) {
      // Hidden layers use ReLU
      auto activation = std::make_shared<layer::activation::ReLU>();
      encoder_->add(activation);
    }
    // Output layer (latent) uses linear activation (no activation added)
  }
}

void DenseAutoencoder::build_decoder() {
  decoder_ = std::make_unique<Sequential>(config_.device);

  for (size_t i = 0; i < config_.decoder_dims.size() - 1; ++i) {
    size_t input_dim = static_cast<size_t>(config_.decoder_dims[i]);
    size_t output_dim = static_cast<size_t>(config_.decoder_dims[i + 1]);

    // Add dense layer
    auto dense_layer = std::make_shared<layer::Dense>(input_dim, output_dim);
    decoder_->add(dense_layer);

    // Add activation function
    if (i < config_.decoder_dims.size() - 2) {
      // Hidden layers use ReLU
      auto activation = std::make_shared<layer::activation::ReLU>();
      decoder_->add(activation);
    } else {
      // Output layer uses sigmoid for bounded output [0,1]
      auto activation = std::make_shared<layer::activation::Sigmoid>();
      decoder_->add(activation);
    }
  }
}

void DenseAutoencoder::calculate_decoder_dims() {
  config_.decoder_dims = {config_.latent_dim};

  // Reverse the encoder hidden dimensions for symmetric architecture
  for (int i = static_cast<int>(config_.encoder_dims.size()) - 2; i >= 1; --i) {
    config_.decoder_dims.push_back(config_.encoder_dims[i]);
  }

  // Add input dimension as final output
  config_.decoder_dims.push_back(config_.encoder_dims[0]);
}

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
