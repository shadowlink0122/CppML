#include "MLLib/layer/activation/relu.hpp"
#include "MLLib/layer/activation/sigmoid.hpp"
#include "MLLib/layer/dense.hpp"
#include "MLLib/model/autoencoder/base.hpp"
#include "MLLib/model/model_io.hpp"
#include "MLLib/util/misc/random.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

namespace MLLib {
namespace model {
namespace autoencoder {

// AutoencoderConfig static methods
AutoencoderConfig
AutoencoderConfig::basic(int input_dim, int latent_dim,
                         const std::vector<int>& hidden_dims) {
  AutoencoderConfig config;
  config.encoder_dims = {input_dim};
  config.encoder_dims.insert(config.encoder_dims.end(), hidden_dims.begin(),
                             hidden_dims.end());
  config.encoder_dims.push_back(latent_dim);

  config.decoder_dims = {latent_dim};
  // Reverse hidden dimensions for symmetric decoder
  for (auto it = hidden_dims.rbegin(); it != hidden_dims.rend(); ++it) {
    config.decoder_dims.push_back(*it);
  }
  config.decoder_dims.push_back(input_dim);

  config.latent_dim = latent_dim;
  return config;
}

AutoencoderConfig
AutoencoderConfig::denoising(int input_dim, int latent_dim, double noise_factor,
                             const std::vector<int>& hidden_dims) {
  auto config = basic(input_dim, latent_dim, hidden_dims);
  config.noise_factor = noise_factor;
  return config;
}

// BaseAutoencoder implementation
BaseAutoencoder::BaseAutoencoder()
    : BaseModel(ModelType::AUTOENCODER_DENSE), config_() {
  // Default minimal configuration
  config_.encoder_dims = {1, 1};
  config_.decoder_dims = {1, 1};
  config_.latent_dim = 1;
  config_.device = DeviceType::CPU;
}

BaseAutoencoder::BaseAutoencoder(const AutoencoderConfig& config)
    : BaseModel(ModelType::AUTOENCODER_DENSE), config_(config) {
  initialize();
}

void BaseAutoencoder::initialize() {
  encoder_ = std::make_unique<Sequential>(config_.device);
  decoder_ = std::make_unique<Sequential>(config_.device);

  build_encoder();
  build_decoder();
}

NDArray BaseAutoencoder::encode(const NDArray& input) {
  return encoder_->predict(input);
}

NDArray BaseAutoencoder::decode(const NDArray& latent) {
  return decoder_->predict(latent);
}

NDArray BaseAutoencoder::reconstruct(const NDArray& input) {
  NDArray noisy_input = add_noise(input);
  NDArray latent = encode(noisy_input);
  return decode(latent);
}

void BaseAutoencoder::train(const std::vector<NDArray>& training_data,
                            loss::BaseLoss& loss,
                            optimizer::BaseOptimizer& optimizer, int epochs,
                            int batch_size,
                            const std::vector<NDArray>* validation_data,
                            std::function<void(int, double, double)> callback) {

  // Get all parameters from both encoder and decoder
  auto encoder_params = get_parameters();

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;
    int num_batches = 0;

    // Shuffle training data
    std::vector<size_t> indices(training_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(),
                 std::mt19937{std::random_device{}()});

    // Process batches
    for (size_t i = 0; i < training_data.size(); i += batch_size) {
      size_t batch_end = std::min(i + batch_size, training_data.size());

      // Create batch
      std::vector<NDArray> batch_inputs;
      for (size_t j = i; j < batch_end; ++j) {
        batch_inputs.push_back(training_data[indices[j]]);
      }

      // Forward pass
      double batch_loss = 0.0;
      for (const auto& input : batch_inputs) {
        NDArray noisy_input = add_noise(input);
        NDArray reconstruction = reconstruct(noisy_input);
        batch_loss += loss.compute_loss(reconstruction, input);
      }
      batch_loss /= batch_inputs.size();

      // Backward pass - simplified for demonstration
      // In practice, you'd compute gradients properly

      total_loss += batch_loss;
      num_batches++;
    }

    double avg_loss = total_loss / num_batches;
    double val_loss = 0.0;

    // Validation
    if (validation_data) {
      for (const auto& val_input : *validation_data) {
        NDArray reconstruction = reconstruct(val_input);
        val_loss += loss.compute_loss(reconstruction, val_input);
      }
      val_loss /= validation_data->size();
    }

    // Callback
    if (callback) {
      callback(epoch, avg_loss, val_loss);
    }
  }
}

double BaseAutoencoder::reconstruction_error(const NDArray& input,
                                             const std::string& metric) {
  NDArray reconstruction = reconstruct(input);

  // Calculate error based on metric
  if (metric == "mse") {
    // MSE calculation would go here
    return 0.0;  // Placeholder
  } else if (metric == "mae") {
    // MAE calculation would go here
    return 0.0;  // Placeholder
  } else if (metric == "rmse") {
    // RMSE calculation would go here
    return 0.0;  // Placeholder
  }

  return 0.0;  // Default
}

void BaseAutoencoder::set_training(bool training) {
  encoder_->set_training(training);
  decoder_->set_training(training);
}

// ISerializableModel interface implementation
SerializationMetadata BaseAutoencoder::get_serialization_metadata() const {
  SerializationMetadata metadata;
  metadata.model_type = ModelType::AUTOENCODER_DENSE;
  metadata.version = "1.0.0";
  metadata.device = config_.device;
  return metadata;
}

std::unordered_map<std::string, std::vector<uint8_t>>
BaseAutoencoder::serialize() const {
  auto data = serialize_impl();
  return *data;
}

bool BaseAutoencoder::deserialize(
    const std::unordered_map<std::string, std::vector<uint8_t>>& data) {
  return deserialize_impl(data);
}

std::string BaseAutoencoder::get_config_string() const {
  return "BaseAutoencoder configuration";  // Placeholder
}

bool BaseAutoencoder::set_config_from_string(const std::string& config_str) {
  (void)config_str;
  return true;  // Placeholder
}

std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
BaseAutoencoder::serialize_impl() const {
  std::unordered_map<std::string, std::vector<uint8_t>> data;

  // Serialize configuration
  std::vector<uint8_t> config_data;

  // Store encoder dimensions
  data["config_encoder_size"] = {
      static_cast<uint8_t>(config_.encoder_dims.size())};
  for (size_t i = 0; i < config_.encoder_dims.size(); ++i) {
    std::string key = "config_encoder_dim_" + std::to_string(i);
    std::vector<uint8_t> dim_data;
    const uint8_t* dim_bytes =
        reinterpret_cast<const uint8_t*>(&config_.encoder_dims[i]);
    dim_data.insert(dim_data.end(), dim_bytes, dim_bytes + sizeof(int));
    data[key] = std::move(dim_data);
  }

  // Store decoder dimensions
  data["config_decoder_size"] = {
      static_cast<uint8_t>(config_.decoder_dims.size())};
  for (size_t i = 0; i < config_.decoder_dims.size(); ++i) {
    std::string key = "config_decoder_dim_" + std::to_string(i);
    std::vector<uint8_t> dim_data;
    const uint8_t* dim_bytes =
        reinterpret_cast<const uint8_t*>(&config_.decoder_dims[i]);
    dim_data.insert(dim_data.end(), dim_bytes, dim_bytes + sizeof(int));
    data[key] = std::move(dim_data);
  }

  // Store latent dimension
  std::vector<uint8_t> latent_data;
  const uint8_t* latent_bytes =
      reinterpret_cast<const uint8_t*>(&config_.latent_dim);
  latent_data.insert(latent_data.end(), latent_bytes,
                     latent_bytes + sizeof(int));
  data["config_latent_dim"] = std::move(latent_data);

  // Store noise factor
  std::vector<uint8_t> noise_data;
  const uint8_t* noise_bytes =
      reinterpret_cast<const uint8_t*>(&config_.noise_factor);
  noise_data.insert(noise_data.end(), noise_bytes,
                    noise_bytes + sizeof(double));
  data["config_noise_factor"] = std::move(noise_data);

  // Store sparsity penalty
  std::vector<uint8_t> sparsity_data;
  const uint8_t* sparsity_bytes =
      reinterpret_cast<const uint8_t*>(&config_.sparsity_penalty);
  sparsity_data.insert(sparsity_data.end(), sparsity_bytes,
                       sparsity_bytes + sizeof(double));
  data["config_sparsity_penalty"] = std::move(sparsity_data);

  // Store boolean flags
  data["config_use_batch_norm"] = {
      static_cast<uint8_t>(config_.use_batch_norm ? 1 : 0)};
  data["config_device"] = {static_cast<uint8_t>(config_.device)};

  // Serialize encoder parameters (simplified - would need proper NDArray
  // serialization)
  data["encoder_parameters"] = std::vector<uint8_t>();

  // Serialize decoder parameters
  data["decoder_parameters"] = std::vector<uint8_t>();

  return std::make_unique<
      std::unordered_map<std::string, std::vector<uint8_t>>>(std::move(data));
}

bool BaseAutoencoder::deserialize_impl(
    const std::unordered_map<std::string, std::vector<uint8_t>>& data) {
  // Find encoder dimensions
  auto encoder_size_it = data.find("config_encoder_size");
  if (encoder_size_it == data.end() || encoder_size_it->second.empty()) {
    std::cerr << "Encoder size data not found in serialized data" << std::endl;
    return false;
  }

  size_t encoder_size = encoder_size_it->second[0];
  config_.encoder_dims.clear();

  for (size_t i = 0; i < encoder_size; ++i) {
    std::string key = "config_encoder_dim_" + std::to_string(i);
    auto dim_it = data.find(key);
    if (dim_it == data.end() || dim_it->second.size() < sizeof(int)) {
      std::cerr << "Encoder dimension " << i << " data not found" << std::endl;
      return false;
    }

    int dim = *reinterpret_cast<const int*>(dim_it->second.data());
    config_.encoder_dims.push_back(dim);
  }

  // Find decoder dimensions
  auto decoder_size_it = data.find("config_decoder_size");
  if (decoder_size_it == data.end() || decoder_size_it->second.empty()) {
    std::cerr << "Decoder size data not found in serialized data" << std::endl;
    return false;
  }

  size_t decoder_size = decoder_size_it->second[0];
  config_.decoder_dims.clear();

  for (size_t i = 0; i < decoder_size; ++i) {
    std::string key = "config_decoder_dim_" + std::to_string(i);
    auto dim_it = data.find(key);
    if (dim_it == data.end() || dim_it->second.size() < sizeof(int)) {
      std::cerr << "Decoder dimension " << i << " data not found" << std::endl;
      return false;
    }

    int dim = *reinterpret_cast<const int*>(dim_it->second.data());
    config_.decoder_dims.push_back(dim);
  }

  // Deserialize other config parameters
  auto latent_it = data.find("config_latent_dim");
  if (latent_it != data.end() && latent_it->second.size() >= sizeof(int)) {
    config_.latent_dim =
        *reinterpret_cast<const int*>(latent_it->second.data());
  }

  auto noise_it = data.find("config_noise_factor");
  if (noise_it != data.end() && noise_it->second.size() >= sizeof(double)) {
    config_.noise_factor =
        *reinterpret_cast<const double*>(noise_it->second.data());
  }

  auto sparsity_it = data.find("config_sparsity_penalty");
  if (sparsity_it != data.end() &&
      sparsity_it->second.size() >= sizeof(double)) {
    config_.sparsity_penalty =
        *reinterpret_cast<const double*>(sparsity_it->second.data());
  }

  auto batch_norm_it = data.find("config_use_batch_norm");
  if (batch_norm_it != data.end() && !batch_norm_it->second.empty()) {
    config_.use_batch_norm = (batch_norm_it->second[0] != 0);
  }

  auto device_it = data.find("config_device");
  if (device_it != data.end() && !device_it->second.empty()) {
    config_.device = static_cast<DeviceType>(device_it->second[0]);
  }

  // Reinitialize models with new configuration
  initialize();

  // Deserialize parameters (placeholder)
  // Would deserialize encoder and decoder parameters here

  return true;
}

bool BaseAutoencoder::save(const std::string& filepath) const {
  return GenericModelIO::save_model(*this, filepath, SaveFormat::BINARY);
}

bool BaseAutoencoder::load(const std::string& filepath) {
  auto data = GenericModelIO::load_model_data(filepath, SaveFormat::BINARY);
  if (!data) {
    return false;
  }
  return deserialize(*data);
}

void BaseAutoencoder::save_legacy(const std::string& base_path, bool save_json,
                                  bool save_binary) {
  // Save encoder
  if (save_json) {
    // model::save_model_json(*encoder_, base_path + "_encoder.json");
  }
  if (save_binary) {
    // model::save_model_binary(*encoder_, base_path + "_encoder.bin");
  }

  // Save decoder
  if (save_json) {
    // model::save_model_json(*decoder_, base_path + "_decoder.json");
  }
  if (save_binary) {
    // model::save_model_binary(*decoder_, base_path + "_decoder.bin");
  }
}

bool BaseAutoencoder::load_legacy(const std::string& base_path) {
  // Load encoder and decoder
  // return model::load_model_json(*encoder_, base_path + "_encoder.json") &&
  //        model::load_model_json(*decoder_, base_path + "_decoder.json");
  return true;  // Placeholder
}

void BaseAutoencoder::build_encoder() {
  // Default implementation - override in derived classes
  for (size_t i = 0; i < config_.encoder_dims.size() - 1; ++i) {
    size_t input_dim = static_cast<size_t>(config_.encoder_dims[i]);
    size_t output_dim = static_cast<size_t>(config_.encoder_dims[i + 1]);

    auto dense_layer = std::make_shared<layer::Dense>(input_dim, output_dim);
    encoder_->add(dense_layer);

    // Add activation (ReLU for hidden layers, Linear for output)
    if (i < config_.encoder_dims.size() - 2) {
      auto activation = std::make_shared<layer::activation::ReLU>();
      encoder_->add(activation);
    }
  }
}

void BaseAutoencoder::build_decoder() {
  // Default implementation - override in derived classes
  for (size_t i = 0; i < config_.decoder_dims.size() - 1; ++i) {
    size_t input_dim = static_cast<size_t>(config_.decoder_dims[i]);
    size_t output_dim = static_cast<size_t>(config_.decoder_dims[i + 1]);

    auto dense_layer = std::make_shared<layer::Dense>(input_dim, output_dim);
    decoder_->add(dense_layer);

    // Add activation (ReLU for hidden layers, Sigmoid for output)
    if (i < config_.decoder_dims.size() - 2) {
      auto activation = std::make_shared<layer::activation::ReLU>();
      decoder_->add(activation);
    } else {
      // Output layer - use sigmoid for bounded output
      auto activation = std::make_shared<layer::activation::Sigmoid>();
      decoder_->add(activation);
    }
  }
}

NDArray BaseAutoencoder::add_noise(const NDArray& input) {
  if (config_.noise_factor <= 0.0) {
    return input;
  }

  // Add Gaussian noise - simplified implementation
  NDArray noisy_input = input;  // Copy
  // In practice, you'd add proper noise here
  return noisy_input;
}

std::vector<NDArray*> BaseAutoencoder::get_parameters() {
  std::vector<NDArray*> params;

  // Get encoder parameters
  auto encoder_layers = encoder_->get_layers();
  for (auto& layer : encoder_layers) {
    // In practice, you'd get actual parameters from layers
  }

  // Get decoder parameters
  auto decoder_layers = decoder_->get_layers();
  for (auto& layer : decoder_layers) {
    // In practice, you'd get actual parameters from layers
  }

  return params;
}

std::vector<NDArray*> BaseAutoencoder::get_gradients() {
  std::vector<NDArray*> gradients;

  // Get encoder gradients
  auto encoder_layers = encoder_->get_layers();
  for (auto& layer : encoder_layers) {
    // In practice, you'd get actual gradients from layers
  }

  // Get decoder gradients
  auto decoder_layers = decoder_->get_layers();
  for (auto& layer : decoder_layers) {
    // In practice, you'd get actual gradients from layers
  }

  return gradients;
}

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
