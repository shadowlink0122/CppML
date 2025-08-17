#include "../../../include/MLLib/model/sequential.hpp"
#include "../../../include/MLLib/layer/dense.hpp"
#include "../../../include/MLLib/model/model_io.hpp"
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

namespace MLLib {
namespace model {

Sequential::Sequential()
    : BaseModel(ModelType::SEQUENTIAL), device_(DeviceType::CPU) {}

Sequential::Sequential(DeviceType device)
    : BaseModel(ModelType::SEQUENTIAL), device_(device) {
  Device::setDeviceWithValidation(device, true);
  device_ = Device::getCurrentDevice();  // Update to actual device (in case of
                                         // fallback)
}

Sequential::~Sequential() {
  // Smart pointers will automatically clean up
}

void Sequential::add_layer(layer::BaseLayer* layer) {
  layers_.emplace_back(layer);
}

void Sequential::add(std::shared_ptr<layer::BaseLayer> layer) {
  layers_.push_back(std::move(layer));
}

void Sequential::set_device(DeviceType device) {
  if (Device::setDeviceWithValidation(device, true)) {
    device_ = device;
  } else {
    device_ = Device::getCurrentDevice();  // Update to actual device (fallback)
  }
}

NDArray Sequential::predict(const NDArray& input) {
  if (layers_.empty()) {
    throw std::runtime_error("No layers added to the model");
  }

  NDArray current_output = input;

  // Set all layers to inference mode
  set_training(false);

  // Forward pass through all layers
  for (const auto& layer : layers_) {
    current_output = layer->forward(current_output);
  }

  return current_output;
}

std::vector<NDArray> Sequential::predict(const std::vector<NDArray>& inputs) {
  std::vector<NDArray> predictions;
  predictions.reserve(inputs.size());

  for (const auto& input : inputs) {
    predictions.push_back(predict(input));
  }

  return predictions;
}

std::vector<double> Sequential::predict(const std::vector<double>& input) {
  NDArray input_array(input);
  input_array.reshape({1, input.size()});  // Add batch dimension

  NDArray output = predict(input_array);

  // Return first (and only) sample from batch
  std::vector<double> result;
  size_t output_size = output.shape()[1];
  result.reserve(output_size);

  for (size_t i = 0; i < output_size; ++i) {
    result.push_back(output.at({0, i}));
  }

  return result;
}

std::vector<double> Sequential::predict(std::initializer_list<double> input) {
  std::vector<double> input_vector(input);
  return predict(input_vector);
}

void Sequential::train(const std::vector<std::vector<double>>& X,
                       const std::vector<std::vector<double>>& Y,
                       loss::BaseLoss& loss,
                       optimizer::BaseOptimizer& optimizer,
                       std::function<void(int, double)> callback, int epochs) {
  if (X.size() != Y.size()) {
    throw std::invalid_argument(
        "Number of input samples must match number of targets");
  }

  if (layers_.empty()) {
    throw std::runtime_error("No layers added to the model");
  }

  // Convert data to NDArrays
  NDArray input_batch = vectorsToNDArray(X);
  NDArray target_batch = vectorsToNDArray(Y);

  // Set all layers to training mode
  set_training(true);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Forward pass
    NDArray current_output = input_batch;
    for (const auto& layer : layers_) {
      current_output = layer->forward(current_output);
    }

    // Compute loss
    double current_loss = loss.compute_loss(current_output, target_batch);

    // Backward pass
    NDArray grad = loss.compute_gradient(current_output, target_batch);

    // Backpropagate through all layers in reverse order
    for (int i = layers_.size() - 1; i >= 0; --i) {
      grad = layers_[i]->backward(grad);
    }

    // Update parameters
    std::vector<NDArray*> all_params = get_all_parameters();
    std::vector<NDArray*> all_grads = get_all_gradients();

    if (!all_params.empty()) {
      optimizer.update(all_params, all_grads);
    }

    // Call callback if provided
    if (callback) {
      callback(epoch, current_loss);
    }
  }
}

void Sequential::set_training(bool training) {
  for (const auto& layer : layers_) {
    layer->set_training(training);
  }
}

NDArray
Sequential::vectorsToNDArray(const std::vector<std::vector<double>>& data) {
  if (data.empty()) {
    return NDArray({0, 0});
  }

  size_t batch_size = data.size();
  size_t feature_size = data[0].size();

  // Check that all samples have the same size
  for (const auto& sample : data) {
    if (sample.size() != feature_size) {
      throw std::invalid_argument(
          "All samples must have the same number of features");
    }
  }

  NDArray result({batch_size, feature_size});

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < feature_size; ++j) {
      result.at({i, j}) = data[i][j];
    }
  }

  return result;
}

std::vector<NDArray*> Sequential::get_all_parameters() {
  std::vector<NDArray*> all_params;

  for (const auto& layer : layers_) {
    auto layer_params = layer->get_parameters();
    all_params.insert(all_params.end(), layer_params.begin(),
                      layer_params.end());
  }

  return all_params;
}

std::vector<NDArray*> Sequential::get_all_gradients() {
  std::vector<NDArray*> all_grads;

  for (const auto& layer : layers_) {
    auto layer_params = layer->get_parameters();

    // For Dense layers, we need to get gradients
    if (auto dense_layer = dynamic_cast<layer::Dense*>(layer.get())) {
      all_grads.push_back(
          const_cast<NDArray*>(&dense_layer->get_weight_gradients()));
      if (dense_layer->get_bias().size() > 0) {  // Check if bias is used
        all_grads.push_back(
            const_cast<NDArray*>(&dense_layer->get_bias_gradients()));
      }
    }
    // Activation layers don't have parameters, so no gradients to add
  }

  return all_grads;
}

// ISerializableModel interface implementation
SerializationMetadata Sequential::get_serialization_metadata() const {
  SerializationMetadata metadata;
  metadata.model_type = ModelType::SEQUENTIAL;
  metadata.version = "1.0.0";
  metadata.device = device_;
  return metadata;
}

std::unordered_map<std::string, std::vector<uint8_t>>
Sequential::serialize() const {
  std::unordered_map<std::string, std::vector<uint8_t>> data;

  // Serialize layer count
  std::vector<uint8_t> layer_count_data;
  size_t layer_count = layers_.size();
  uint8_t* count_bytes = reinterpret_cast<uint8_t*>(&layer_count);
  layer_count_data.insert(layer_count_data.end(), count_bytes,
                          count_bytes + sizeof(size_t));
  data.emplace("layer_count", std::move(layer_count_data));

  // Serialize each layer's configuration and parameters
  for (size_t i = 0; i < layers_.size(); ++i) {
    std::string layer_key = "layer_" + std::to_string(i);

    // For now, create placeholder data - would need proper layer serialization
    std::vector<uint8_t> layer_data;

    // Check layer type and serialize accordingly
    if (auto dense_layer =
            dynamic_cast<const layer::Dense*>(layers_[i].get())) {
      // Store layer type identifier
      layer_data.push_back(1);  // Dense layer type = 1

      // Store Dense layer configuration
      size_t input_size = dense_layer->get_input_size();
      size_t output_size = dense_layer->get_output_size();
      bool use_bias = dense_layer->get_use_bias();

      uint8_t* input_bytes = reinterpret_cast<uint8_t*>(&input_size);
      layer_data.insert(layer_data.end(), input_bytes,
                        input_bytes + sizeof(size_t));

      uint8_t* output_bytes = reinterpret_cast<uint8_t*>(&output_size);
      layer_data.insert(layer_data.end(), output_bytes,
                        output_bytes + sizeof(size_t));

      layer_data.push_back(use_bias ? 1 : 0);

      // TODO: Serialize weights and biases
    } else {
      // Other layer types (activation, etc.)
      layer_data.push_back(0);  // Generic layer type = 0
    }

    data.emplace(layer_key, std::move(layer_data));
  }

  return data;
}

std::string Sequential::get_config_string() const {
  return "Sequential model configuration";  // Placeholder
}

bool Sequential::set_config_from_string(const std::string& config_str) {
  (void)config_str;
  return true;  // Placeholder
}

bool Sequential::deserialize(
    const std::unordered_map<std::string, std::vector<uint8_t>>& data) {
  // Clear existing layers
  layers_.clear();

  // Find layer count
  auto count_it = data.find("layer_count");
  if (count_it == data.end()) {
    std::cerr << "Layer count not found in serialized data" << std::endl;
    return false;
  }

  const auto& count_data = count_it->second;
  if (count_data.size() < sizeof(size_t)) {
    std::cerr << "Invalid layer count data size" << std::endl;
    return false;
  }

  size_t layer_count = *reinterpret_cast<const size_t*>(count_data.data());

  // Deserialize each layer
  for (size_t i = 0; i < layer_count; ++i) {
    std::string layer_key = "layer_" + std::to_string(i);
    auto layer_it = data.find(layer_key);

    if (layer_it == data.end()) {
      std::cerr << "Layer " << i << " data not found" << std::endl;
      return false;
    }

    const auto& layer_data = layer_it->second;
    if (layer_data.empty()) {
      std::cerr << "Empty layer data for layer " << i << std::endl;
      return false;
    }

    uint8_t layer_type = layer_data[0];

    if (layer_type == 1) {  // Dense layer
      if (layer_data.size() < 1 + 2 * sizeof(size_t) + 1) {
        std::cerr << "Invalid Dense layer data size" << std::endl;
        return false;
      }

      size_t offset = 1;
      size_t input_size = *reinterpret_cast<const size_t*>(&layer_data[offset]);
      offset += sizeof(size_t);

      size_t output_size =
          *reinterpret_cast<const size_t*>(&layer_data[offset]);
      offset += sizeof(size_t);

      bool use_bias = (layer_data[offset] != 0);

      // Create Dense layer
      auto dense_layer =
          std::make_shared<layer::Dense>(input_size, output_size, use_bias);
      layers_.push_back(dense_layer);

      // TODO: Deserialize weights and biases
    } else {
      // Generic layer - would need to identify specific type
      std::cerr << "Unsupported layer type: " << static_cast<int>(layer_type)
                << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace model
}  // namespace MLLib
