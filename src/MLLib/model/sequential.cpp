#include "../../../include/MLLib/model/sequential.hpp"
#include "../../../include/MLLib/layer/dense.hpp"
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

namespace MLLib {
namespace model {

Sequential::Sequential() : device_(DeviceType::CPU) {}

Sequential::Sequential(DeviceType device) : device_(device) {
  Device::setDevice(device);
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
  device_ = device;
  Device::setDevice(device);
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

}  // namespace model
}  // namespace MLLib
