#pragma once

#include "../device/device.hpp"
#include "../layer/base.hpp"
#include "../loss/base.hpp"
#include "../optimizer/base.hpp"
#include <functional>
#include <initializer_list>
#include <memory>
#include <vector>

/**
 * @file sequential.hpp
 * @brief Sequential model implementation
 */

namespace MLLib {
namespace model {

/**
 * @class Sequential
 * @brief Sequential neural network model
 */
class Sequential {
public:
  /**
   * @brief Constructor
   */
  Sequential();

  /**
   * @brief Constructor with device type
   * @param device Device type for computation
   */
  Sequential(DeviceType device);

  /**
   * @brief Destructor
   */
  ~Sequential();

  /**
   * @brief Add a layer to the model
   * @param layer Pointer to layer (ownership transferred)
   */
  void add_layer(layer::BaseLayer* layer);

  /**
   * @brief Add a layer to the model (shared_ptr version)
   * @param layer Shared pointer to layer
   */
  void add(std::shared_ptr<layer::BaseLayer> layer);

  /**
   * @brief Set device for computation
   * @param device Device type
   */
  void set_device(DeviceType device);

  /**
   * @brief Forward propagation
   * @param input Input data
   * @return Output predictions
   */
  NDArray predict(const NDArray& input);

  /**
   * @brief Forward propagation for multiple samples
   * @param inputs Vector of input samples
   * @return Vector of predictions
   */
  std::vector<NDArray> predict(const std::vector<NDArray>& inputs);

  /**
   * @brief Predict from vector input (convenience method)
   * @param input Input vector
   * @return Output vector
   */
  std::vector<double> predict(const std::vector<double>& input);

  /**
   * @brief Predict from initializer list (convenience method)
   * @param input Input as initializer list (e.g., {1.0, 2.0, 3.0})
   * @return Output vector
   */
  std::vector<double> predict(std::initializer_list<double> input);

  /**
   * @brief Train the model
   * @param X Training inputs
   * @param Y Training targets
   * @param loss Loss function
   * @param optimizer Optimizer
   * @param callback Optional callback function for epoch end
   * @param epochs Number of training epochs
   */
  void train(const std::vector<std::vector<double>>& X,
             const std::vector<std::vector<double>>& Y, loss::BaseLoss& loss,
             optimizer::BaseOptimizer& optimizer,
             std::function<void(int, double)> callback = nullptr,
             int epochs = 1000);

  /**
   * @brief Set training mode for all layers
   * @param training True for training mode, false for inference
   */
  void set_training(bool training);

  /**
   * @brief Get number of layers
   * @return Number of layers
   */
  size_t num_layers() const { return layers_.size(); }

  /**
   * @brief Get device type
   * @return Current device type
   */
  DeviceType get_device() const { return device_; }

  /**
   * @brief Get layers (const access)
   * @return Const reference to layers vector
   */
  const std::vector<std::shared_ptr<layer::BaseLayer>>& get_layers() const {
    return layers_;
  }

  /**
   * @brief Get layers (non-const access)
   * @return Reference to layers vector
   */
  std::vector<std::shared_ptr<layer::BaseLayer>>& get_layers() {
    return layers_;
  }

private:
  std::vector<std::shared_ptr<layer::BaseLayer>> layers_;
  DeviceType device_;

  /**
   * @brief Convert vector data to NDArray batch
   * @param data Vector of vectors
   * @return NDArray with shape [batch_size, features]
   */
  NDArray vectorsToNDArray(const std::vector<std::vector<double>>& data);

  /**
   * @brief Get all trainable parameters from all layers
   * @return Vector of parameter pointers
   */
  std::vector<NDArray*> get_all_parameters();

  /**
   * @brief Get all gradients from all layers
   * @return Vector of gradient pointers
   */
  std::vector<NDArray*> get_all_gradients();
};

}  // namespace model
}  // namespace MLLib
