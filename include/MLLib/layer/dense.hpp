#pragma once

#include "base.hpp"

/**
 * @file dense.hpp
 * @brief Dense (fully connected) layer implementation
 */

namespace MLLib {
namespace layer {

/**
 * @class Dense
 * @brief Dense (fully connected) layer
 */
class Dense : public BaseLayer {
public:
  /**
   * @brief Constructor
   * @param input_size Number of input features
   * @param output_size Number of output features
   * @param use_bias Whether to use bias term
   */
  Dense(size_t input_size, size_t output_size, bool use_bias = true);

  /**
   * @brief Destructor
   */
  virtual ~Dense() = default;

  /**
   * @brief Forward propagation
   * @param input Input data [batch_size, input_size]
   * @return Output data [batch_size, output_size]
   */
  NDArray forward(const NDArray& input) override;

  /**
   * @brief Backward propagation
   * @param grad_output Gradient from the next layer [batch_size, output_size]
   * @return Gradient with respect to input [batch_size, input_size]
   */
  NDArray backward(const NDArray& grad_output) override;

  /**
   * @brief Get trainable parameters
   * @return Vector of parameter pointers (weights and bias)
   */
  std::vector<NDArray*> get_parameters() override;

  /**
   * @brief Get weights
   * @return Reference to weights matrix
   */
  const NDArray& get_weights() const { return weights_; }

  /**
   * @brief Get bias
   * @return Reference to bias vector
   */
  const NDArray& get_bias() const { return bias_; }

  /**
   * @brief Get weight gradients
   * @return Reference to weight gradients
   */
  const NDArray& get_weight_gradients() const { return weight_gradients_; }

  /**
   * @brief Get bias gradients
   * @return Reference to bias gradients
   */
  const NDArray& get_bias_gradients() const { return bias_gradients_; }

private:
  size_t input_size_;
  size_t output_size_;
  bool use_bias_;

  NDArray weights_;           ///< Weight matrix [input_size, output_size]
  NDArray bias_;              ///< Bias vector [output_size]
  NDArray weight_gradients_;  ///< Gradients for weights
  NDArray bias_gradients_;    ///< Gradients for bias

  NDArray last_input_;  ///< Cache input for backward pass

  /**
   * @brief Initialize weights and bias
   */
  void initialize_parameters();
};

}  // namespace layer
}  // namespace MLLib
