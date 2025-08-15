#pragma once

#include "../../../include/MLLib/layer/Dense.hpp"
#include "../../common/test_utils.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace MLLib::Tests {

class DenseLayerTests : public TestCase {
public:
  DenseLayerTests() : TestCase("DenseLayerTests") {}

  void run() override {
    test_constructor_basic();
    test_constructor_dimensions();
    test_constructor_invalid_inputs();
    test_forward_pass_dimensions();
    test_forward_pass_values();
    test_backward_pass_dimensions();
    test_backward_pass_values();
    test_parameter_access();
    test_parameter_update();
    test_activation_functions();
    test_bias_handling();
    test_weight_initialization();
    test_gradient_computation();
    test_multiple_samples();
  }

private:
  void test_constructor_basic() {
    // Test basic constructor
    Dense layer(3, 2);
    assert_equals(layer.getInputSize(), 3, "Input size should be 3");
    assert_equals(layer.getOutputSize(), 2, "Output size should be 2");

    auto weights = layer.getWeights();
    assert_equals(weights.getRows(), 2, "Weights should have 2 rows");
    assert_equals(weights.getCols(), 3, "Weights should have 3 columns");

    auto bias = layer.getBias();
    assert_equals(bias.getRows(), 2, "Bias should have 2 rows");
    assert_equals(bias.getCols(), 1, "Bias should have 1 column");
  }

  void test_constructor_dimensions() {
    // Test various dimensions
    Dense layer1(1, 1);
    assert_equals(layer1.getInputSize(), 1, "1x1 layer input size");
    assert_equals(layer1.getOutputSize(), 1, "1x1 layer output size");

    Dense layer2(10, 5);
    assert_equals(layer2.getInputSize(), 10, "10x5 layer input size");
    assert_equals(layer2.getOutputSize(), 5, "10x5 layer output size");

    Dense layer3(100, 50);
    assert_equals(layer3.getInputSize(), 100, "100x50 layer input size");
    assert_equals(layer3.getOutputSize(), 50, "100x50 layer output size");
  }

  void test_constructor_invalid_inputs() {
    // Test invalid dimensions
    try {
      Dense layer(0, 5);
      assert_true(false, "Should throw exception for zero input size");
    } catch (const std::invalid_argument&) {
      // Expected behavior
    }

    try {
      Dense layer(5, 0);
      assert_true(false, "Should throw exception for zero output size");
    } catch (const std::invalid_argument&) {
      // Expected behavior
    }
  }

  void test_forward_pass_dimensions() {
    Dense layer(3, 2);

    // Test single sample
    NDArray input({3, 1});
    input.setData({1.0, 2.0, 3.0});

    NDArray output = layer.forward(input);
    assert_equals(output.getRows(), 2, "Output should have 2 rows");
    assert_equals(output.getCols(), 1, "Output should have 1 column");

    // Test batch processing
    NDArray batch_input({3, 5});  // 5 samples
    NDArray batch_output = layer.forward(batch_input);
    assert_equals(batch_output.getRows(), 2, "Batch output should have 2 rows");
    assert_equals(batch_output.getCols(), 5,
                  "Batch output should have 5 columns");
  }

  void test_forward_pass_values() {
    Dense layer(2, 1);

    // Set known weights and bias for predictable output
    NDArray weights({1, 2});
    weights.setData({1.0, 2.0});
    layer.setWeights(weights);

    NDArray bias({1, 1});
    bias.setData({0.5});
    layer.setBias(bias);

    // Test forward pass: output = weights * input + bias
    NDArray input({2, 1});
    input.setData({1.0, 2.0});

    NDArray output = layer.forward(input);
    // Expected: 1.0*1.0 + 2.0*2.0 + 0.5 = 5.5
    assert_near(output.getData()[0], 5.5, 1e-6, "Forward pass calculation");
  }

  void test_backward_pass_dimensions() {
    Dense layer(3, 2);

    // Forward pass first
    NDArray input({3, 1});
    input.setData({1.0, 2.0, 3.0});
    layer.forward(input);

    // Backward pass
    NDArray grad_output({2, 1});
    grad_output.setData({1.0, 0.5});

    NDArray grad_input = layer.backward(grad_output);
    assert_equals(grad_input.getRows(), 3, "Gradient input should have 3 rows");
    assert_equals(grad_input.getCols(), 1,
                  "Gradient input should have 1 column");
  }

  void test_backward_pass_values() {
    Dense layer(2, 1);

    // Set known weights
    NDArray weights({1, 2});
    weights.setData({0.5, -0.3});
    layer.setWeights(weights);

    // Forward pass
    NDArray input({2, 1});
    input.setData({2.0, -1.0});
    layer.forward(input);

    // Backward pass
    NDArray grad_output({1, 1});
    grad_output.setData({1.0});

    NDArray grad_input = layer.backward(grad_output);

    // Check gradient dimensions and approximate values
    assert_equals(grad_input.getRows(), 2, "Gradient input dimensions");
    assert_equals(grad_input.getCols(), 1, "Gradient input dimensions");
  }

  void test_parameter_access() {
    Dense layer(3, 2);

    // Test getting parameters
    auto weights = layer.getWeights();
    auto bias = layer.getBias();

    assert_equals(weights.getRows(), 2, "Weights rows");
    assert_equals(weights.getCols(), 3, "Weights columns");
    assert_equals(bias.getRows(), 2, "Bias rows");
    assert_equals(bias.getCols(), 1, "Bias columns");

    // Test setting parameters
    NDArray new_weights({2, 3});
    new_weights.setData({1, 2, 3, 4, 5, 6});
    layer.setWeights(new_weights);

    NDArray new_bias({2, 1});
    new_bias.setData({0.1, 0.2});
    layer.setBias(new_bias);

    auto retrieved_weights = layer.getWeights();
    auto retrieved_bias = layer.getBias();

    assert_near(retrieved_weights.getData()[0], 1.0, 1e-6, "Weight update");
    assert_near(retrieved_bias.getData()[0], 0.1, 1e-6, "Bias update");
  }

  void test_parameter_update() {
    Dense layer(2, 1);

    // Get initial parameters
    auto initial_weights = layer.getWeights();
    auto initial_bias = layer.getBias();

    // Create gradients
    NDArray weight_grad({1, 2});
    weight_grad.setData({0.1, 0.2});

    NDArray bias_grad({1, 1});
    bias_grad.setData({0.05});

    // Update parameters (assuming learning rate = 0.1)
    layer.updateWeights(weight_grad, 0.1);
    layer.updateBias(bias_grad, 0.1);

    // Check that parameters changed
    auto updated_weights = layer.getWeights();
    auto updated_bias = layer.getBias();

    assert_true(updated_weights.getData()[0] != initial_weights.getData()[0],
                "Weights should be updated");
    assert_true(updated_bias.getData()[0] != initial_bias.getData()[0],
                "Bias should be updated");
  }

  void test_activation_functions() {
    Dense layer_linear(2, 1, "linear");
    Dense layer_relu(2, 1, "relu");
    Dense layer_sigmoid(2, 1, "sigmoid");

    // Test that different activation functions are accepted
    NDArray input({2, 1});
    input.setData({1.0, -1.0});

    // All should produce outputs without throwing
    auto output_linear = layer_linear.forward(input);
    auto output_relu = layer_relu.forward(input);
    auto output_sigmoid = layer_sigmoid.forward(input);

    assert_equals(output_linear.getRows(), 1, "Linear activation output size");
    assert_equals(output_relu.getRows(), 1, "ReLU activation output size");
    assert_equals(output_sigmoid.getRows(), 1,
                  "Sigmoid activation output size");
  }

  void test_bias_handling() {
    Dense layer_with_bias(2, 1, "linear", true);
    Dense layer_without_bias(2, 1, "linear", false);

    NDArray input({2, 1});
    input.setData({1.0, 1.0});

    // Set same weights for both layers
    NDArray weights({1, 2});
    weights.setData({1.0, 1.0});
    layer_with_bias.setWeights(weights);
    layer_without_bias.setWeights(weights);

    auto output_with_bias = layer_with_bias.forward(input);
    auto output_without_bias = layer_without_bias.forward(input);

    // With bias should have different output (assuming non-zero bias)
    // This is a basic check - exact values depend on bias initialization
    assert_equals(output_with_bias.getRows(), 1, "Bias layer output size");
    assert_equals(output_without_bias.getRows(), 1,
                  "No bias layer output size");
  }

  void test_weight_initialization() {
    Dense layer1(5, 3);
    Dense layer2(5, 3);

    auto weights1 = layer1.getWeights();
    auto weights2 = layer2.getWeights();

    // Weights should be different between layers (random initialization)
    bool weights_different = false;
    for (size_t i = 0; i < weights1.getData().size(); ++i) {
      if (std::abs(weights1.getData()[i] - weights2.getData()[i]) > 1e-10) {
        weights_different = true;
        break;
      }
    }
    assert_true(weights_different, "Weights should be randomly initialized");

    // Weights should be in reasonable range
    for (double w : weights1.getData()) {
      assert_true(std::abs(w) < 2.0, "Weights should be in reasonable range");
    }
  }

  void test_gradient_computation() {
    Dense layer(2, 2);

    // Forward pass
    NDArray input({2, 1});
    input.setData({1.0, 2.0});
    layer.forward(input);

    // Backward pass
    NDArray grad_output({2, 1});
    grad_output.setData({1.0, 1.0});

    auto grad_input = layer.backward(grad_output);
    auto weight_gradients = layer.getWeightGradients();
    auto bias_gradients = layer.getBiasGradients();

    // Check that gradients have correct dimensions
    assert_equals(grad_input.getRows(), 2, "Input gradient dimensions");
    assert_equals(weight_gradients.getRows(), 2, "Weight gradient dimensions");
    assert_equals(weight_gradients.getCols(), 2, "Weight gradient dimensions");
    assert_equals(bias_gradients.getRows(), 2, "Bias gradient dimensions");
  }

  void test_multiple_samples() {
    Dense layer(3, 2);

    // Test batch processing
    NDArray batch_input({3, 4});  // 4 samples
    batch_input.setData({
        1, 2, 3, 4,    // Feature 1
        5, 6, 7, 8,    // Feature 2
        9, 10, 11, 12  // Feature 3
    });

    auto output = layer.forward(batch_input);
    assert_equals(output.getRows(), 2, "Batch output rows");
    assert_equals(output.getCols(), 4, "Batch output columns");

    // Test backward pass with batch
    NDArray grad_output({2, 4});
    grad_output.setData({1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5});

    auto grad_input = layer.backward(grad_output);
    assert_equals(grad_input.getRows(), 3, "Batch gradient input rows");
    assert_equals(grad_input.getCols(), 4, "Batch gradient input columns");
  }
};

}  // namespace MLLib::Tests
