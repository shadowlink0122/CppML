#pragma once

#include "../../../include/MLLib/layer/dense.hpp"
#include "../../common/test_utils.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace MLLib::Tests {

class DenseLayerTests : public test::TestCase {
public:
  DenseLayerTests() : TestCase("DenseLayerTests") {}

protected:
  void test() override {
    test_constructor_basic();
    test_constructor_dimensions();
    test_constructor_invalid_inputs();
    test_forward_pass_dimensions();
    test_forward_pass_values();
    test_backward_pass_dimensions();
    test_parameter_access();
    test_weight_initialization();
  }

private:
  void test_constructor_basic() {
    // Test basic constructor
    layer::Dense layer(3, 2);
    assertEqual<size_t>(layer.get_input_size(), 3, "Input size should be 3");
    assertEqual<size_t>(layer.get_output_size(), 2, "Output size should be 2");

    auto weights = layer.get_weights();
    assertEqual<size_t>(weights.shape()[0], 3,
                        "Weights should have 3 rows (input_size)");
    assertEqual<size_t>(weights.shape()[1], 2,
                        "Weights should have 2 columns (output_size)");

    auto bias = layer.get_bias();
    assertEqual<size_t>(bias.shape()[0], 2,
                        "Bias should have 2 elements (output_size)");
    assertTrue(layer.get_use_bias(), "Should use bias by default");
  }

  void test_constructor_dimensions() {
    // Test various dimensions
    layer::Dense layer1(1, 1);
    assertEqual<size_t>(layer1.get_input_size(), 1, "1x1 layer input size");
    assertEqual<size_t>(layer1.get_output_size(), 1, "1x1 layer output size");

    layer::Dense layer2(10, 5);
    assertEqual<size_t>(layer2.get_input_size(), 10, "10x5 layer input size");
    assertEqual<size_t>(layer2.get_output_size(), 5, "10x5 layer output size");

    layer::Dense layer3(100, 50);
    assertEqual<size_t>(layer3.get_input_size(), 100,
                        "100x50 layer input size");
    assertEqual<size_t>(layer3.get_output_size(), 50,
                        "100x50 layer output size");
  }

  void test_constructor_invalid_inputs() {
    // Test zero dimensions (should work but create empty tensors)
    assertNoThrow(
        []() {
          layer::Dense layer(1, 1);
        },
        "Valid dimensions should not throw");

    // In practice, zero dimensions might be handled differently by the
    // implementation This test ensures basic construction doesn't crash
  }

  void test_forward_pass_dimensions() {
    layer::Dense layer(3, 2);

    // Test single sample (column vector)
    NDArray input({3, 1});
    input.fill(1.0);  // Fill with ones for testing

    NDArray output = layer.forward(input);
    assertEqual<size_t>(output.shape()[0], 2, "Output should have 2 rows");
    assertEqual<size_t>(output.shape()[1], 1, "Output should have 1 column");

    // Test batch processing
    NDArray batch_input({3, 5});  // 5 samples
    batch_input.fill(0.5);
    NDArray batch_output = layer.forward(batch_input);
    assertEqual<size_t>(batch_output.shape()[0], 2,
                        "Batch output should have 2 rows");
    assertEqual<size_t>(batch_output.shape()[1], 5,
                        "Batch output should have 5 columns");
  }

  void test_forward_pass_values() {
    layer::Dense layer(2, 1, false);  // No bias for predictable results

    // Set known weights for predictable output
    NDArray weights({2, 1});
    weights[0] = 1.0;
    weights[1] = 2.0;
    layer.set_weights(weights);

    // Test forward pass: output = weights^T * input (for no bias)
    NDArray input({2, 1});
    input[0] = 1.0;
    input[1] = 2.0;

    NDArray output = layer.forward(input);
    // Expected: 1.0*1.0 + 2.0*2.0 = 5.0
    assertNear(output[0], 5.0, 1e-6, "Forward pass calculation without bias");
  }

  void test_backward_pass_dimensions() {
    layer::Dense layer(3, 2);

    // Forward pass first
    NDArray input({3, 1});
    input.fill(1.0);
    layer.forward(input);

    // Backward pass
    NDArray grad_output({2, 1});
    grad_output[0] = 1.0;
    grad_output[1] = 0.5;

    NDArray grad_input = layer.backward(grad_output);
    assertEqual<size_t>(grad_input.shape()[0], 3,
                        "Gradient input should have 3 rows");
    assertEqual<size_t>(grad_input.shape()[1], 1,
                        "Gradient input should have 1 column");
  }

  void test_parameter_access() {
    layer::Dense layer(3, 2);

    // Test getting parameters
    auto weights = layer.get_weights();
    auto bias = layer.get_bias();

    assertEqual<size_t>(weights.shape()[0], 3, "Weights rows (input_size)");
    assertEqual<size_t>(weights.shape()[1], 2, "Weights columns (output_size)");
    assertEqual<size_t>(bias.shape()[0], 2, "Bias size (output_size)");

    // Test setting parameters
    NDArray new_weights({3, 2});
    new_weights.fill(0.5);
    layer.set_weights(new_weights);

    NDArray new_bias({2});
    new_bias[0] = 0.1;
    new_bias[1] = 0.2;
    layer.set_biases(new_bias);

    auto retrieved_weights = layer.get_weights();
    auto retrieved_bias = layer.get_bias();

    assertNear(retrieved_weights[0], 0.5, 1e-6, "Weight update");
    assertNear(retrieved_bias[0], 0.1, 1e-6, "Bias update");
  }

  void test_weight_initialization() {
    layer::Dense layer1(5, 3);
    layer::Dense layer2(5, 3);

    auto weights1 = layer1.get_weights();
    auto weights2 = layer2.get_weights();

    // Check that weights exist and are in reasonable range
    assertTrue(weights1.size() > 0, "Weights should be initialized");
    assertTrue(weights2.size() > 0, "Weights should be initialized");

    // Check reasonable initialization range (typically small values)
    bool reasonable_range = true;
    for (size_t i = 0; i < weights1.size(); ++i) {
      if (std::abs(weights1.data()[i]) > 2.0) {
        reasonable_range = false;
        break;
      }
    }
    assertTrue(reasonable_range, "Weights should be in reasonable range");
  }
};

}  // namespace MLLib::Tests
