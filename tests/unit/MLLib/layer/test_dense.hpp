#pragma once

#include "../../../../include/MLLib/layer/dense.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"

namespace MLLib {
namespace test {

/**
 * @class DenseConstructorTest
 * @brief Test Dense layer constructors
 */
class DenseConstructorTest : public TestCase {
public:
  DenseConstructorTest() : TestCase("DenseConstructorTest") {}

protected:
  void test() override {
    using namespace MLLib::layer;

    // Start with very minimal test
    try {
      Dense layer1(3, 4);
      assertTrue(true, "Dense layer created successfully");
    } catch (...) {
      assertTrue(false, "Exception during Dense layer creation");
    }
  }
};

/**
 * @class DenseForwardTest
 * @brief Test Dense layer forward pass
 */
class DenseForwardTest : public TestCase {
public:
  DenseForwardTest() : TestCase("DenseForwardTest") {}

protected:
  void test() override {
    using namespace MLLib::layer;

    Dense layer(3, 2);

    // Test single sample forward pass - Dense expects 2D input (batch_size,
    // features)
    NDArray input({1, 3});  // 1 sample, 3 features
    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;

    NDArray output = layer.forward(input);
    assertEqual(output.shape().size(), static_cast<size_t>(2),
                "Output should be 2D for batch processing");
    assertEqual(output.shape()[0], static_cast<size_t>(1),
                "Batch size should be 1");
    assertEqual(output.shape()[1], static_cast<size_t>(2),
                "Output features should match layer output size");

    // Test batch forward pass
    NDArray batch_input({5, 3});  // 5 samples, 3 features
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        batch_input.at({i, j}) = static_cast<double>(i + j + 1);
      }
    }

    NDArray batch_output = layer.forward(batch_input);
    assertEqual(batch_output.shape().size(), static_cast<size_t>(2),
                "Batch output should be 2D");
    assertEqual(batch_output.shape()[0], static_cast<size_t>(5),
                "Batch size should be preserved");
    assertEqual(batch_output.shape()[1], static_cast<size_t>(2),
                "Output features should match layer output size");
  }
};

/**
 * @class DenseBackwardTest
 * @brief Test Dense layer backward pass
 */
class DenseBackwardTest : public TestCase {
public:
  DenseBackwardTest() : TestCase("DenseBackwardTest") {}

protected:
  void test() override {
    using namespace MLLib::layer;

    Dense layer(3, 2);

    // Forward pass first - Dense expects 2D input
    NDArray input({1, 3});  // 1 sample, 3 features
    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;

    NDArray output = layer.forward(input);

    // Backward pass
    NDArray grad_output({1, 2});  // 1 sample, 2 output features
    grad_output[0] = 1.0;
    grad_output[1] = 0.5;

    NDArray grad_input = layer.backward(grad_output);
    assertEqual(grad_input.shape().size(), static_cast<size_t>(2),
                "Gradient input should be 2D");
    assertEqual(grad_input.shape()[0], static_cast<size_t>(1),
                "Gradient batch size should match input");
    assertEqual(grad_input.shape()[1], static_cast<size_t>(3),
                "Gradient features should match input size");

    // Test that gradients were computed (should have weight and bias gradients)
    const NDArray& weight_grad = layer.get_weight_gradients();
    const NDArray& bias_grad = layer.get_bias_gradients();

    assertEqual(weight_grad.shape().size(), static_cast<size_t>(2),
                "Weight gradients should be 2D");
    assertEqual(bias_grad.shape().size(), static_cast<size_t>(1),
                "Bias gradients should be 1D");
  }
};

/**
 * @class DenseParameterTest
 * @brief Test Dense layer parameter management
 */
class DenseParameterTest : public TestCase {
public:
  DenseParameterTest() : TestCase("DenseParameterTest") {}

protected:
  void test() override {
    using namespace MLLib::layer;

    Dense layer(2, 3);

    // Test parameter access - verify they exist and have correct dimensions
    const NDArray& weights = layer.get_weights();
    const NDArray& bias = layer.get_bias();

    assertEqual(weights.shape().size(), static_cast<size_t>(2),
                "Weights should be 2D");
    assertEqual(bias.shape().size(), static_cast<size_t>(1),
                "Bias should be 1D");

    // Test setting custom weights
    NDArray custom_weights({2, 3});  // input_size x output_size
    custom_weights[0] = 1.0;
    custom_weights[1] = 2.0;
    custom_weights[2] = 3.0;
    custom_weights[3] = 4.0;
    custom_weights[4] = 5.0;
    custom_weights[5] = 6.0;

    layer.set_weights(custom_weights);
    const NDArray& updated_weights = layer.get_weights();

    assertNear(updated_weights[0], 1.0, 1e-9, "Weights should be updated");
    assertNear(updated_weights[5], 6.0, 1e-9, "Weights should be updated");

    // Test setting custom bias
    NDArray custom_bias({3});
    custom_bias[0] = 0.1;
    custom_bias[1] = 0.2;
    custom_bias[2] = 0.3;

    layer.set_biases(custom_bias);
    const NDArray& updated_bias = layer.get_bias();

    assertNear(updated_bias[0], 0.1, 1e-9, "Bias should be updated");
    assertNear(updated_bias[2], 0.3, 1e-9, "Bias should be updated");
  }
};

}  // namespace test
}  // namespace MLLib
