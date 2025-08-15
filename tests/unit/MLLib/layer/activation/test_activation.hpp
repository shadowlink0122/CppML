#pragma once

#include "../../../../../include/MLLib/layer/activation/relu.hpp"
#include "../../../../../include/MLLib/layer/activation/sigmoid.hpp"
#include "../../../../../include/MLLib/layer/activation/tanh.hpp"
#include "../../../../../include/MLLib/ndarray.hpp"
#include "../../../../common/test_utils.hpp"
#include <cmath>

namespace MLLib {
namespace test {

/**
 * @class ReLUTest
 * @brief Test ReLU activation function
 */
class ReLUTest : public TestCase {
public:
  ReLUTest() : TestCase("ReLUTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    ReLU relu;

    // Test positive values (should remain unchanged)
    NDArray positive_input({4});
    positive_input[0] = 1.0;
    positive_input[1] = 2.5;
    positive_input[2] = 0.1;
    positive_input[3] = 10.0;

    NDArray positive_output = relu.forward(positive_input);

    assertNear(1.0, positive_output[0], 1e-9, "ReLU(1.0) should be 1.0");
    assertNear(2.5, positive_output[1], 1e-9, "ReLU(2.5) should be 2.5");
    assertNear(0.1, positive_output[2], 1e-9, "ReLU(0.1) should be 0.1");
    assertNear(10.0, positive_output[3], 1e-9, "ReLU(10.0) should be 10.0");

    // Test negative values (should become zero)
    NDArray negative_input({4});
    negative_input[0] = -1.0;
    negative_input[1] = -2.5;
    negative_input[2] = -0.1;
    negative_input[3] = -10.0;

    NDArray negative_output = relu.forward(negative_input);

    assertNear(0.0, negative_output[0], 1e-9, "ReLU(-1.0) should be 0.0");
    assertNear(0.0, negative_output[1], 1e-9, "ReLU(-2.5) should be 0.0");
    assertNear(0.0, negative_output[2], 1e-9, "ReLU(-0.1) should be 0.0");
    assertNear(0.0, negative_output[3], 1e-9, "ReLU(-10.0) should be 0.0");

    // Test zero
    NDArray zero_input({1});
    zero_input[0] = 0.0;
    NDArray zero_output = relu.forward(zero_input);
    assertNear(0.0, zero_output[0], 1e-9, "ReLU(0.0) should be 0.0");

    // Test mixed values
    NDArray mixed_input({4});
    mixed_input[0] = -2.0;
    mixed_input[1] = 0.0;
    mixed_input[2] = 3.0;
    mixed_input[3] = -1.5;

    NDArray mixed_output = relu.forward(mixed_input);

    assertNear(0.0, mixed_output[0], 1e-9, "ReLU(-2.0) should be 0.0");
    assertNear(0.0, mixed_output[1], 1e-9, "ReLU(0.0) should be 0.0");
    assertNear(3.0, mixed_output[2], 1e-9, "ReLU(3.0) should be 3.0");
    assertNear(0.0, mixed_output[3], 1e-9, "ReLU(-1.5) should be 0.0");
  }
};

/**
 * @class ReLUBackwardTest
 * @brief Test ReLU backward propagation
 */
class ReLUBackwardTest : public TestCase {
public:
  ReLUBackwardTest() : TestCase("ReLUBackwardTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    ReLU relu;

    // Forward pass first
    NDArray input({4});
    input[0] = -1.0;  // Will be clipped to 0
    input[1] = 0.0;   // Boundary case
    input[2] = 2.0;   // Will remain 2.0
    input[3] = -0.5;  // Will be clipped to 0

    NDArray output = relu.forward(input);

    // Backward pass
    NDArray grad_output({4});
    grad_output.fill(1.0);  // All gradients from next layer are 1.0

    NDArray grad_input = relu.backward(grad_output);

    // ReLU derivative: 1 if input > 0, 0 if input <= 0
    assertNear(0.0, grad_input[0], 1e-9, "ReLU gradient at -1.0 should be 0.0");
    assertNear(0.0, grad_input[1], 1e-9, "ReLU gradient at 0.0 should be 0.0");
    assertNear(1.0, grad_input[2], 1e-9, "ReLU gradient at 2.0 should be 1.0");
    assertNear(0.0, grad_input[3], 1e-9, "ReLU gradient at -0.5 should be 0.0");

    // Test with different gradients from next layer
    NDArray grad_output2({4});
    grad_output2[0] = 2.0;
    grad_output2[1] = 3.0;
    grad_output2[2] = 4.0;
    grad_output2[3] = 5.0;

    NDArray grad_input2 = relu.backward(grad_output2);

    assertNear(0.0, grad_input2[0], 1e-9,
               "ReLU gradient should be 0 regardless of upstream gradient");
    assertNear(0.0, grad_input2[1], 1e-9,
               "ReLU gradient should be 0 regardless of upstream gradient");
    assertNear(4.0, grad_input2[2], 1e-9,
               "ReLU gradient should pass through upstream gradient");
    assertNear(0.0, grad_input2[3], 1e-9,
               "ReLU gradient should be 0 regardless of upstream gradient");
  }
};

/**
 * @class SigmoidTest
 * @brief Test Sigmoid activation function
 */
class SigmoidTest : public TestCase {
public:
  SigmoidTest() : TestCase("SigmoidTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    Sigmoid sigmoid;

    // Test known values
    NDArray input({5});
    input[0] = 0.0;    // sigmoid(0) = 0.5
    input[1] = 1.0;    // sigmoid(1) ≈ 0.731
    input[2] = -1.0;   // sigmoid(-1) ≈ 0.269
    input[3] = 10.0;   // sigmoid(10) ≈ 1.0 (very large positive)
    input[4] = -10.0;  // sigmoid(-10) ≈ 0.0 (very large negative)

    NDArray output = sigmoid.forward(input);

    assertNear(0.5, output[0], 1e-6, "sigmoid(0) should be 0.5");
    assertNear(0.731058579, output[1], 1e-6,
               "sigmoid(1) should be approximately 0.731");
    assertNear(0.268941421, output[2], 1e-6,
               "sigmoid(-1) should be approximately 0.269");
    assertTrue(output[3] > 0.99, "sigmoid(10) should be very close to 1.0");
    assertTrue(output[4] < 0.01, "sigmoid(-10) should be very close to 0.0");

    // Test that all outputs are between 0 and 1
    for (size_t i = 0; i < output.size(); ++i) {
      assertTrue(output[i] >= 0.0 && output[i] <= 1.0,
                 "Sigmoid output should be between 0 and 1");
    }

    // Test symmetry property: sigmoid(-x) = 1 - sigmoid(x)
    NDArray symmetric_input({3});
    symmetric_input[0] = 2.0;
    symmetric_input[1] = -2.0;
    symmetric_input[2] = 0.0;

    NDArray symmetric_output = sigmoid.forward(symmetric_input);

    assertNear(1.0 - symmetric_output[1], symmetric_output[0], 1e-9,
               "Sigmoid should satisfy symmetry property");
  }
};

/**
 * @class SigmoidBackwardTest
 * @brief Test Sigmoid backward propagation
 */
class SigmoidBackwardTest : public TestCase {
public:
  SigmoidBackwardTest() : TestCase("SigmoidBackwardTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    Sigmoid sigmoid;

    // Forward pass
    NDArray input({3});
    input[0] = 0.0;   // sigmoid(0) = 0.5
    input[1] = 2.0;   // sigmoid(2) ≈ 0.881
    input[2] = -2.0;  // sigmoid(-2) ≈ 0.119

    NDArray output = sigmoid.forward(input);

    // Backward pass
    NDArray grad_output({3});
    grad_output.fill(1.0);

    NDArray grad_input = sigmoid.backward(grad_output);

    // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    double expected_grad_0 = output[0] * (1.0 - output[0]);  // 0.5 * 0.5 = 0.25
    double expected_grad_1 = output[1] * (1.0 - output[1]);
    double expected_grad_2 = output[2] * (1.0 - output[2]);

    assertNear(expected_grad_0, grad_input[0], 1e-6,
               "Sigmoid gradient at 0 should be 0.25");
    assertNear(expected_grad_1, grad_input[1], 1e-6,
               "Sigmoid gradient should match derivative formula");
    assertNear(expected_grad_2, grad_input[2], 1e-6,
               "Sigmoid gradient should match derivative formula");

    // Test maximum gradient occurs at sigmoid(0) = 0.5
    assertTrue(grad_input[0] >= grad_input[1],
               "Gradient at 0 should be >= gradient at 2");
    assertTrue(grad_input[0] >= grad_input[2],
               "Gradient at 0 should be >= gradient at -2");

    // Test with custom upstream gradients
    NDArray custom_grad({3});
    custom_grad[0] = 2.0;
    custom_grad[1] = 0.5;
    custom_grad[2] = 3.0;

    NDArray custom_grad_input = sigmoid.backward(custom_grad);

    assertNear(expected_grad_0 * 2.0, custom_grad_input[0], 1e-6,
               "Gradient should be scaled by upstream gradient");
    assertNear(expected_grad_1 * 0.5, custom_grad_input[1], 1e-6,
               "Gradient should be scaled by upstream gradient");
    assertNear(expected_grad_2 * 3.0, custom_grad_input[2], 1e-6,
               "Gradient should be scaled by upstream gradient");
  }
};

/**
 * @class TanhTest
 * @brief Test Tanh activation function
 */
class TanhTest : public TestCase {
public:
  TanhTest() : TestCase("TanhTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    Tanh tanh_layer;

    // Test known values
    NDArray input({5});
    input[0] = 0.0;   // tanh(0) = 0
    input[1] = 1.0;   // tanh(1) ≈ 0.762
    input[2] = -1.0;  // tanh(-1) ≈ -0.762
    input[3] = 5.0;   // tanh(5) ≈ 1.0 (saturated)
    input[4] = -5.0;  // tanh(-5) ≈ -1.0 (saturated)

    NDArray output = tanh_layer.forward(input);

    assertNear(0.0, output[0], 1e-9, "tanh(0) should be 0");
    assertNear(0.761594156, output[1], 1e-6,
               "tanh(1) should be approximately 0.762");
    assertNear(-0.761594156, output[2], 1e-6,
               "tanh(-1) should be approximately -0.762");
    assertTrue(output[3] > 0.99, "tanh(5) should be very close to 1.0");
    assertTrue(output[4] < -0.99, "tanh(-5) should be very close to -1.0");

    // Test that all outputs are between -1 and 1
    for (size_t i = 0; i < output.size(); ++i) {
      assertTrue(output[i] >= -1.0 && output[i] <= 1.0,
                 "Tanh output should be between -1 and 1");
    }

    // Test odd function property: tanh(-x) = -tanh(x)
    NDArray symmetric_input({3});
    symmetric_input[0] = 2.0;
    symmetric_input[1] = -2.0;
    symmetric_input[2] = 0.0;

    NDArray symmetric_output = tanh_layer.forward(symmetric_input);

    assertNear(-symmetric_output[1], symmetric_output[0], 1e-9,
               "Tanh should satisfy odd function property");
    assertNear(0.0, symmetric_output[2], 1e-9, "tanh(0) should be exactly 0");
  }
};

/**
 * @class TanhBackwardTest
 * @brief Test Tanh backward propagation
 */
class TanhBackwardTest : public TestCase {
public:
  TanhBackwardTest() : TestCase("TanhBackwardTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    Tanh tanh_layer;

    // Forward pass
    NDArray input({3});
    input[0] = 0.0;   // tanh(0) = 0
    input[1] = 1.0;   // tanh(1) ≈ 0.762
    input[2] = -1.0;  // tanh(-1) ≈ -0.762

    NDArray output = tanh_layer.forward(input);

    // Backward pass
    NDArray grad_output({3});
    grad_output.fill(1.0);

    NDArray grad_input = tanh_layer.backward(grad_output);

    // Tanh derivative: 1 - tanh²(x)
    double expected_grad_0 = 1.0 - output[0] * output[0];  // 1 - 0² = 1
    double expected_grad_1 = 1.0 - output[1] * output[1];
    double expected_grad_2 = 1.0 - output[2] * output[2];

    assertNear(expected_grad_0, grad_input[0], 1e-6,
               "Tanh gradient at 0 should be 1.0");
    assertNear(expected_grad_1, grad_input[1], 1e-6,
               "Tanh gradient should match derivative formula");
    assertNear(expected_grad_2, grad_input[2], 1e-6,
               "Tanh gradient should match derivative formula");

    // Test maximum gradient occurs at tanh(0) = 0
    assertTrue(grad_input[0] >= grad_input[1],
               "Gradient at 0 should be >= gradient at 1");
    assertTrue(grad_input[0] >= grad_input[2],
               "Gradient at 0 should be >= gradient at -1");

    // Due to symmetry, gradients at ±1 should be equal
    assertNear(grad_input[1], grad_input[2], 1e-9,
               "Gradients at ±1 should be equal");
  }
};

/**
 * @class ActivationErrorTest
 * @brief Test activation function error conditions
 */
class ActivationErrorTest : public TestCase {
public:
  ActivationErrorTest() : TestCase("ActivationErrorTest") {}

protected:
  void test() override {
    using namespace MLLib::layer::activation;

    ReLU relu;
    Sigmoid sigmoid;
    Tanh tanh_layer;

    // Test backward without forward
    NDArray grad({3});
    grad.fill(1.0);

    assertThrows<std::runtime_error>(
        [&]() { relu.backward(grad); },
        "ReLU backward without forward should throw");

    assertThrows<std::runtime_error>(
        [&]() { sigmoid.backward(grad); },
        "Sigmoid backward without forward should throw");

    assertThrows<std::runtime_error>(
        [&]() { tanh_layer.backward(grad); },
        "Tanh backward without forward should throw");

    // Test with mismatched gradient size
    NDArray input({3});
    input.fill(1.0);

    relu.forward(input);
    sigmoid.forward(input);
    tanh_layer.forward(input);

    NDArray wrong_grad({2});  // Wrong size
    wrong_grad.fill(1.0);

    assertThrows<std::invalid_argument>(
        [&]() { relu.backward(wrong_grad); },
        "ReLU backward with wrong gradient size should throw");

    assertThrows<std::invalid_argument>(
        [&]() { sigmoid.backward(wrong_grad); },
        "Sigmoid backward with wrong gradient size should throw");

    assertThrows<std::invalid_argument>(
        [&]() { tanh_layer.backward(wrong_grad); },
        "Tanh backward with wrong gradient size should throw");
  }
};

}  // namespace test
}  // namespace MLLib
