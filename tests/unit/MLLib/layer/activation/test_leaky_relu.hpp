#pragma once

#include "../../../../../include/MLLib/layer/activation/leaky_relu.hpp"
#include "../../../../common/test_utils.hpp"

/**
 * @file test_leaky_relu.hpp
 * @brief Unit tests for Leaky ReLU activation function
 */

namespace MLLib {
namespace test {

class LeakyReLUTest : public TestCase {
public:
  LeakyReLUTest() : TestCase("LeakyReLU Test") {}
  
protected:
  void test() override {
    layer::activation::LeakyReLU leaky_relu(0.01);
    assertNear(leaky_relu.get_alpha(), 0.01, 1e-9, "Alpha parameter");
    
    // Test forward pass
    NDArray input({4});
    input[0] = -2.0; input[1] = -0.5; input[2] = 0.0; input[3] = 1.0;
    
    NDArray output = leaky_relu.forward(input);
    
    assertNear(output[0], -0.02, 1e-9, "Negative input * alpha");
    assertNear(output[1], -0.005, 1e-9, "Negative input * alpha");
    assertNear(output[2], 0.0, 1e-9, "Zero input");
    assertNear(output[3], 1.0, 1e-9, "Positive input");
    
    // Test backward pass
    NDArray grad_output({4});
    grad_output[0] = 1.0; grad_output[1] = 1.0; grad_output[2] = 1.0; grad_output[3] = 1.0;
    
    NDArray grad_input = leaky_relu.backward(grad_output);
    
    assertNear(grad_input[0], 0.01, 1e-9, "Gradient for negative input");
    assertNear(grad_input[1], 0.01, 1e-9, "Gradient for negative input");
    assertNear(grad_input[2], 0.01, 1e-9, "Gradient at zero");
    assertNear(grad_input[3], 1.0, 1e-9, "Gradient for positive input");
  }
};

class LeakyReLUErrorTest : public TestCase {
public:
  LeakyReLUErrorTest() : TestCase("LeakyReLU Error Test") {}
  
protected:
  void test() override {
    // Test invalid alpha
    assertThrows<std::invalid_argument>([]() {
      layer::activation::LeakyReLU leaky_relu(-0.1);
    }, "Should throw for negative alpha");
    
    // Test backward without forward
    assertThrows<std::runtime_error>([]() {
      layer::activation::LeakyReLU leaky_relu(0.01);
      NDArray grad_output({2});
      leaky_relu.backward(grad_output);
    }, "Should throw when backward called before forward");
  }
};

}  // namespace test
}  // namespace MLLib
