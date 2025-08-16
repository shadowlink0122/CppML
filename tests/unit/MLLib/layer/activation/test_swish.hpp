#pragma once

#include "../../../../../include/MLLib/layer/activation/swish.hpp"
#include "../../../../common/test_utils.hpp"

/**
 * @file test_swish.hpp
 * @brief Unit tests for Swish activation function
 */

namespace MLLib {
namespace test {

class SwishTest : public TestCase {
public:
  SwishTest() : TestCase("Swish Test") {}
  
protected:
  void test() override {
    layer::activation::Swish swish;
    
    // Test forward pass
    NDArray input({3});
    input[0] = -1.0; input[1] = 0.0; input[2] = 1.0;
    
    NDArray output = swish.forward(input);
    
    // Swish(x) = x * sigmoid(x)
    double expected_0 = -1.0 * (1.0 / (1.0 + std::exp(1.0)));
    double expected_1 = 0.0;
    double expected_2 = 1.0 * (1.0 / (1.0 + std::exp(-1.0)));
    
    assertNear(output[0], expected_0, 1e-6, "Swish for negative input");
    assertNear(output[1], expected_1, 1e-9, "Swish for zero input");
    assertNear(output[2], expected_2, 1e-6, "Swish for positive input");
  }
};

class SwishErrorTest : public TestCase {
public:
  SwishErrorTest() : TestCase("Swish Error Test") {}
  
protected:
  void test() override {
    // Test backward without forward
    assertThrows<std::runtime_error>([]() {
      layer::activation::Swish swish;
      NDArray grad_output({2});
      swish.backward(grad_output);
    }, "Swish should throw when backward called before forward");
  }
};

}  // namespace test
}  // namespace MLLib
