#pragma once

#include "../../../../../include/MLLib/layer/activation/softmax.hpp"
#include "../../../../common/test_utils.hpp"

/**
 * @file test_softmax.hpp
 * @brief Unit tests for Softmax activation function
 */

namespace MLLib {
namespace test {

class SoftmaxTest : public TestCase {
public:
  SoftmaxTest() : TestCase("Softmax Test") {}
  
protected:
  void test() override {
    layer::activation::Softmax softmax;
    
    // Test forward pass with 2D array (1 batch, 3 features)
    NDArray input({1, 3});
    input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;
    
    NDArray output = softmax.forward(input);
    
    // Check output is valid probability distribution
    double sum = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
      assertTrue(output[i] > 0.0, "Softmax output should be positive");
      sum += output[i];
    }
    assertNear(sum, 1.0, 1e-9, "Softmax outputs should sum to 1");
    
    // Verify relative ordering
    assertTrue(output[2] > output[1] && output[1] > output[0], 
               "Softmax should preserve relative ordering");
  }
};

class SoftmaxBatchTest : public TestCase {
public:
  SoftmaxBatchTest() : TestCase("Softmax Batch Test") {}
  
protected:
  void test() override {
    layer::activation::Softmax softmax;
    
    // Test forward pass with 2D array (2 batches, 3 features)
    NDArray input({2, 3});
    input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;
    input[3] = 3.0; input[4] = 2.0; input[5] = 1.0;
    
    NDArray output = softmax.forward(input);
    
    // Check both batches have valid probability distributions
    // First batch
    double sum1 = output[0] + output[1] + output[2];
    assertNear(sum1, 1.0, 1e-9, "First batch should sum to 1");
    
    // Second batch
    double sum2 = output[3] + output[4] + output[5];
    assertNear(sum2, 1.0, 1e-9, "Second batch should sum to 1");
  }
};

class SoftmaxErrorTest : public TestCase {
public:
  SoftmaxErrorTest() : TestCase("Softmax Error Test") {}
  
protected:
  void test() override {
    // Test backward without forward
    assertThrows<std::runtime_error>([]() {
      layer::activation::Softmax softmax;
      NDArray grad_output({2});
      softmax.backward(grad_output);
    }, "Softmax should throw when backward called before forward");
    
    // Test 1D input (should throw)
    assertThrows<std::invalid_argument>([]() {
      layer::activation::Softmax softmax;
      NDArray input({3});
      input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;
      softmax.forward(input);
    }, "Softmax should throw for 1D input");
  }
};

}  // namespace test
}  // namespace MLLib
