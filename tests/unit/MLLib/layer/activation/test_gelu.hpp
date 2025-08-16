#pragma once

#include "../../../../../include/MLLib/layer/activation/gelu.hpp"
#include "../../../../common/test_utils.hpp"

/**
 * @file test_gelu.hpp
 * @brief Unit tests for GELU activation function
 */

namespace MLLib {
namespace test {

class GELUTest : public TestCase {
public:
  GELUTest() : TestCase("GELU Test") {}

protected:
  void test() override {
    layer::activation::GELU gelu(false);  // Use exact GELU

    // Test forward pass with simple values
    NDArray input({3});
    input[0] = -1.0;
    input[1] = 0.0;
    input[2] = 1.0;

    NDArray output = gelu.forward(input);

    // GELU(0) should be 0
    assertNear(output[1], 0.0, 1e-9, "GELU for zero input");

    // GELU should be approximately x/2 for large positive x
    // For x=1, GELU(1) ≈ 0.8413
    assertTrue(output[2] > 0.5,
               "GELU for positive input should be positive and substantial");

    // GELU should be small for negative inputs
    assertTrue(output[0] < 0.0 && output[0] > -0.5,
               "GELU for negative input should be small negative");
  }
};

class GELUApproximateTest : public TestCase {
public:
  GELUApproximateTest() : TestCase("GELU Approximate Test") {}

protected:
  void test() override {
    layer::activation::GELU gelu_approx(true);  // Use approximate GELU

    // Test forward pass
    NDArray input({3});
    input[0] = -1.0;
    input[1] = 0.0;
    input[2] = 1.0;

    NDArray output = gelu_approx.forward(input);

    // GELU(0) should be 0
    assertNear(output[1], 0.0, 1e-9, "Approximate GELU for zero input");

    // Approximate GELU should give similar results to exact GELU
    assertTrue(
        output[2] > 0.5,
        "Approximate GELU for positive input should be positive and substantial");
    assertTrue(output[0] < 0.0 && output[0] > -0.5,
               "Approximate GELU for negative input should be small negative");
  }
};

class GELUErrorTest : public TestCase {
public:
  GELUErrorTest() : TestCase("GELU Error Test") {}

protected:
  void test() override {
    // Test backward without forward
    assertThrows<std::runtime_error>(
        []() {
          layer::activation::GELU gelu;
          NDArray grad_output({2});
          gelu.backward(grad_output);
        },
        "GELU should throw when backward called before forward");
  }
};

}  // namespace test
}  // namespace MLLib
