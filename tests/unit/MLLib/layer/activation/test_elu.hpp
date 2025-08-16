#pragma once

#include "../../../../../include/MLLib/layer/activation/elu.hpp"
#include "../../../../common/test_utils.hpp"

/**
 * @file test_elu.hpp
 * @brief Unit tests for ELU activation function
 */

namespace MLLib {
namespace test {

class ELUTest : public TestCase {
public:
  ELUTest() : TestCase("ELU Test") {}

protected:
  void test() override {
    layer::activation::ELU elu(1.0);
    assertNear(elu.get_alpha(), 1.0, 1e-9, "Alpha parameter");

    // Test forward pass
    NDArray input({4});
    input[0] = -2.0;
    input[1] = -0.5;
    input[2] = 0.0;
    input[3] = 1.0;

    NDArray output = elu.forward(input);

    assertNear(output[0], 1.0 * (std::exp(-2.0) - 1), 1e-9,
               "ELU for negative input");
    assertNear(output[1], 1.0 * (std::exp(-0.5) - 1), 1e-9,
               "ELU for negative input");
    assertNear(output[2], 0.0, 1e-9, "Zero input");
    assertNear(output[3], 1.0, 1e-9, "Positive input");

    // Test backward pass
    NDArray grad_output({4});
    grad_output.fill(1.0);

    NDArray grad_input = elu.backward(grad_output);

    assertNear(grad_input[0], 1.0 * std::exp(-2.0), 1e-9,
               "ELU gradient for negative input");
    assertNear(grad_input[1], 1.0 * std::exp(-0.5), 1e-9,
               "ELU gradient for negative input");
    assertNear(grad_input[2], 1.0, 1e-9, "ELU gradient at zero");
    assertNear(grad_input[3], 1.0, 1e-9, "ELU gradient for positive input");
  }
};

class ELUErrorTest : public TestCase {
public:
  ELUErrorTest() : TestCase("ELU Error Test") {}

protected:
  void test() override {
    // Test ELU with negative alpha
    assertThrows<std::invalid_argument>(
        []() {
          layer::activation::ELU elu(-0.5);
        },
        "ELU should throw for negative alpha");

    // Test backward without forward
    assertThrows<std::runtime_error>(
        []() {
          layer::activation::ELU elu;
          NDArray grad_output({2});
          elu.backward(grad_output);
        },
        "ELU should throw when backward called before forward");
  }
};

}  // namespace test
}  // namespace MLLib
