#pragma once

#include "../../../../include/MLLib/optimizer/adam.hpp"
#include "../../../common/test_utils.hpp"
#include <cmath>
#include <vector>

/**
 * @file test_adam.hpp
 * @brief Unit tests for Adam optimizer
 */

namespace MLLib {
namespace test {

class AdamConstructorTest : public TestCase {
public:
  AdamConstructorTest() : TestCase("Adam Constructor Test") {}

protected:
  void test() override {
    optimizer::Adam adam1;
    assertNear(adam1.get_learning_rate(), 0.001, 1e-9, "Default learning rate");

    optimizer::Adam adam2(0.01);
    assertNear(adam2.get_learning_rate(), 0.01, 1e-9, "Custom learning rate");

    // Test invalid parameters
    assertThrows<std::invalid_argument>(
        []() {
          optimizer::Adam adam(-0.1);
        },
        "Should throw for negative learning rate");

    assertThrows<std::invalid_argument>(
        []() {
          optimizer::Adam adam(0.01, -0.5);
        },
        "Should throw for invalid beta1");
  }
};

class AdamUpdateTest : public TestCase {
public:
  AdamUpdateTest() : TestCase("Adam Update Test") {}

protected:
  void test() override {
    optimizer::Adam adam(0.01);

    // Create test parameters
    NDArray param({3});
    param[0] = 1.0;
    param[1] = 2.0;
    param[2] = 3.0;

    // Store original values
    std::vector<double> original_values;
    for (size_t i = 0; i < param.size(); ++i) {
      original_values.push_back(param[i]);
    }

    // Create gradient
    NDArray grad({3});
    grad[0] = 0.1;
    grad[1] = 0.2;
    grad[2] = 0.3;

    // Update parameters with vector interface
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};

    adam.update(parameters, gradients);

    // Parameters should have changed
    for (size_t i = 0; i < param.size(); ++i) {
      assertTrue(param[i] < original_values[i],
                 "Parameter should decrease with positive gradient");
    }
  }
};

class AdamResetTest : public TestCase {
public:
  AdamResetTest() : TestCase("Adam Reset Test") {}

protected:
  void test() override {
    optimizer::Adam adam(0.01);

    // Create test parameters and gradient
    NDArray param({2});
    param[0] = 1.0;
    param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1;
    grad[1] = 0.2;

    // Update once with vector interface
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};

    adam.update(parameters, gradients);

    // Reset optimizer
    adam.reset();

    assertTrue(true, "Adam reset completed without error");
  }
};

class AdamErrorTest : public TestCase {
public:
  AdamErrorTest() : TestCase("Adam Error Test") {}

protected:
  void test() override {
    optimizer::Adam adam(0.01);

    // Test mismatched parameter and gradient sizes
    NDArray param({3});
    param.fill(1.0);

    NDArray grad({2});  // Different size
    grad.fill(0.1);

    // This should throw an error
    assertThrows<std::invalid_argument>(
        [&]() {
          std::vector<NDArray*> parameters = {&param};
          std::vector<NDArray*> gradients = {&grad};
          adam.update(parameters, gradients);
        },
        "Should throw for size mismatch");
  }
};

}  // namespace test
}  // namespace MLLib
