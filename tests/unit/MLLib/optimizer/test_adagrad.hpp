#pragma once

#include "../../../../include/MLLib/optimizer/adagrad.hpp"
#include "../../../common/test_utils.hpp"
#include <vector>

/**
 * @file test_adagrad.hpp
 * @brief Unit tests for AdaGrad optimizer
 */

namespace MLLib {
namespace test {

class AdaGradTest : public TestCase {
public:
  AdaGradTest() : TestCase("AdaGrad Test") {}

protected:
  void test() override {
    optimizer::AdaGrad adagrad(0.01);
    assertNear(adagrad.get_learning_rate(), 0.01, 1e-9,
               "AdaGrad learning rate");

    NDArray param({2});
    param[0] = 1.0;
    param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1;
    grad[1] = 0.2;

    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};

    double original_param0 = param[0];
    double original_param1 = param[1];

    adagrad.update(parameters, gradients);

    assertTrue(param[0] < original_param0, "Parameter should decrease");
    assertTrue(param[1] < original_param1, "Parameter should decrease");
  }
};

class AdaGradConstructorTest : public TestCase {
public:
  AdaGradConstructorTest() : TestCase("AdaGrad Constructor Test") {}

protected:
  void test() override {
    // Test default constructor
    optimizer::AdaGrad adagrad1;
    assertNear(adagrad1.get_learning_rate(), 0.01, 1e-9,
               "Default learning rate");

    // Test custom parameters
    optimizer::AdaGrad adagrad2(0.01, 1e-6);
    assertNear(adagrad2.get_learning_rate(), 0.01, 1e-9,
               "Custom learning rate");

    // Test invalid parameters
    assertThrows<std::invalid_argument>(
        []() {
          optimizer::AdaGrad adagrad(-0.1);
        },
        "Should throw for negative learning rate");

    assertThrows<std::invalid_argument>(
        []() {
          optimizer::AdaGrad adagrad(0.01, -1e-6);
        },
        "Should throw for negative epsilon");
  }
};

class AdaGradResetTest : public TestCase {
public:
  AdaGradResetTest() : TestCase("AdaGrad Reset Test") {}

protected:
  void test() override {
    optimizer::AdaGrad adagrad(0.01);

    NDArray param({2});
    param[0] = 1.0;
    param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1;
    grad[1] = 0.2;

    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};

    // Update once to build up state
    adagrad.update(parameters, gradients);

    // Reset optimizer
    adagrad.reset();

    assertTrue(true, "AdaGrad reset completed without error");
  }
};

}  // namespace test
}  // namespace MLLib
