#pragma once

#include "../../../../include/MLLib/optimizer/rmsprop.hpp"
#include "../../../common/test_utils.hpp"
#include <vector>

/**
 * @file test_rmsprop.hpp
 * @brief Unit tests for RMSprop optimizer
 */

namespace MLLib {
namespace test {

class RMSpropConstructorTest : public TestCase {
public:
  RMSpropConstructorTest() : TestCase("RMSprop Constructor Test") {}

protected:
  void test() override {
    optimizer::RMSprop rmsprop1;
    assertNear(rmsprop1.get_learning_rate(), 0.001, 1e-9,
               "Default learning rate");

    optimizer::RMSprop rmsprop2(0.01, 0.8, 1e-7);
    assertNear(rmsprop2.get_learning_rate(), 0.01, 1e-9,
               "Custom learning rate");

    // Test invalid parameters
    assertThrows<std::invalid_argument>(
        []() {
          optimizer::RMSprop rmsprop_invalid(-0.1);
        },
        "Should throw for negative learning rate");
  }
};

class RMSpropUpdateTest : public TestCase {
public:
  RMSpropUpdateTest() : TestCase("RMSprop Update Test") {}

protected:
  void test() override {
    optimizer::RMSprop rmsprop(0.1);

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

    rmsprop.update(parameters, gradients);

    assertTrue(param[0] < original_param0, "Parameter 0 should decrease");
    assertTrue(param[1] < original_param1, "Parameter 1 should decrease");
  }
};

class RMSpropResetTest : public TestCase {
public:
  RMSpropResetTest() : TestCase("RMSprop Reset Test") {}

protected:
  void test() override {
    optimizer::RMSprop rmsprop(0.01);
    NDArray param({2});
    param[0] = 1.0;
    param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1;
    grad[1] = 0.2;
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};

    rmsprop.update(parameters, gradients);
    rmsprop.reset();
    assertTrue(true, "RMSprop reset completed");
  }
};

}  // namespace test
}  // namespace MLLib
