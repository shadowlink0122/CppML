#pragma once

#include "../../../../include/MLLib/layer/dense.hpp"
#include "../../../../include/MLLib/model/sequential.hpp"
#include "../../../common/test_utils.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

namespace MLLib {
namespace test {

/**
 * @class SequentialModelTests
 * @brief Test Sequential model functionality
 */
class SequentialModelTests : public TestCase {
public:
  SequentialModelTests() : TestCase("SequentialModelTests") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    // Test basic construction
    Sequential model;

    // Test adding layers
    model.add(std::make_shared<Dense>(3, 4));
    model.add(std::make_shared<Dense>(4, 2));

    // Basic forward pass test
    NDArray input({1, 3});  // 1 sample, 3 features (2D for batch processing)
    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;

    NDArray output = model.predict(input);
    assertEqual(output.shape().size(), static_cast<size_t>(2),
                "Output should be 2D");
    assertEqual(output.shape()[0], static_cast<size_t>(1),
                "Batch size should be 1");
    assertEqual(output.shape()[1], static_cast<size_t>(2),
                "Output should have 2 features");

    std::cout << "Sequential model basic test passed" << std::endl;
  }
};

}  // namespace test
}  // namespace MLLib
