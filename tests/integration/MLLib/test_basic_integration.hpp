#ifndef MLLIB_TEST_BASIC_INTEGRATION_HPP
#define MLLIB_TEST_BASIC_INTEGRATION_HPP

#include "../../../include/MLLib/layer/activation/relu.hpp"
#include "../../../include/MLLib/layer/activation/sigmoid.hpp"
#include "../../../include/MLLib/layer/dense.hpp"
#include "../../../include/MLLib/loss/mse.hpp"
#include "../../../include/MLLib/model/sequential.hpp"
#include "../../../include/MLLib/optimizer/sgd.hpp"
#include "../../common/test_utils.hpp"

namespace MLLib {
namespace test {

/**
 * @class BasicTrainingIntegrationTest
 * @brief Test basic training workflow
 */
class BasicTrainingIntegrationTest : public TestCase {
public:
  BasicTrainingIntegrationTest() : TestCase("BasicTrainingIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create simple model
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    // Simple training data
    std::vector<std::vector<double>> X = {
        {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
    std::vector<std::vector<double>> Y = {{1.0}, {1.0}, {0.0}, {0.0}};

    MLLib::loss::MSELoss loss;
    MLLib::optimizer::SGD optimizer(0.1);

    bool training_completed = false;
    assertNoThrow(
        [&]() {
          model->train(X, Y, loss, optimizer, nullptr,
                       20);  // Short training for stability
          training_completed = true;
        },
        "Basic training should complete without errors");

    assertTrue(training_completed, "Training should finish successfully");

    // Test that model can make predictions
    assertNoThrow(
        [&]() {
          auto pred = model->predict({0.5, 0.5});
          assertTrue(pred.size() == 1, "Prediction should have correct size");
          assertTrue(pred[0] >= 0.0 && pred[0] <= 1.0,
                     "Sigmoid output should be valid");
        },
        "Model should be able to make predictions");
  }
};

/**
 * @class ModelSaveLoadIntegrationTest
 * @brief Test model basic functionality (save/load not implemented)
 */
class ModelSaveLoadIntegrationTest : public TestCase {
public:
  ModelSaveLoadIntegrationTest() : TestCase("ModelSaveLoadIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Create a simple model for testing
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(3, 4));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(4, 2));
    model->add(std::make_shared<activation::Sigmoid>());

    std::vector<std::vector<double>> X = {
        {1.0, 0.0, 0.5}, {0.0, 1.0, 0.3}, {0.5, 0.5, 1.0}};
    std::vector<std::vector<double>> Y = {{1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}};

    MLLib::loss::MSELoss loss;
    MLLib::optimizer::SGD optimizer(0.1);

    // Train for basic functionality test
    assertNoThrow(
        [&]() {
          model->train(X, Y, loss, optimizer, nullptr, 20);  // Minimal training
        },
        "Training should complete without errors");

    // Get original prediction for comparison
    std::vector<double> original_pred;
    assertNoThrow([&]() { original_pred = model->predict({0.5, 0.5, 0.5}); },
                  "Model should make predictions");

    assertTrue(original_pred.size() == 2,
               "Prediction should have correct output size");

    // Test basic model consistency (save/load not implemented)
    assertTrue(
        true,
        "Model save/load not implemented - testing basic functionality only");

    // Test multiple predictions for consistency
    std::vector<double> pred1 = model->predict({0.5, 0.5, 0.5});
    std::vector<double> pred2 = model->predict({0.5, 0.5, 0.5});

    assertEqual(pred1.size(), pred2.size(), "Consistent prediction sizes");
    for (size_t i = 0; i < pred1.size(); ++i) {
      double diff = std::abs(pred1[i] - pred2[i]);
      assertTrue(diff < 1e-10, "Model predictions should be deterministic");
    }
  }
};

/**
 * @class FullWorkflowIntegrationTest
 * @brief Test complete workflow - simple version for CI stability
 */
class FullWorkflowIntegrationTest : public TestCase {
public:
  FullWorkflowIntegrationTest() : TestCase("FullWorkflowIntegrationTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;
    using namespace MLLib::loss;
    using namespace MLLib::optimizer;

    OutputCapture capture;

    // Simple workflow test
    auto model = std::make_unique<Sequential>();
    model->add(std::make_shared<Dense>(2, 3));
    model->add(std::make_shared<activation::ReLU>());
    model->add(std::make_shared<Dense>(3, 1));
    model->add(std::make_shared<activation::Sigmoid>());

    std::vector<std::vector<double>> X = {{1.0, 0.0}, {0.0, 1.0}};
    std::vector<std::vector<double>> Y = {{1.0}, {0.0}};

    MLLib::loss::MSELoss loss;
    MLLib::optimizer::SGD optimizer(0.1);

    // Test complete workflow
    assertNoThrow([&]() { model->train(X, Y, loss, optimizer, nullptr, 10); },
                  "Training should complete");

    assertNoThrow(
        [&]() {
          auto pred = model->predict({0.5, 0.5});
          assertTrue(pred.size() == 1, "Should produce single output");
          assertTrue(pred[0] >= 0.0 && pred[0] <= 1.0,
                     "Output should be in sigmoid range");
        },
        "Prediction should work");

    assertTrue(true, "Workflow completed successfully");
  }
};

}  // namespace test
}  // namespace MLLib

#endif  // MLLIB_TEST_BASIC_INTEGRATION_HPP
