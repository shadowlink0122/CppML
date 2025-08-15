#include "../common/test_utils.hpp"
#include "MLLib/layer/activation/test_activation.hpp"
#include "MLLib/layer/test_dense.hpp"
// #include "MLLib/model/test_model_io.hpp"  // Temporarily disabled
#include "MLLib/backend/test_gpu_backend.hpp"
#include "MLLib/model/test_sequential.hpp"
#include "MLLib/test_config.hpp"
#include "MLLib/test_ndarray.hpp"
#include <chrono>
#include <iomanip>

/**
 * @file unit_test_main.cpp
 * @brief Main entry point for unit tests
 *
 * This file contains all unit tests for the MLLib library.
 * Tests are organized by component and run in a structured manner.
 */

int main() {
  using namespace MLLib::test;

  printf("=== MLLib Unit Test Suite ===\n");
  printf("Running comprehensive unit tests for MLLib v1.0.0\n");
  printf("Test execution with output capture enabled\n");
  printf("\n");

  // Start timing the entire test suite
  auto suite_start_time = std::chrono::high_resolution_clock::now();

  bool all_tests_passed = true;
  int total_tests = 0;
  int passed_tests = 0;
  double total_execution_time = 0.0;

  // Helper function to run test and update counters
  auto runTest = [&](std::unique_ptr<TestCase> test) {
    total_tests++;
    bool result = test->run();
    if (result) passed_tests++;
    all_tests_passed &= result;
    total_execution_time += test->getExecutionTimeMs();
  };

  // Config tests
  std::cout << "\n--- Config Module Tests ---" << std::endl;
  runTest(std::make_unique<ConfigConstantsTest>());
  runTest(std::make_unique<ConfigUsageTest>());
  runTest(std::make_unique<ConfigMathTest>());

  // NDArray tests
  std::cout << "\n--- NDArray Module Tests ---" << std::endl;
  runTest(std::make_unique<NDArrayConstructorTest>());
  runTest(std::make_unique<NDArrayAccessTest>());
  runTest(std::make_unique<NDArrayOperationsTest>());
  runTest(std::make_unique<NDArrayArithmeticTest>());
  runTest(std::make_unique<NDArrayMatmulTest>());
  runTest(std::make_unique<NDArrayErrorTest>());

  // Dense layer tests
  std::cout << "\n--- Dense Layer Tests ---" << std::endl;
  runTest(std::make_unique<DenseConstructorTest>());
  runTest(std::make_unique<DenseForwardTest>());
  runTest(std::make_unique<DenseBackwardTest>());
  runTest(std::make_unique<DenseParameterTest>());

  // Activation function tests
  std::cout << "\n--- Activation Function Tests ---" << std::endl;
  runTest(std::make_unique<ReLUTest>());
  runTest(std::make_unique<ReLUBackwardTest>());
  runTest(std::make_unique<SigmoidTest>());
  runTest(std::make_unique<SigmoidBackwardTest>());
  runTest(std::make_unique<TanhTest>());
  runTest(std::make_unique<TanhBackwardTest>());
  runTest(std::make_unique<ActivationErrorTest>());

  // Sequential model tests
  std::cout << "\n--- Sequential Model Tests ---" << std::endl;
  runTest(std::make_unique<SequentialModelTests>());

  // GPU backend tests
  printf("\n--- GPU Backend Tests ---\n");
  runTest(std::make_unique<GPUAvailabilityTest>());
  runTest(std::make_unique<GPUDeviceValidationTest>());
  runTest(std::make_unique<GPUBackendOperationsTest>());
  runTest(std::make_unique<GPUArrayOperationsTest>());
  runTest(std::make_unique<GPUModelTest>());
  runTest(std::make_unique<GPUPerformanceTest>());

  // Multi-GPU tests
  printf("\n--- Multi-GPU Support Tests ---\n");
  runTest(std::make_unique<MultiGPUDetectionTest>());
  runTest(std::make_unique<GPUBackendTypesTest>());
  runTest(std::make_unique<MultiGPUBackendOperationsTest>());
  runTest(std::make_unique<GPUVendorPriorityTest>());
  runTest(std::make_unique<GPUMemoryTest>());
  runTest(std::make_unique<GPUErrorHandlingTest>());
  runTest(std::make_unique<GPUCompilationTest>());

  // Model I/O tests (temporarily disabled)
  // std::cout << "\n--- Model I/O Tests ---" << std::endl;
  // runTest(std::make_unique<ModelFormatTest>());
  // runTest(std::make_unique<ModelSaveLoadTest>());
  // runTest(std::make_unique<ModelParameterTest>());
  // runTest(std::make_unique<ModelIOErrorTest>());
  // runTest(std::make_unique<ModelIOFileHandlingTest>());

  // Print final summary
  printf("\n============================================================\n");
  printf("FINAL TEST SUMMARY\n");
  printf("============================================================\n");
  printf("Total individual tests: %d\n", total_tests);
  printf("Passed tests: %d\n", passed_tests);
  printf("Failed tests: %d\n", (total_tests - passed_tests));

  // Calculate total suite execution time
  auto suite_end_time = std::chrono::high_resolution_clock::now();
  auto suite_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      suite_end_time - suite_start_time);
  double suite_time_ms = suite_duration.count() / 1000.0;

  printf("Total test execution time: %.2fms\n", total_execution_time);
  printf("Total suite time (including overhead): %.2fms\n", suite_time_ms);
  printf("\n");
  if (all_tests_passed) {
    printf("🎉 ALL UNIT TESTS PASSED! 🎉\n");
    printf("MLLib is ready for production use.\n");
  } else {
    printf("❌ SOME UNIT TESTS FAILED\n");
    printf("Please review the test output above and fix the issues.\n");
  }

  printf("============================================================\n");

  return all_tests_passed ? 0 : 1;
}
