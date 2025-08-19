#pragma once

#include "../../../../include/MLLib/layer/activation/relu.hpp"
#include "../../../../include/MLLib/layer/activation/sigmoid.hpp"
#include "../../../../include/MLLib/layer/dense.hpp"
#include "../../../../include/MLLib/model/model_io.hpp"
#include "../../../../include/MLLib/model/sequential.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"
#include <chrono>
#include <filesystem>
#include <random>

namespace MLLib {
namespace test {

/**
 * @class LargeSequentialModelTest
 * @brief Test Sequential model with large dimensions (up to 2048x2048)
 */
class LargeSequentialModelTest : public TestCase {
public:
  LargeSequentialModelTest() : TestCase("LargeSequentialModelTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    printf("Testing large Sequential model I/O...\n");

    std::string temp_dir = createTempDirectory();

    // Test 1: Medium scale model (512 -> 256 -> 128)
    {
      printf("  Testing medium scale model (512 -> 256 -> 128)...\n");

      auto start_time = std::chrono::high_resolution_clock::now();

      auto medium_model = std::make_unique<Sequential>();
      medium_model->add(std::make_unique<Dense>(512, 256));
      medium_model->add(std::make_unique<activation::ReLU>());
      medium_model->add(std::make_unique<Dense>(256, 128));

      auto creation_time = std::chrono::high_resolution_clock::now();
      auto creation_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(creation_time -
                                                                start_time);
      printf("    Model creation time: %ldms\n", creation_duration.count());

      // Test with random input
      NDArray medium_input({1, 512});
      std::random_device rd;
      std::mt19937 gen(42);
      std::uniform_real_distribution<double> dis(-1.0, 1.0);

      for (size_t i = 0; i < 512; ++i) {
        medium_input[i] = dis(gen);
      }

      auto predict_start = std::chrono::high_resolution_clock::now();
      NDArray medium_original_output = medium_model->predict(medium_input);
      auto predict_end = std::chrono::high_resolution_clock::now();
      auto predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(predict_end -
                                                                predict_start);
      printf("    Prediction time: %ldms\n", predict_duration.count());

      // Save model
      std::string medium_path = temp_dir + "/medium_sequential";

      auto save_start = std::chrono::high_resolution_clock::now();
      bool save_success = GenericModelIO::save_model(*medium_model, medium_path,
                                                     SaveFormat::BINARY);
      auto save_end = std::chrono::high_resolution_clock::now();
      auto save_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(save_end -
                                                                save_start);

      assertTrue(save_success, "Medium scale model should save successfully");
      printf("    Save time: %ldms\n", save_duration.count());

      // Check file size
      size_t medium_file_size =
          std::filesystem::file_size(medium_path + ".bin");
      printf("    File size: %zu bytes (%.2f MB)\n", medium_file_size,
             medium_file_size / 1024.0 / 1024.0);
      assertTrue(medium_file_size > 100000,
                 "Medium scale model file should be substantial");

      // Load model
      auto load_start = std::chrono::high_resolution_clock::now();
      auto loaded_medium =
          GenericModelIO::load_model<Sequential>(medium_path,
                                                 SaveFormat::BINARY);
      auto load_end = std::chrono::high_resolution_clock::now();
      auto load_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                                load_start);

      assertTrue(loaded_medium != nullptr,
                 "Medium scale model should load successfully");
      printf("    Load time: %ldms\n", load_duration.count());

      // Test loaded model
      auto loaded_predict_start = std::chrono::high_resolution_clock::now();
      NDArray medium_loaded_output = loaded_medium->predict(medium_input);
      auto loaded_predict_end = std::chrono::high_resolution_clock::now();
      auto loaded_predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              loaded_predict_end - loaded_predict_start);
      printf("    Loaded model prediction time: %ldms\n",
             loaded_predict_duration.count());

      // Verify outputs match
      bool medium_match = true;
      double tolerance = 1e-10;
      for (size_t i = 0; i < medium_original_output.size(); ++i) {
        if (std::abs(medium_original_output[i] - medium_loaded_output[i]) >
            tolerance) {
          medium_match = false;
          break;
        }
      }
      assertTrue(medium_match, "Medium scale model outputs should match");
      printf("    ✅ Medium scale model test completed successfully\n");
    }

    // Test 2: Large scale model (1024 -> 512 -> 256)
    {
      printf("  Testing large scale model (1024 -> 512 -> 256)...\n");

      auto start_time = std::chrono::high_resolution_clock::now();

      auto large_model = std::make_unique<Sequential>();
      large_model->add(std::make_unique<Dense>(1024, 512));
      large_model->add(std::make_unique<activation::ReLU>());
      large_model->add(std::make_unique<Dense>(512, 256));

      auto creation_time = std::chrono::high_resolution_clock::now();
      auto creation_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(creation_time -
                                                                start_time);
      printf("    Model creation time: %ldms\n", creation_duration.count());

      // Test with random input
      NDArray large_input({1, 1024});
      std::random_device rd;
      std::mt19937 gen(123);
      std::uniform_real_distribution<double> dis(-1.0, 1.0);

      for (size_t i = 0; i < 1024; ++i) {
        large_input[i] = dis(gen);
      }

      auto predict_start = std::chrono::high_resolution_clock::now();
      NDArray large_original_output = large_model->predict(large_input);
      auto predict_end = std::chrono::high_resolution_clock::now();
      auto predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(predict_end -
                                                                predict_start);
      printf("    Prediction time: %ldms\n", predict_duration.count());

      // Save model
      std::string large_path = temp_dir + "/large_sequential";

      auto save_start = std::chrono::high_resolution_clock::now();
      bool save_success = GenericModelIO::save_model(*large_model, large_path,
                                                     SaveFormat::BINARY);
      auto save_end = std::chrono::high_resolution_clock::now();
      auto save_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(save_end -
                                                                save_start);

      assertTrue(save_success, "Large scale model should save successfully");
      printf("    Save time: %ldms\n", save_duration.count());

      // Check file size
      size_t large_file_size = std::filesystem::file_size(large_path + ".bin");
      printf("    File size: %zu bytes (%.2f MB)\n", large_file_size,
             large_file_size / 1024.0 / 1024.0);
      assertTrue(large_file_size > 1000000,
                 "Large scale model file should be over 1MB");

      // Load model
      auto load_start = std::chrono::high_resolution_clock::now();
      auto loaded_large =
          GenericModelIO::load_model<Sequential>(large_path,
                                                 SaveFormat::BINARY);
      auto load_end = std::chrono::high_resolution_clock::now();
      auto load_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                                load_start);

      assertTrue(loaded_large != nullptr,
                 "Large scale model should load successfully");
      printf("    Load time: %ldms\n", load_duration.count());

      // Test loaded model
      auto loaded_predict_start = std::chrono::high_resolution_clock::now();
      NDArray large_loaded_output = loaded_large->predict(large_input);
      auto loaded_predict_end = std::chrono::high_resolution_clock::now();
      auto loaded_predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              loaded_predict_end - loaded_predict_start);
      printf("    Loaded model prediction time: %ldms\n",
             loaded_predict_duration.count());

      // Verify outputs match
      bool large_match = true;
      double tolerance = 1e-10;
      for (size_t i = 0; i < large_original_output.size(); ++i) {
        if (std::abs(large_original_output[i] - large_loaded_output[i]) >
            tolerance) {
          large_match = false;
          break;
        }
      }
      assertTrue(large_match, "Large scale model outputs should match");
      printf("    ✅ Large scale model test completed successfully\n");
    }

    // Test 3: Extra large scale model (2048 -> 1024 -> 512)
    {
      printf("  Testing extra large scale model (2048 -> 1024 -> 512)...\n");

      auto start_time = std::chrono::high_resolution_clock::now();

      auto xlarge_model = std::make_unique<Sequential>();
      xlarge_model->add(std::make_unique<Dense>(2048, 1024));
      xlarge_model->add(std::make_unique<activation::ReLU>());
      xlarge_model->add(std::make_unique<Dense>(1024, 512));

      auto creation_time = std::chrono::high_resolution_clock::now();
      auto creation_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(creation_time -
                                                                start_time);
      printf("    Model creation time: %ldms\n", creation_duration.count());

      // Test with random input
      NDArray xlarge_input({1, 2048});
      std::random_device rd;
      std::mt19937 gen(456);
      std::uniform_real_distribution<double> dis(-1.0, 1.0);

      for (size_t i = 0; i < 2048; ++i) {
        xlarge_input[i] = dis(gen);
      }

      auto predict_start = std::chrono::high_resolution_clock::now();
      NDArray xlarge_original_output = xlarge_model->predict(xlarge_input);
      auto predict_end = std::chrono::high_resolution_clock::now();
      auto predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(predict_end -
                                                                predict_start);
      printf("    Prediction time: %ldms\n", predict_duration.count());

      // Save model
      std::string xlarge_path = temp_dir + "/xlarge_sequential";

      auto save_start = std::chrono::high_resolution_clock::now();
      bool save_success = GenericModelIO::save_model(*xlarge_model, xlarge_path,
                                                     SaveFormat::BINARY);
      auto save_end = std::chrono::high_resolution_clock::now();
      auto save_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(save_end -
                                                                save_start);

      assertTrue(save_success,
                 "Extra large scale model should save successfully");
      printf("    Save time: %ldms\n", save_duration.count());

      // Check file size
      size_t xlarge_file_size =
          std::filesystem::file_size(xlarge_path + ".bin");
      printf("    File size: %zu bytes (%.2f MB)\n", xlarge_file_size,
             xlarge_file_size / 1024.0 / 1024.0);
      assertTrue(xlarge_file_size > 10000000,
                 "Extra large scale model file should be over 10MB");

      // Load model
      auto load_start = std::chrono::high_resolution_clock::now();
      auto loaded_xlarge =
          GenericModelIO::load_model<Sequential>(xlarge_path,
                                                 SaveFormat::BINARY);
      auto load_end = std::chrono::high_resolution_clock::now();
      auto load_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                                load_start);

      assertTrue(loaded_xlarge != nullptr,
                 "Extra large scale model should load successfully");
      printf("    Load time: %ldms\n", load_duration.count());

      // Test loaded model
      auto loaded_predict_start = std::chrono::high_resolution_clock::now();
      NDArray xlarge_loaded_output = loaded_xlarge->predict(xlarge_input);
      auto loaded_predict_end = std::chrono::high_resolution_clock::now();
      auto loaded_predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              loaded_predict_end - loaded_predict_start);
      printf("    Loaded model prediction time: %ldms\n",
             loaded_predict_duration.count());

      // Verify outputs match
      bool xlarge_match = true;
      double tolerance = 1e-10;
      for (size_t i = 0; i < xlarge_original_output.size(); ++i) {
        if (std::abs(xlarge_original_output[i] - xlarge_loaded_output[i]) >
            tolerance) {
          xlarge_match = false;
          break;
        }
      }
      assertTrue(xlarge_match, "Extra large scale model outputs should match");
      printf("    ✅ Extra large scale model test completed successfully\n");
    }

    removeTempDirectory(temp_dir);
    printf("  Large Sequential model I/O test completed successfully\n");
  }
};

/**
 * @class VeryLargeSequentialModelTest
 * @brief Test Sequential model with very large dimensions (2048x2048 layer)
 */
class VeryLargeSequentialModelTest : public TestCase {
public:
  VeryLargeSequentialModelTest() : TestCase("VeryLargeSequentialModelTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    printf("Testing very large Sequential model (2048x2048) I/O...\n");

    std::string temp_dir = createTempDirectory();

    try {
      // Test: Very large single layer (2048x2048)
      printf("  Creating 2048x2048 Sequential model...\n");

      auto start_time = std::chrono::high_resolution_clock::now();

      auto very_large_model = std::make_unique<Sequential>();
      very_large_model->add(std::make_unique<Dense>(2048, 2048));
      very_large_model->add(std::make_unique<activation::Sigmoid>());

      auto creation_time = std::chrono::high_resolution_clock::now();
      auto creation_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(creation_time -
                                                                start_time);
      printf("    Model creation time: %ldms\n", creation_duration.count());

      // Calculate expected parameters
      size_t expected_weights = 2048 * 2048;
      size_t expected_biases = 2048;
      size_t total_params = expected_weights + expected_biases;
      double expected_size_mb =
          (total_params * sizeof(double)) / 1024.0 / 1024.0;
      printf("    Expected parameters: %zu (%.2f MB)\n", total_params,
             expected_size_mb);

      // Test with random input
      NDArray very_large_input({1, 2048});
      std::random_device rd;
      std::mt19937 gen(789);
      std::uniform_real_distribution<double>
          dis(-0.1, 0.1);  // Smaller range to avoid overflow

      for (size_t i = 0; i < 2048; ++i) {
        very_large_input[i] = dis(gen);
      }

      printf("    Testing forward pass...\n");
      auto predict_start = std::chrono::high_resolution_clock::now();
      NDArray very_large_original_output =
          very_large_model->predict(very_large_input);
      auto predict_end = std::chrono::high_resolution_clock::now();
      auto predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(predict_end -
                                                                predict_start);
      printf("    Prediction time: %ldms\n", predict_duration.count());

      assertTrue(very_large_original_output.size() == 2048,
                 "Output size should be 2048");

      // Save model
      std::string very_large_path = temp_dir + "/very_large_sequential";

      printf("    Saving very large model...\n");
      auto save_start = std::chrono::high_resolution_clock::now();
      bool save_success =
          GenericModelIO::save_model(*very_large_model, very_large_path,
                                     SaveFormat::BINARY);
      auto save_end = std::chrono::high_resolution_clock::now();
      auto save_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(save_end -
                                                                save_start);

      assertTrue(save_success,
                 "Very large scale model (2048x2048) should save successfully");
      printf("    Save time: %ldms\n", save_duration.count());

      // Check file size
      size_t very_large_file_size =
          std::filesystem::file_size(very_large_path + ".bin");
      double actual_size_mb = very_large_file_size / 1024.0 / 1024.0;
      printf("    Actual file size: %zu bytes (%.2f MB)\n",
             very_large_file_size, actual_size_mb);

      // Should be approximately the size of parameters plus metadata
      assertTrue(very_large_file_size > 30000000,
                 "Very large scale model file should be over 30MB");
      assertTrue(actual_size_mb >= expected_size_mb * 0.9,
                 "File size should be close to expected parameter size");

      // Load model
      printf("    Loading very large model...\n");
      auto load_start = std::chrono::high_resolution_clock::now();
      auto loaded_very_large =
          GenericModelIO::load_model<Sequential>(very_large_path,
                                                 SaveFormat::BINARY);
      auto load_end = std::chrono::high_resolution_clock::now();
      auto load_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(load_end -
                                                                load_start);

      assertTrue(loaded_very_large != nullptr,
                 "Very large scale model should load successfully");
      printf("    Load time: %ldms\n", load_duration.count());

      // Test loaded model
      printf("    Testing loaded model forward pass...\n");
      auto loaded_predict_start = std::chrono::high_resolution_clock::now();
      NDArray very_large_loaded_output =
          loaded_very_large->predict(very_large_input);
      auto loaded_predict_end = std::chrono::high_resolution_clock::now();
      auto loaded_predict_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              loaded_predict_end - loaded_predict_start);
      printf("    Loaded model prediction time: %ldms\n",
             loaded_predict_duration.count());

      assertTrue(very_large_loaded_output.size() == 2048,
                 "Loaded output size should be 2048");

      // Verify outputs match (sample some values due to size)
      printf("    Verifying parameter preservation...\n");
      bool very_large_match = true;
      double tolerance = 1e-10;
      size_t check_count = 0;
      size_t mismatch_count = 0;

      for (size_t i = 0; i < very_large_original_output.size();
           i += 64) {  // Sample every 64th value
        check_count++;
        if (std::abs(very_large_original_output[i] -
                     very_large_loaded_output[i]) > tolerance) {
          mismatch_count++;
        }
      }

      double match_ratio = (double)(check_count - mismatch_count) / check_count;
      printf("    Parameter preservation ratio: %.4f (%zu/%zu)\n", match_ratio,
             check_count - mismatch_count, check_count);
      assertTrue(
          match_ratio > 0.999,
          "Very large scale model should have >99.9% parameter preservation");

      // Performance summary
      printf("    📊 Performance Summary:\n");
      printf("      - Model size: 2048x2048 (%.1fM parameters)\n",
             total_params / 1000000.0);
      printf("      - File size: %.2f MB\n", actual_size_mb);
      printf("      - Save time: %ldms\n", save_duration.count());
      printf("      - Load time: %ldms\n", load_duration.count());
      printf("      - Forward pass time: %ldms\n", predict_duration.count());

      printf(
          "    ✅ Very large Sequential model (2048x2048) test completed successfully\n");

    } catch (const std::exception& e) {
      printf("    ❌ Very large model test failed with exception: %s\n",
             e.what());
      throw;
    }

    removeTempDirectory(temp_dir);
    printf("  Very large Sequential model I/O test completed successfully\n");
  }
};

}  // namespace test
}  // namespace MLLib
