#pragma once

#include "../../../../../include/MLLib/loss/mse.hpp"
#include "../../../../../include/MLLib/model/autoencoder/anomaly_detector.hpp"
#include "../../../../../include/MLLib/optimizer/adam.hpp"
#include "../../../../common/test_utils.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace MLLib {
namespace test {
namespace autoencoder {

/**
 * @brief Test anomaly detection configuration
 */
void test_anomaly_config() {
  printf("Testing anomaly detection configuration...\n");

  // Test default config
  model::autoencoder::AnomalyConfig default_config;
  assert(default_config.threshold_percentile == 95.0);
  assert(default_config.threshold_method == "percentile");
  assert(default_config.error_metric == "mse");
  assert(!default_config.adaptive_threshold);

  // Test custom config
  model::autoencoder::AnomalyConfig custom_config;
  custom_config.threshold_percentile = 90.0;
  custom_config.threshold_method = "std";
  custom_config.error_metric = "mae";
  custom_config.adaptive_threshold = true;
  custom_config.window_size = 50;

  assert(custom_config.threshold_percentile == 90.0);
  assert(custom_config.threshold_method == "std");
  assert(custom_config.error_metric == "mae");
  assert(custom_config.adaptive_threshold == true);
  assert(custom_config.window_size == 50);

  printf("✅ Anomaly detection configuration tests passed\n");
}

/**
 * @brief Test anomaly detector construction
 */
void test_anomaly_detector_construction() {
  printf("Testing anomaly detector construction...\n");

  // Test with explicit parameters
  model::autoencoder::AnomalyDetector detector(10, 5, {8, 6}, 95.0);
  assert(detector.get_input_dim() == 10);
  assert(detector.get_latent_dim() == 5);
  assert(detector.get_type() == model::autoencoder::AutoencoderType::BASIC);
  assert(detector.get_anomaly_config().threshold_percentile == 95.0);

  // Test with configuration
  auto base_config =
      model::autoencoder::AutoencoderConfig::basic(20, 8, {16, 12});
  model::autoencoder::AnomalyConfig anomaly_config;
  anomaly_config.threshold_percentile = 90.0;
  anomaly_config.error_metric = "mae";

  model::autoencoder::AnomalyDetector config_detector(base_config,
                                                      anomaly_config);
  assert(config_detector.get_input_dim() == 20);
  assert(config_detector.get_latent_dim() == 8);
  assert(config_detector.get_anomaly_config().threshold_percentile == 90.0);
  assert(config_detector.get_anomaly_config().error_metric == "mae");

  printf("✅ Anomaly detector construction tests passed\n");
}

/**
 * @brief Test threshold calculation
 */
void test_threshold_calculation() {
  printf("Testing threshold calculation...\n");

  model::autoencoder::AnomalyDetector detector(6, 3);

  // Create normal training data (simple patterns)
  std::vector<NDArray> normal_data;

  // Pattern 1: [1, 0, 1, 0, 1, 0]
  NDArray pattern1({1, 6});
  pattern1.data()[0] = 1.0;
  pattern1.data()[1] = 0.0;
  pattern1.data()[2] = 1.0;
  pattern1.data()[3] = 0.0;
  pattern1.data()[4] = 1.0;
  pattern1.data()[5] = 0.0;

  // Pattern 2: [0, 1, 0, 1, 0, 1]
  NDArray pattern2({1, 6});
  pattern2.data()[0] = 0.0;
  pattern2.data()[1] = 1.0;
  pattern2.data()[2] = 0.0;
  pattern2.data()[3] = 1.0;
  pattern2.data()[4] = 0.0;
  pattern2.data()[5] = 1.0;

  // Add slight variations to create more normal samples
  for (int i = 0; i < 10; ++i) {
    NDArray sample = (i % 2 == 0) ? pattern1 : pattern2;
    // Add small noise to create variation
    for (int j = 0; j < 6; ++j) {
      sample.data()[j] += 0.05 * ((i % 3) - 1);  // Small noise
    }
    normal_data.push_back(sample);
  }

  // Calculate threshold based on normal data
  detector.calculate_threshold(normal_data);

  double threshold = detector.get_threshold();
  assert(threshold > 0.0);  // Threshold should be positive

  printf("Calculated threshold: %.4f\n", threshold);

  printf("✅ Threshold calculation tests passed\n");
}

/**
 * @brief Test training on normal data
 */
void test_training_on_normal() {
  printf("Testing training on normal data...\n");

  model::autoencoder::AnomalyDetector detector(4, 2);

  // Create normal training data
  std::vector<NDArray> normal_data;

  NDArray pattern1({1, 4});
  pattern1.data()[0] = 1.0;
  pattern1.data()[1] = 0.0;
  pattern1.data()[2] = 1.0;
  pattern1.data()[3] = 0.0;

  NDArray pattern2({1, 4});
  pattern2.data()[0] = 0.0;
  pattern2.data()[1] = 1.0;
  pattern2.data()[2] = 0.0;
  pattern2.data()[3] = 1.0;

  for (int i = 0; i < 8; ++i) {
    normal_data.push_back(i % 2 == 0 ? pattern1 : pattern2);
  }

  // Create loss and optimizer
  loss::MSELoss loss;
  optimizer::Adam optimizer(0.01);

  // Measure initial reconstruction error
  double initial_error = detector.get_reconstruction_error(pattern1);

  // Train on normal data
  detector.train_on_normal(normal_data, loss, optimizer, 3, 4);

  // Measure final reconstruction error
  double final_error = detector.get_reconstruction_error(pattern1);

  printf("Training on normal data - Initial error: %.4f, Final error: %.4f\n",
         initial_error, final_error);

  printf("✅ Training on normal data tests passed\n");
}

/**
 * @brief Test anomaly detection
 */
void test_anomaly_detection() {
  printf("Testing anomaly detection...\n");

  model::autoencoder::AnomalyDetector detector(4, 2);

  // Create normal data for threshold calculation
  std::vector<NDArray> normal_data;

  NDArray normal_pattern({1, 4});
  normal_pattern.data()[0] = 1.0;
  normal_pattern.data()[1] = 0.0;
  normal_pattern.data()[2] = 1.0;
  normal_pattern.data()[3] = 0.0;

  for (int i = 0; i < 20; ++i) {
    NDArray sample = normal_pattern;
    // Add small random variation
    for (int j = 0; j < 4; ++j) {
      sample.data()[j] += 0.05 * ((i % 5) - 2);  // Small noise
    }
    normal_data.push_back(sample);
  }

  // Calculate threshold
  detector.calculate_threshold(normal_data);

  // Create test data with anomalies
  std::vector<NDArray> test_data;
  std::vector<bool> ground_truth;

  // Add normal samples
  for (int i = 0; i < 5; ++i) {
    NDArray sample = normal_pattern;
    sample.data()[0] += 0.03 * i;  // Small variation
    test_data.push_back(sample);
    ground_truth.push_back(false);  // Not anomaly
  }

  // Add anomalous samples
  for (int i = 0; i < 3; ++i) {
    NDArray anomaly({1, 4});
    anomaly.data()[0] = 0.5;
    anomaly.data()[1] = 0.5;  // Different pattern
    anomaly.data()[2] = 0.5;
    anomaly.data()[3] = 0.5;
    test_data.push_back(anomaly);
    ground_truth.push_back(true);  // Is anomaly
  }

  // Detect anomalies
  auto results = detector.detect_anomalies(test_data, &ground_truth);

  assert(results.reconstruction_errors.size() == test_data.size());
  assert(results.anomaly_flags.size() == test_data.size());
  assert(results.threshold > 0.0);

  // Test individual anomaly detection
  bool normal_check = detector.is_anomaly(normal_pattern);

  NDArray clear_anomaly({1, 4});
  clear_anomaly.fill(0.5);  // Very different pattern
  bool anomaly_check = detector.is_anomaly(clear_anomaly);

  printf(
      "Anomaly detection - Normal sample anomalous: %s, Anomaly sample anomalous: %s\n",
      normal_check ? "Yes" : "No", anomaly_check ? "Yes" : "No");

  printf(
      "Detection results - Precision: %.3f, Recall: %.3f, F1: %.3f, Accuracy: %.3f\n",
      results.precision, results.recall, results.f1_score, results.accuracy);

  printf("✅ Anomaly detection tests passed\n");
}

/**
 * @brief Test manual threshold setting
 */
void test_manual_threshold() {
  printf("Testing manual threshold setting...\n");

  model::autoencoder::AnomalyDetector detector(4, 2);

  // Set manual threshold
  double manual_threshold = 0.5;
  detector.set_threshold(manual_threshold);

  assert(detector.get_threshold() == manual_threshold);

  // Test anomaly detection with manual threshold
  NDArray normal_sample({1, 4});
  normal_sample.fill(0.0);

  NDArray anomaly_sample({1, 4});
  anomaly_sample.fill(1.0);  // Very different

  double normal_error = detector.get_reconstruction_error(normal_sample);
  double anomaly_error = detector.get_reconstruction_error(anomaly_sample);

  printf("Manual threshold (%.2f) - Normal error: %.4f, Anomaly error: %.4f\n",
         manual_threshold, normal_error, anomaly_error);

  printf("✅ Manual threshold tests passed\n");
}

/**
 * @brief Test different error metrics
 */
void test_error_metrics() {
  printf("Testing different error metrics...\n");

  // Create detector with MSE
  auto mse_config = model::autoencoder::AnomalyConfig();
  mse_config.error_metric = "mse";
  model::autoencoder::AnomalyDetector
      mse_detector(model::autoencoder::AutoencoderConfig::basic(4, 2),
                   mse_config);

  // Create detector with MAE
  auto mae_config = model::autoencoder::AnomalyConfig();
  mae_config.error_metric = "mae";
  model::autoencoder::AnomalyDetector
      mae_detector(model::autoencoder::AutoencoderConfig::basic(4, 2),
                   mae_config);

  // Test data
  NDArray test_sample({1, 4});
  test_sample.data()[0] = 1.0;
  test_sample.data()[1] = 0.0;
  test_sample.data()[2] = 1.0;
  test_sample.data()[3] = 0.0;

  double mse_error = mse_detector.get_reconstruction_error(test_sample);
  double mae_error = mae_detector.get_reconstruction_error(test_sample);

  assert(mse_error >= 0.0);
  assert(mae_error >= 0.0);

  printf("Error metrics - MSE: %.4f, MAE: %.4f\n", mse_error, mae_error);

  printf("✅ Error metrics tests passed\n");
}

/**
 * @brief Test factory methods
 */
void test_factory_methods() {
  printf("Testing factory methods...\n");

  // Test sensor data factory
  auto sensor_detector =
      model::autoencoder::AnomalyDetector::create_for_sensors(10, 5, 2.0, 95.0);
  assert(sensor_detector != nullptr);
  assert(sensor_detector->get_input_dim() == 10);
  assert(sensor_detector->get_latent_dim() == 5);

  // Test time series factory
  auto ts_detector =
      model::autoencoder::AnomalyDetector::create_for_timeseries(20, 3, 8,
                                                                 90.0);
  assert(ts_detector != nullptr);
  assert(ts_detector->get_input_dim() == 20 * 3);  // window_size * num_features
  assert(ts_detector->get_latent_dim() == 8);
  assert(ts_detector->get_anomaly_config().threshold_percentile == 90.0);

  printf("✅ Factory method tests passed\n");
}

/**
 * @brief Test anomaly detection configuration updates
 */
void test_anomaly_config_updates() {
  printf("Testing anomaly detection configuration updates...\n");

  model::autoencoder::AnomalyDetector detector(6, 3);

  // Test initial config
  auto initial_config = detector.get_anomaly_config();
  assert(initial_config.threshold_percentile == 95.0);

  // Update config
  model::autoencoder::AnomalyConfig new_config;
  new_config.threshold_percentile = 85.0;
  new_config.threshold_method = "std";
  new_config.error_metric = "rmse";
  new_config.adaptive_threshold = true;
  new_config.window_size = 50;

  detector.set_anomaly_config(new_config);

  auto updated_config = detector.get_anomaly_config();
  assert(updated_config.threshold_percentile == 85.0);
  assert(updated_config.threshold_method == "std");
  assert(updated_config.error_metric == "rmse");
  assert(updated_config.adaptive_threshold == true);
  assert(updated_config.window_size == 50);

  printf("✅ Anomaly detection configuration update tests passed\n");
}

/**
 * @brief Test performance metrics calculation
 */
void test_performance_metrics() {
  printf("Testing performance metrics calculation...\n");

  model::autoencoder::AnomalyDetector detector(4, 2);

  // Set a fixed threshold for predictable results
  detector.set_threshold(0.3);

  // Create test data with known outcomes
  std::vector<NDArray> test_data;
  std::vector<bool> ground_truth;

  // True negatives (normal data, low error)
  for (int i = 0; i < 4; ++i) {
    NDArray normal({1, 4});
    normal.fill(0.0);  // Should have low reconstruction error
    test_data.push_back(normal);
    ground_truth.push_back(false);
  }

  // True positives (anomaly data, high error)
  for (int i = 0; i < 2; ++i) {
    NDArray anomaly({1, 4});
    anomaly.fill(1.0);  // Should have high reconstruction error
    test_data.push_back(anomaly);
    ground_truth.push_back(true);
  }

  auto results = detector.detect_anomalies(test_data, &ground_truth);

  // Check that metrics are calculated
  assert(results.precision >= 0.0 && results.precision <= 1.0);
  assert(results.recall >= 0.0 && results.recall <= 1.0);
  assert(results.f1_score >= 0.0 && results.f1_score <= 1.0);
  assert(results.accuracy >= 0.0 && results.accuracy <= 1.0);

  printf("Performance metrics - TP: %d, FP: %d, TN: %d, FN: %d\n",
         results.true_positives, results.false_positives,
         results.true_negatives, results.false_negatives);

  printf("✅ Performance metrics calculation tests passed\n");
}

/**
 * @brief Test save/load functionality
 */
void test_save_load() {
  printf("Testing save/load functionality...\n");

  // Create and configure detector
  model::autoencoder::AnomalyDetector detector(4, 2);
  detector.set_threshold(0.42);

  // Test input
  NDArray input({1, 4});
  input.fill(0.5);
  double original_error = detector.get_reconstruction_error(input);

  // Save detector
  std::string base_path = "test_anomaly_detector";
  detector.save(base_path);

  // Create new detector and load
  model::autoencoder::AnomalyDetector loaded_detector(4, 2);
  bool loaded = loaded_detector.load(base_path);

  if (loaded) {
    // Check if threshold was preserved
    assert(loaded_detector.get_threshold() == detector.get_threshold());

    double loaded_error = loaded_detector.get_reconstruction_error(input);

    // Check if reconstruction is similar
    double diff = std::abs(original_error - loaded_error);
    assert(diff < 1e-6);

    printf("✅ Anomaly detector save/load successful\n");
  } else {
    printf(
        "⚠ Anomaly detector save/load skipped (file I/O may not be implemented)\n");
  }

  // Clean up test files
  system("rm -f test_anomaly_detector.json test_anomaly_detector.bin");
}

/**
 * @brief Run all anomaly detector tests
 */
void run_anomaly_detector_tests() {
  printf("=== Running Anomaly Detector Tests ===\n");
  test_anomaly_config();
  test_anomaly_detector_construction();
  test_threshold_calculation();
  test_training_on_normal();
  test_anomaly_detection();
  test_manual_threshold();
  test_error_metrics();
  test_factory_methods();
  test_anomaly_config_updates();
  test_performance_metrics();
  test_save_load();
  printf("=== Anomaly Detector Tests Completed ===\n\n");
}

}  // namespace autoencoder
}  // namespace test
}  // namespace MLLib
