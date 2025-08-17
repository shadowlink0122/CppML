#include "MLLib/model/autoencoder/anomaly_detector.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace MLLib {
namespace model {
namespace autoencoder {

AnomalyDetector::AnomalyDetector(const AutoencoderConfig& config,
                                const AnomalyConfig& anomaly_config)
  : BaseAutoencoder(config), anomaly_config_(anomaly_config), 
    threshold_(0.0), threshold_calculated_(false) {
}

AnomalyDetector::AnomalyDetector(int input_dim, int latent_dim,
                                const std::vector<int>& hidden_dims,
                                double threshold_percentile,
                                DeviceType device) 
  : BaseAutoencoder(AutoencoderConfig()), threshold_(0.0), threshold_calculated_(false) {
  config_.encoder_dims = {input_dim};
  config_.encoder_dims.insert(config_.encoder_dims.end(), 
                             hidden_dims.begin(), hidden_dims.end());
  config_.encoder_dims.push_back(latent_dim);
  config_.latent_dim = latent_dim;
  config_.device = device;
  
  // Calculate symmetric decoder dimensions
  config_.decoder_dims = {latent_dim};
  for (int i = static_cast<int>(config_.encoder_dims.size()) - 2; i >= 1; --i) {
    config_.decoder_dims.push_back(config_.encoder_dims[i]);
  }
  config_.decoder_dims.push_back(input_dim);
  
  anomaly_config_.threshold_percentile = threshold_percentile;
  threshold_ = 0.0;
  threshold_calculated_ = false;
  
  initialize();
}

void AnomalyDetector::train_on_normal(const std::vector<NDArray>& normal_data,
                                     loss::BaseLoss& loss,
                                     optimizer::BaseOptimizer& optimizer,
                                     int epochs,
                                     int batch_size,
                                     const std::vector<NDArray>* validation_data,
                                     std::function<void(int, double, double)> callback) {
  // Train on normal data only
  train(normal_data, loss, optimizer, epochs, batch_size, validation_data, callback);
}

void AnomalyDetector::calculate_threshold(const std::vector<NDArray>& normal_data) {
  std::vector<double> errors;
  
  for (const auto& sample : normal_data) {
    double error = reconstruction_error(sample, anomaly_config_.error_metric);
    errors.push_back(error);
  }
  
  if (anomaly_config_.threshold_method == "percentile") {
    threshold_ = calculate_percentile_threshold(errors);
  } else if (anomaly_config_.threshold_method == "std") {
    threshold_ = calculate_std_threshold(errors);
  } else if (anomaly_config_.threshold_method == "manual") {
    threshold_ = anomaly_config_.manual_threshold;
  }
  
  threshold_calculated_ = true;
}

AnomalyResults AnomalyDetector::detect_anomalies(const std::vector<NDArray>& test_data,
                                                const std::vector<bool>* ground_truth) {
  AnomalyResults results;
  
  if (!threshold_calculated_) {
    // Use a default threshold if not calculated
    threshold_ = 1.0; // Placeholder
  }
  
  results.threshold = threshold_;
  
  for (const auto& sample : test_data) {
    double error = reconstruction_error(sample, anomaly_config_.error_metric);
    results.reconstruction_errors.push_back(error);
    results.anomaly_flags.push_back(error > threshold_);
  }
  
  if (ground_truth) {
    calculate_performance_metrics(results, *ground_truth);
  }
  
  return results;
}

bool AnomalyDetector::is_anomaly(const NDArray& sample) {
  double error = get_reconstruction_error(sample);
  return error > threshold_;
}

double AnomalyDetector::get_reconstruction_error(const NDArray& sample) {
  return reconstruction_error(sample, anomaly_config_.error_metric);
}

void AnomalyDetector::set_threshold(double threshold) {
  threshold_ = threshold;
  threshold_calculated_ = true;
}

void AnomalyDetector::set_anomaly_config(const AnomalyConfig& config) {
  anomaly_config_ = config;
}

std::unique_ptr<AnomalyDetector> AnomalyDetector::create_for_sensors(
  int num_sensors,
  int latent_dim,
  double compression_ratio,
  double threshold_percentile,
  DeviceType device) {
  
  if (latent_dim == 0) {
    latent_dim = static_cast<int>(num_sensors / compression_ratio);
    latent_dim = std::max(1, latent_dim); // At least 1 dimension
  }
  
  // Create intermediate layer for smooth compression
  int intermediate_dim = static_cast<int>(std::sqrt(num_sensors * latent_dim));
  std::vector<int> hidden_dims = {intermediate_dim};
  
  return std::make_unique<AnomalyDetector>(
    num_sensors, latent_dim, hidden_dims, threshold_percentile, device);
}

std::unique_ptr<AnomalyDetector> AnomalyDetector::create_for_timeseries(
  int window_size, int num_features,
  int latent_dim,
  double threshold_percentile,
  DeviceType device) {
  
  int input_dim = window_size * num_features;
  
  if (latent_dim == 0) {
    latent_dim = std::max(1, input_dim / 4); // Default compression ratio of 4
  }
  
  // Progressive compression for time series
  std::vector<int> hidden_dims = {input_dim / 2, input_dim / 4};
  
  return std::make_unique<AnomalyDetector>(
    input_dim, latent_dim, hidden_dims, threshold_percentile, device);
}

void AnomalyDetector::save(const std::string& base_path, 
                          bool save_json, 
                          bool save_binary) {
  BaseAutoencoder::save(base_path, save_json, save_binary);
  
  // Save threshold separately
  // std::ofstream threshold_file(base_path + "_threshold.txt");
  // threshold_file << threshold_;
}

bool AnomalyDetector::load(const std::string& base_path) {
  bool success = BaseAutoencoder::load(base_path);
  
  if (success) {
    // Load threshold
    // std::ifstream threshold_file(base_path + "_threshold.txt");
    // threshold_file >> threshold_;
    // threshold_calculated_ = true;
  }
  
  return success;
}

double AnomalyDetector::calculate_percentile_threshold(const std::vector<double>& errors) {
  if (errors.empty()) return 0.0;
  
  std::vector<double> sorted_errors = errors;
  std::sort(sorted_errors.begin(), sorted_errors.end());
  
  double percentile = anomaly_config_.threshold_percentile / 100.0;
  size_t index = static_cast<size_t>(percentile * (sorted_errors.size() - 1));
  
  return sorted_errors[index];
}

double AnomalyDetector::calculate_std_threshold(const std::vector<double>& errors) {
  if (errors.empty()) return 0.0;
  
  double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
  
  double variance = 0.0;
  for (double error : errors) {
    variance += (error - mean) * (error - mean);
  }
  variance /= errors.size();
  
  double std_dev = std::sqrt(variance);
  
  // Threshold = mean + 2 * std_dev (95% confidence)
  return mean + 2.0 * std_dev;
}

void AnomalyDetector::calculate_performance_metrics(AnomalyResults& results,
                                                   const std::vector<bool>& ground_truth) {
  if (results.anomaly_flags.size() != ground_truth.size()) {
    return; // Size mismatch
  }
  
  for (size_t i = 0; i < ground_truth.size(); ++i) {
    bool predicted = results.anomaly_flags[i];
    bool actual = ground_truth[i];
    
    if (predicted && actual) {
      results.true_positives++;
    } else if (predicted && !actual) {
      results.false_positives++;
    } else if (!predicted && actual) {
      results.false_negatives++;
    } else {
      results.true_negatives++;
    }
  }
  
  // Calculate metrics
  int total_positives = results.true_positives + results.false_negatives;
  int total_predicted_positives = results.true_positives + results.false_positives;
  int total_samples = static_cast<int>(ground_truth.size());
  
  if (total_predicted_positives > 0) {
    results.precision = static_cast<double>(results.true_positives) / total_predicted_positives;
  }
  
  if (total_positives > 0) {
    results.recall = static_cast<double>(results.true_positives) / total_positives;
  }
  
  if (results.precision + results.recall > 0) {
    results.f1_score = 2.0 * results.precision * results.recall / (results.precision + results.recall);
  }
  
  results.accuracy = static_cast<double>(results.true_positives + results.true_negatives) / total_samples;
}

void AnomalyDetector::update_adaptive_threshold(const std::vector<double>& recent_errors) {
  if (!anomaly_config_.adaptive_threshold || recent_errors.empty()) {
    return;
  }
  
  // Simple adaptive threshold update - in practice you'd use more sophisticated methods
  threshold_ = calculate_percentile_threshold(recent_errors);
}

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
