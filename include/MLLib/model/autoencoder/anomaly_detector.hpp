#pragma once

#include "base.hpp"
#include <map>

/**
 * @file anomaly_detector.hpp
 * @brief Anomaly detection using autoencoders
 */

namespace MLLib {
namespace model {
namespace autoencoder {

/**
 * @struct AnomalyConfig
 * @brief Configuration for anomaly detection
 */
struct AnomalyConfig {
  double threshold_percentile = 95.0;  ///< Percentile for threshold calculation
  std::string threshold_method =
      "percentile";                  ///< "percentile", "std", "manual"
  double manual_threshold = 0.0;     ///< Manual threshold value
  std::string error_metric = "mse";  ///< Error metric ("mse", "mae", "rmse")
  bool adaptive_threshold = false;   ///< Use adaptive thresholding
  int window_size = 100;             ///< Window size for adaptive threshold
};

/**
 * @struct AnomalyResults
 * @brief Results from anomaly detection
 */
struct AnomalyResults {
  std::vector<double> reconstruction_errors;  ///< Reconstruction errors
  std::vector<bool> anomaly_flags;            ///< True for anomalies
  double threshold;                           ///< Used threshold

  // Performance metrics (if ground truth provided)
  int true_positives = 0;
  int false_positives = 0;
  int true_negatives = 0;
  int false_negatives = 0;

  double precision = 0.0;
  double recall = 0.0;
  double f1_score = 0.0;
  double accuracy = 0.0;
};

/**
 * @class AnomalyDetector
 * @brief Autoencoder-based anomaly detection
 */
class AnomalyDetector : public BaseAutoencoder {
public:
  /**
   * @brief Constructor with configuration
   * @param config Base autoencoder configuration
   * @param anomaly_config Anomaly detection configuration
   */
  AnomalyDetector(const AutoencoderConfig& config,
                  const AnomalyConfig& anomaly_config = {});

  /**
   * @brief Constructor with explicit parameters
   * @param input_dim Input dimension
   * @param latent_dim Latent dimension
   * @param hidden_dims Hidden layer dimensions
   * @param threshold_percentile Percentile for threshold (95.0 = 95th
   * percentile)
   * @param device Computation device
   */
  AnomalyDetector(int input_dim, int latent_dim,
                  const std::vector<int>& hidden_dims = {},
                  double threshold_percentile = 95.0,
                  DeviceType device = DeviceType::CPU);

  /**
   * @brief Get autoencoder type
   * @return BASIC type (used for anomaly detection)
   */
  AutoencoderType get_type() const override { return AutoencoderType::BASIC; }

  /**
   * @brief Train on normal data only
   * @param normal_data Normal training data (no anomalies)
   * @param loss Loss function
   * @param optimizer Optimizer
   * @param epochs Number of epochs
   * @param batch_size Batch size
   * @param validation_data Optional normal validation data
   * @param callback Optional training callback
   */
  void
  train_on_normal(const std::vector<NDArray>& normal_data, loss::BaseLoss& loss,
                  optimizer::BaseOptimizer& optimizer, int epochs = 100,
                  int batch_size = 32,
                  const std::vector<NDArray>* validation_data = nullptr,
                  std::function<void(int, double, double)> callback = nullptr);

  /**
   * @brief Calculate and set anomaly threshold based on normal data
   * @param normal_data Normal validation data
   */
  void calculate_threshold(const std::vector<NDArray>& normal_data);

  /**
   * @brief Detect anomalies in test data
   * @param test_data Test data to check for anomalies
   * @param ground_truth Optional ground truth labels for evaluation
   * @return Anomaly detection results
   */
  AnomalyResults
  detect_anomalies(const std::vector<NDArray>& test_data,
                   const std::vector<bool>* ground_truth = nullptr);

  /**
   * @brief Check if single sample is anomalous
   * @param sample Input sample
   * @return True if anomalous, false if normal
   */
  bool is_anomaly(const NDArray& sample);

  /**
   * @brief Get reconstruction error for single sample
   * @param sample Input sample
   * @return Reconstruction error
   */
  double get_reconstruction_error(const NDArray& sample);

  /**
   * @brief Set anomaly detection threshold manually
   * @param threshold New threshold value
   */
  void set_threshold(double threshold);

  /**
   * @brief Get current threshold
   * @return Current threshold value
   */
  double get_threshold() const { return threshold_; }

  /**
   * @brief Set anomaly configuration
   * @param config New anomaly configuration
   */
  void set_anomaly_config(const AnomalyConfig& config);

  /**
   * @brief Get current anomaly configuration
   * @return Current anomaly configuration
   */
  const AnomalyConfig& get_anomaly_config() const { return anomaly_config_; }

  /**
   * @brief Create anomaly detector for sensor data
   * @param num_sensors Number of sensors
   * @param latent_dim Latent dimension
   * @param compression_ratio Compression ratio
   * @param threshold_percentile Threshold percentile
   * @param device Computation device
   * @return AnomalyDetector instance for sensor data
   */
  static std::unique_ptr<AnomalyDetector>
  create_for_sensors(int num_sensors,
                     int latent_dim = 0,  // 0 = auto-calculate
                     double compression_ratio = 4.0,
                     double threshold_percentile = 95.0,
                     DeviceType device = DeviceType::CPU);

  /**
   * @brief Create anomaly detector for time series data
   * @param window_size Time series window size
   * @param num_features Number of features per time step
   * @param latent_dim Latent dimension
   * @param threshold_percentile Threshold percentile
   * @param device Computation device
   * @return AnomalyDetector instance for time series
   */
  static std::unique_ptr<AnomalyDetector>
  create_for_timeseries(int window_size, int num_features = 1,
                        int latent_dim = 0,  // 0 = auto-calculate
                        double threshold_percentile = 95.0,
                        DeviceType device = DeviceType::CPU);

  /**
   * @brief Save anomaly detector with threshold
   * @param base_path Base path for saving
   * @param save_json Save JSON metadata
   * @param save_binary Save binary weights
   */
  void save(const std::string& base_path, bool save_json = true,
            bool save_binary = true);

  /**
   * @brief Load anomaly detector with threshold
   * @param base_path Base path for loading
   * @return True if successful
   */
  bool load(const std::string& base_path);

private:
  AnomalyConfig anomaly_config_;
  double threshold_;
  bool threshold_calculated_;

  /**
   * @brief Calculate threshold using percentile method
   * @param errors Vector of reconstruction errors
   * @return Calculated threshold
   */
  double calculate_percentile_threshold(const std::vector<double>& errors);

  /**
   * @brief Calculate threshold using standard deviation method
   * @param errors Vector of reconstruction errors
   * @return Calculated threshold
   */
  double calculate_std_threshold(const std::vector<double>& errors);

  /**
   * @brief Calculate performance metrics
   * @param results Anomaly results to update
   * @param ground_truth Ground truth labels
   */
  void calculate_performance_metrics(AnomalyResults& results,
                                     const std::vector<bool>& ground_truth);

  /**
   * @brief Update adaptive threshold
   * @param recent_errors Recent reconstruction errors
   */
  void update_adaptive_threshold(const std::vector<double>& recent_errors);
};

}  // namespace autoencoder
}  // namespace model
}  // namespace MLLib
