# Autoencoder Examples

This directory contains comprehensive examples demonstrating the MLLib Autoencoder Library. All examples use the new library-based API for easy integration and high performance.

## Library-Based Examples

### 1. Basic Dense Autoencoder (`01_basic_dense_autoencoder.cpp`)
- **Purpose**: Introduction to basic autoencoder concepts
- **Dataset**: Synthetic 2D circle data
- **Features**:
  - Simple dense architecture (2→4→2→4→2)
  - Basic reconstruction and training loop
  - Performance evaluation
  - Model saving/loading

**Key Learning Points**:
- Autoencoder configuration setup
- Training loop implementation
- Reconstruction error calculation
- Model persistence

### 2. Denoising Autoencoder (`02_denoising_autoencoder.cpp`)
- **Purpose**: Noise removal and data cleaning
- **Dataset**: Synthetic 28x28 grayscale images with geometric patterns
- **Features**:
  - Gaussian noise injection
  - PSNR/SSIM evaluation metrics
  - Multiple noise type testing
  - Image restoration pipeline

**Key Learning Points**:
- Noisy input with clean target training
- Image quality metrics
- Robustness evaluation
- Factory method usage

### 3. Variational Autoencoder (`03_variational_autoencoder.cpp`)
- **Purpose**: Generative modeling and latent space exploration
- **Dataset**: Multi-cluster 2D data
- **Features**:
  - KL divergence regularization
  - Latent space sampling
  - Interpolation between points
  - Beta-VAE parameter control

**Key Learning Points**:
- Probabilistic latent representations
- Generative sample creation
- Latent space interpolation
- KL divergence analysis

### 4. Anomaly Detection (`04_anomaly_detection.cpp`)
- **Purpose**: Unsupervised outlier detection
- **Dataset**: Sensor data (temperature, pressure, vibration)
- **Features**:
  - Training on normal data only
  - Threshold calculation methods
  - Confusion matrix evaluation
  - ROC analysis

**Key Learning Points**:
- Anomaly detection workflow
- Threshold selection strategies
- Performance metrics (precision, recall, F1)
- Real-world sensor applications

### 5. Autoencoder Comparison (`05_autoencoder_comparison.cpp`)
- **Purpose**: Comprehensive comparison and model selection
- **Dataset**: Benchmark data for fair comparison
- **Features**:
  - Side-by-side performance evaluation
  - Memory and speed benchmarking
  - Use case recommendations
  - Model selection guidance

**Key Learning Points**:
- When to use each autoencoder type
- Performance trade-offs
- Application-specific recommendations
- Benchmarking methodology

### 6. Model I/O Example (`06_model_io_example.cpp`) 🆕
- **Purpose**: Demonstrate the new generic model serialization system
- **Dataset**: Synthetic pattern data
- **Features**:
  - GenericModelIO save/load operations
  - Multiple save formats (BINARY, JSON, CONFIG)
  - Model consistency validation
  - Deployment scenario simulation
  - Serialization metadata handling

**Key Learning Points**:
- Using the new GenericModelIO architecture
- Model persistence for production deployment
- Configuration export/import
- Cross-format compatibility
- Production workflow best practices

## Building and Running

### Prerequisites
- C++17 compatible compiler
- MLLib library built (`make all`)

### Build All Examples
```bash
# From the project root
make samples
```

### Run Individual Examples
```bash
# From build/samples/autoencoder/
./01_basic_dense_autoencoder
./02_denoising_autoencoder
./03_variational_autoencoder
./04_anomaly_detection
./05_autoencoder_comparison
./06_model_io_example          # New model I/O demonstration
```

## Expected Outputs

Each example provides detailed console output including:
- Dataset generation and statistics
- Training progress and metrics
- Model evaluation results
- Performance summaries
- Recommendations and insights

## MLLib Autoencoder Library API

### Core Classes
- `BaseAutoencoder`: Abstract base class for all autoencoders
- `DenseAutoencoder`: Standard fully-connected autoencoder
- `DenoisingAutoencoder`: Noise-resistant autoencoder
- `VariationalAutoencoder`: Probabilistic generative autoencoder
- `AnomalyDetector`: Specialized for outlier detection

### Factory Methods
```cpp
// Dense autoencoder with simple architecture
auto ae = DenseAutoencoder::create_simple(input_size, latent_size, learning_rate);

// Denoising autoencoder for images
auto denoising = DenoisingAutoencoder::create_for_images(784, 128, NoiseType::GAUSSIAN, 0.2);

// Anomaly detector for sensors
auto detector = AnomalyDetector::create_for_sensors(num_sensors, hidden_size);
```

### Configuration
```cpp
AutoencoderConfig config;
config.input_size = 784;
config.hidden_sizes = {256, 128, 64, 128, 256};
config.learning_rate = 0.001;
config.device = DeviceType::CPU;  // or GPU

auto autoencoder = std::make_unique<DenseAutoencoder>(config);
```

### Model I/O with GenericModelIO 🆕
```cpp
using namespace MLLib::model;

// Save in different formats
GenericModelIO::save_model(*autoencoder, "model.bin", SaveFormat::BINARY);
GenericModelIO::save_model(*autoencoder, "model.json", SaveFormat::JSON);
GenericModelIO::save_model(*autoencoder, "model.cfg", SaveFormat::CONFIG);

// Load model (when fully implemented)
auto loaded_model = GenericModelIO::load_model<DenseAutoencoder>("model.bin", SaveFormat::BINARY);

// Get model metadata
auto metadata = autoencoder->get_serialization_metadata();
std::cout << "Model type: " << model_type_to_string(metadata.model_type) << std::endl;

// Export configuration
std::string config_str = autoencoder->get_config_string();
// Use config_str to recreate model architecture
```

## Performance Tips

1. **Data Preprocessing**: Normalize input data for stable training
2. **Architecture**: Start with symmetric encoder-decoder designs
3. **Learning Rate**: Use adaptive optimizers (Adam) for better convergence
4. **Regularization**: Apply dropout or weight decay to prevent overfitting
5. **Validation**: Always use separate validation data for hyperparameter tuning

## Common Use Cases

| Use Case | Recommended Type | Key Benefits |
|----------|-----------------|--------------|
| Dimensionality Reduction | Dense | Simple, interpretable |
| Data Cleaning | Denoising | Noise robustness |
| Sample Generation | VAE | Controllable latent space |
| Fault Detection | Anomaly Detector | Optimized thresholds |
| Feature Learning | Dense/Denoising | Unsupervised representation |

## Advanced Topics

For more advanced usage, see:
- Custom loss functions with `BaseAutoencoder::set_loss_function()`
- GPU acceleration with `DeviceType::CUDA`
- Model ensemble techniques
- Transfer learning approaches

## Troubleshooting

**Common Issues**:
- **NaN Loss**: Check data normalization and learning rate
- **Poor Reconstruction**: Increase model capacity or reduce bottleneck
- **Overfitting**: Add regularization or reduce model complexity
- **Slow Training**: Use GPU acceleration or reduce batch size

For more help, see the main MLLib documentation.
