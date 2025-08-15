#include "MLLib/model/model_io.hpp"
#include "MLLib/layer/activation/relu.hpp"
#include "MLLib/layer/activation/sigmoid.hpp"
#include "MLLib/layer/dense.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/stat.h>

namespace MLLib {
namespace model {

bool ModelIO::save_model(const Sequential& model, const std::string& filepath,
                         ModelFormat format) {
  switch (format) {
  case ModelFormat::BINARY: return save_binary(model, filepath);
  case ModelFormat::JSON: return save_json(model, filepath);
  case ModelFormat::CONFIG: return save_config(model, filepath);
  default: std::cerr << "Unsupported format" << std::endl; return false;
  }
}

std::unique_ptr<Sequential> ModelIO::load_model(const std::string& filepath,
                                                ModelFormat format) {
  switch (format) {
  case ModelFormat::BINARY: return load_binary(filepath);
  case ModelFormat::JSON: return load_json(filepath);
  case ModelFormat::CONFIG: return load_config(filepath);
  default: std::cerr << "Unsupported format" << std::endl; return nullptr;
  }
}

bool ModelIO::save_config(const Sequential& model,
                          const std::string& filepath) {
  // Ensure directory exists
  if (!ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  ModelConfig config = extract_config(model);

  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  file << "# MLLib Model Configuration\n";
  file << "model_type: " << config.model_type << "\n";
  file << "version: " << config.version << "\n";
  file << "device: " << (config.device == DeviceType::CPU ? "CPU" : "GPU")
       << "\n";
  file << "layers:\n";

  for (const auto& layer_info : config.layers) {
    file << "  - type: " << layer_info.type << "\n";
    if (layer_info.type == "Dense") {
      file << "    input_size: " << layer_info.input_size << "\n";
      file << "    output_size: " << layer_info.output_size << "\n";
      file << "    use_bias: " << (layer_info.use_bias ? "true" : "false")
           << "\n";
    }
  }

  file.close();
  return true;
}

std::unique_ptr<Sequential> ModelIO::load_config(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filepath << std::endl;
    return nullptr;
  }

  ModelConfig config;
  std::string line;
  bool in_layers = false;
  LayerInfo current_layer;

  while (std::getline(file, line)) {
    // Skip comments and empty lines
    if (line.empty() || line[0] == '#') continue;

    // Parse key-value pairs
    size_t colon_pos = line.find(':');
    if (colon_pos == std::string::npos) continue;

    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);

    // Trim whitespace
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    if (key == "model_type") {
      config.model_type = value;
    } else if (key == "version") {
      config.version = value;
    } else if (key == "device") {
      config.device = (value == "CPU") ? DeviceType::CPU : DeviceType::GPU;
    } else if (key == "layers") {
      in_layers = true;
    } else if (in_layers && key == "- type") {
      // Save previous layer if exists
      if (!current_layer.type.empty()) {
        config.layers.push_back(current_layer);
      }
      current_layer = LayerInfo(value);
    } else if (in_layers && key == "input_size") {
      current_layer.input_size = std::stoull(value);
    } else if (in_layers && key == "output_size") {
      current_layer.output_size = std::stoull(value);
    } else if (in_layers && key == "use_bias") {
      current_layer.use_bias = (value == "true");
    }
  }

  // Add last layer
  if (!current_layer.type.empty()) {
    config.layers.push_back(current_layer);
  }

  file.close();
  return create_from_config(config);
}

bool ModelIO::save_parameters(const Sequential& model,
                              const std::string& filepath) {
  // Ensure directory exists
  if (!ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  // Write parameter version
  uint32_t param_version = 1;
  write_binary_data(file, &param_version, sizeof(param_version));

  // Write number of layers
  uint32_t num_layers = static_cast<uint32_t>(model.get_layers().size());
  write_binary_data(file, &num_layers, sizeof(num_layers));

  // Write parameters for each layer
  for (const auto& layer : model.get_layers()) {
    auto dense_layer = dynamic_cast<const layer::Dense*>(layer.get());
    if (dense_layer) {
      // Write layer type identifier
      uint32_t layer_type = 1;  // Dense = 1
      write_binary_data(file, &layer_type, sizeof(layer_type));

      // Write weights and biases
      write_ndarray(file, dense_layer->get_weights());
      if (dense_layer->get_use_bias()) {
        write_ndarray(file, dense_layer->get_bias());
      }
    } else {
      // Non-parametric layer
      uint32_t layer_type = 0;  // Non-parametric = 0
      write_binary_data(file, &layer_type, sizeof(layer_type));
    }
  }

  file.close();
  return true;
}

bool ModelIO::load_parameters(Sequential& model, const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filepath << std::endl;
    return false;
  }

  // Read parameter version
  uint32_t param_version;
  if (!read_binary_data(file, &param_version, sizeof(param_version))) {
    std::cerr << "Failed to read parameter version" << std::endl;
    return false;
  }

  if (param_version != 1) {
    std::cerr << "Unsupported parameter version: " << param_version
              << std::endl;
    return false;
  }

  // Read number of layers
  uint32_t num_layers;
  if (!read_binary_data(file, &num_layers, sizeof(num_layers))) {
    std::cerr << "Failed to read number of layers" << std::endl;
    return false;
  }

  if (num_layers != model.get_layers().size()) {
    std::cerr << "Layer count mismatch" << std::endl;
    return false;
  }

  // Load parameters for each layer
  for (size_t i = 0; i < num_layers; ++i) {
    uint32_t layer_type;
    if (!read_binary_data(file, &layer_type, sizeof(layer_type))) {
      std::cerr << "Failed to read layer type" << std::endl;
      return false;
    }

    if (layer_type == 1) {  // Dense layer
      auto dense_layer =
          dynamic_cast<layer::Dense*>(model.get_layers()[i].get());
      if (!dense_layer) {
        std::cerr << "Layer type mismatch at layer " << i << std::endl;
        return false;
      }

      // Load weights
      NDArray weights = read_ndarray(file);
      dense_layer->set_weights(weights);

      // Load biases if the layer uses them
      if (dense_layer->get_use_bias()) {
        NDArray biases = read_ndarray(file);
        dense_layer->set_biases(biases);
      }
    }
    // For non-parametric layers (layer_type == 0), nothing to load
  }

  file.close();
  return true;
}

ModelConfig ModelIO::extract_config(const Sequential& model) {
  ModelConfig config;
  config.device = model.get_device();

  for (const auto& layer : model.get_layers()) {
    auto dense_layer = dynamic_cast<const layer::Dense*>(layer.get());
    if (dense_layer) {
      LayerInfo layer_info("Dense", dense_layer->get_input_size(),
                           dense_layer->get_output_size(),
                           dense_layer->get_use_bias());
      config.layers.push_back(layer_info);
    } else if (dynamic_cast<const layer::activation::ReLU*>(layer.get())) {
      config.layers.push_back(LayerInfo("ReLU"));
    } else if (dynamic_cast<const layer::activation::Sigmoid*>(layer.get())) {
      config.layers.push_back(LayerInfo("Sigmoid"));
    }
  }

  return config;
}

std::unique_ptr<Sequential>
ModelIO::create_from_config(const ModelConfig& config) {
  auto model = std::make_unique<Sequential>(config.device);

  for (const auto& layer_info : config.layers) {
    if (layer_info.type == "Dense") {
      model->add(std::make_shared<layer::Dense>(
          layer_info.input_size, layer_info.output_size, layer_info.use_bias));
    } else if (layer_info.type == "ReLU") {
      model->add(std::make_shared<layer::activation::ReLU>());
    } else if (layer_info.type == "Sigmoid") {
      model->add(std::make_shared<layer::activation::Sigmoid>());
    }
  }

  return model;
}

bool ModelIO::save_binary(const Sequential& model,
                          const std::string& filepath) {
  // Ensure directory exists
  if (!ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  // Write magic number and version
  uint32_t magic = 0x4D4C4C42;  // "MLLB" (ML Lib Binary)
  uint32_t version = 1;
  write_binary_data(file, &magic, sizeof(magic));
  write_binary_data(file, &version, sizeof(version));

  // Write device type
  uint32_t device_type = static_cast<uint32_t>(model.get_device());
  write_binary_data(file, &device_type, sizeof(device_type));

  // Write model configuration
  ModelConfig config = extract_config(model);
  uint32_t num_layers = static_cast<uint32_t>(config.layers.size());
  write_binary_data(file, &num_layers, sizeof(num_layers));

  for (const auto& layer_info : config.layers) {
    // Write layer type
    uint32_t type_len = static_cast<uint32_t>(layer_info.type.length());
    write_binary_data(file, &type_len, sizeof(type_len));
    write_binary_data(file, layer_info.type.c_str(), type_len);

    if (layer_info.type == "Dense") {
      write_binary_data(file, &layer_info.input_size,
                        sizeof(layer_info.input_size));
      write_binary_data(file, &layer_info.output_size,
                        sizeof(layer_info.output_size));
      write_binary_data(file, &layer_info.use_bias,
                        sizeof(layer_info.use_bias));
    }
  }

  // Write parameters
  for (const auto& layer : model.get_layers()) {
    auto dense_layer = dynamic_cast<const layer::Dense*>(layer.get());
    if (dense_layer) {
      write_ndarray(file, dense_layer->get_weights());
      if (dense_layer->get_use_bias()) {
        write_ndarray(file, dense_layer->get_bias());
      }
    }
  }

  file.close();
  return true;
}

std::unique_ptr<Sequential> ModelIO::load_binary(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filepath << std::endl;
    return nullptr;
  }

  // Read and verify magic number
  uint32_t magic;
  if (!read_binary_data(file, &magic, sizeof(magic)) || magic != 0x4D4C4C42) {
    std::cerr << "Invalid file format" << std::endl;
    return nullptr;
  }

  // Read version
  uint32_t version;
  if (!read_binary_data(file, &version, sizeof(version)) || version != 1) {
    std::cerr << "Unsupported file version" << std::endl;
    return nullptr;
  }

  // Read device type
  uint32_t device_type;
  if (!read_binary_data(file, &device_type, sizeof(device_type))) {
    std::cerr << "Failed to read device type" << std::endl;
    return nullptr;
  }

  DeviceType device = static_cast<DeviceType>(device_type);
  auto model = std::make_unique<Sequential>(device);

  // Read model configuration
  uint32_t num_layers;
  if (!read_binary_data(file, &num_layers, sizeof(num_layers))) {
    std::cerr << "Failed to read number of layers" << std::endl;
    return nullptr;
  }

  std::vector<LayerInfo> layers_info;
  for (uint32_t i = 0; i < num_layers; ++i) {
    uint32_t type_len;
    if (!read_binary_data(file, &type_len, sizeof(type_len))) {
      std::cerr << "Failed to read layer type length" << std::endl;
      return nullptr;
    }

    std::string type(type_len, '\0');
    if (!read_binary_data(file, &type[0], type_len)) {
      std::cerr << "Failed to read layer type" << std::endl;
      return nullptr;
    }

    LayerInfo layer_info(type);
    if (type == "Dense") {
      if (!read_binary_data(file, &layer_info.input_size,
                            sizeof(layer_info.input_size)) ||
          !read_binary_data(file, &layer_info.output_size,
                            sizeof(layer_info.output_size)) ||
          !read_binary_data(file, &layer_info.use_bias,
                            sizeof(layer_info.use_bias))) {
        std::cerr << "Failed to read Dense layer configuration" << std::endl;
        return nullptr;
      }
    }

    layers_info.push_back(layer_info);
  }

  // Create layers
  for (const auto& layer_info : layers_info) {
    if (layer_info.type == "Dense") {
      model->add(std::make_shared<layer::Dense>(
          layer_info.input_size, layer_info.output_size, layer_info.use_bias));
    } else if (layer_info.type == "ReLU") {
      model->add(std::make_shared<layer::activation::ReLU>());
    } else if (layer_info.type == "Sigmoid") {
      model->add(std::make_shared<layer::activation::Sigmoid>());
    }
  }

  // Load parameters
  for (size_t i = 0; i < layers_info.size(); ++i) {
    const auto& layer_info = layers_info[i];
    if (layer_info.type == "Dense") {
      auto dense_layer =
          dynamic_cast<layer::Dense*>(model->get_layers()[i].get());

      NDArray weights = read_ndarray(file);
      dense_layer->set_weights(weights);

      if (layer_info.use_bias) {
        NDArray biases = read_ndarray(file);
        dense_layer->set_biases(biases);
      }
    }
  }

  file.close();
  return model;
}

bool ModelIO::save_json(const Sequential& model, const std::string& filepath) {
  // Ensure directory exists
  if (!ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  ModelConfig config = extract_config(model);

  file << "{\n";
  file << "  \"model_type\": \"" << config.model_type << "\",\n";
  file << "  \"version\": \"" << config.version << "\",\n";
  file << "  \"device\": \""
       << (config.device == DeviceType::CPU ? "CPU" : "GPU") << "\",\n";
  file << "  \"layers\": [\n";

  for (size_t i = 0; i < config.layers.size(); ++i) {
    const auto& layer_info = config.layers[i];
    file << "    {\n";
    file << "      \"type\": \"" << layer_info.type << "\"";

    if (layer_info.type == "Dense") {
      file << ",\n";
      file << "      \"input_size\": " << layer_info.input_size << ",\n";
      file << "      \"output_size\": " << layer_info.output_size << ",\n";
      file << "      \"use_bias\": "
           << (layer_info.use_bias ? "true" : "false");
    }

    file << "\n    }";
    if (i < config.layers.size() - 1) file << ",";
    file << "\n";
  }

  file << "  ],\n";
  file << "  \"parameters\": {\n";

  bool first_param = true;
  for (size_t i = 0; i < model.get_layers().size(); ++i) {
    auto dense_layer =
        dynamic_cast<const layer::Dense*>(model.get_layers()[i].get());
    if (dense_layer) {
      if (!first_param) file << ",\n";
      first_param = false;

      file << "    \"layer_" << i << "\": {\n";

      // Save weights
      const auto& weights = dense_layer->get_weights();
      file << "      \"weights\": {\n";
      file << "        \"shape\": [" << weights.shape()[0] << ", "
           << weights.shape()[1] << "],\n";
      file << "        \"data\": [";

      for (size_t j = 0; j < weights.size(); ++j) {
        if (j > 0) file << ", ";
        file << weights.data()[j];
      }

      file << "]\n      }";

      // Save biases if present
      if (dense_layer->get_use_bias()) {
        const auto& biases = dense_layer->get_bias();
        file << ",\n      \"biases\": {\n";
        file << "        \"shape\": [" << biases.shape()[0] << "],\n";
        file << "        \"data\": [";

        for (size_t j = 0; j < biases.size(); ++j) {
          if (j > 0) file << ", ";
          file << biases.data()[j];
        }

        file << "]\n      }";
      }

      file << "\n    }";
    }
  }

  file << "\n  }\n";
  file << "}\n";

  file.close();
  return true;
}

std::unique_ptr<Sequential> ModelIO::load_json(const std::string& filepath) {
  // Note: This is a simplified JSON parser for demonstration
  // In a real implementation, you would use a proper JSON library
  (void)filepath;  // Suppress unused parameter warning
  std::cerr << "JSON loading not fully implemented - use binary format instead"
            << std::endl;
  return nullptr;
}

void ModelIO::write_binary_data(std::ofstream& file, const void* data,
                                size_t size) {
  file.write(static_cast<const char*>(data), size);
}

bool ModelIO::read_binary_data(std::ifstream& file, void* data, size_t size) {
  file.read(static_cast<char*>(data), size);
  return file.gcount() == static_cast<std::streamsize>(size);
}

void ModelIO::write_ndarray(std::ofstream& file, const NDArray& array) {
  // Write dimension count
  uint32_t ndim = static_cast<uint32_t>(array.shape().size());
  write_binary_data(file, &ndim, sizeof(ndim));

  // Write shape
  for (size_t i = 0; i < array.shape().size(); ++i) {
    uint32_t dim_size = static_cast<uint32_t>(array.shape()[i]);
    write_binary_data(file, &dim_size, sizeof(dim_size));
  }

  // Write data
  size_t data_size = array.size() * sizeof(float);
  write_binary_data(file, array.data(), data_size);
}

NDArray ModelIO::read_ndarray(std::ifstream& file) {
  // Read dimension count
  uint32_t ndim;
  read_binary_data(file, &ndim, sizeof(ndim));

  // Read shape
  std::vector<size_t> shape(ndim);
  for (uint32_t i = 0; i < ndim; ++i) {
    uint32_t dim_size;
    read_binary_data(file, &dim_size, sizeof(dim_size));
    shape[i] = static_cast<size_t>(dim_size);
  }

  // Create array and read data
  NDArray array(shape);
  size_t data_size = array.size() * sizeof(float);
  read_binary_data(file, array.data(), data_size);

  return array;
}

bool ModelIO::create_directories(const std::string& path) {
  if (path.empty()) {
    return true;
  }

  try {
// Use C++17 filesystem if available
#if __cplusplus >= 201703L && defined(__has_include)
#if __has_include(<filesystem>)
    return std::filesystem::create_directories(path);
#endif
#endif

// Fallback to platform-specific implementation
#ifdef _WIN32
    // Windows implementation
    std::string command = "mkdir \"" + path + "\" 2>nul";
    return system(command.c_str()) == 0 || directory_exists(path);
#else
    // Unix/Linux/macOS implementation
    std::string command = "mkdir -p \"" + path + "\"";
    return system(command.c_str()) == 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "Failed to create directory: " << e.what() << std::endl;
    return false;
  }
}

std::string ModelIO::get_directory_path(const std::string& filepath) {
  size_t last_slash = filepath.find_last_of("/\\");
  if (last_slash == std::string::npos) {
    return "";  // No directory component
  }
  return filepath.substr(0, last_slash);
}

bool ModelIO::directory_exists(const std::string& path) {
  if (path.empty()) {
    return true;
  }

  try {
#if __cplusplus >= 201703L && defined(__has_include)
#if __has_include(<filesystem>)
    return std::filesystem::exists(path) && std::filesystem::is_directory(path);
#endif
#endif

    // Fallback to stat
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
      return false;  // Path doesn't exist
    }
    return (info.st_mode & S_IFDIR) != 0;
  } catch (const std::exception&) {
    return false;
  }
}

bool ModelIO::ensure_directory_exists(const std::string& filepath) {
  std::string dir_path = get_directory_path(filepath);
  if (dir_path.empty()) {
    return true;  // No directory to create
  }

  if (directory_exists(dir_path)) {
    return true;  // Directory already exists
  }

  return create_directories(dir_path);
}

ModelFormat ModelIO::string_to_format(const std::string& format_str) {
  if (format_str == "binary") {
    return ModelFormat::BINARY;
  } else if (format_str == "json") {
    return ModelFormat::JSON;
  } else if (format_str == "config") {
    return ModelFormat::CONFIG;
  } else {
    // Default to binary for unknown formats
    std::cerr << "Unknown format '" << format_str << "', defaulting to binary"
              << std::endl;
    return ModelFormat::BINARY;
  }
}

std::string ModelIO::format_to_string(ModelFormat format) {
  switch (format) {
  case ModelFormat::BINARY: return "binary";
  case ModelFormat::JSON: return "json";
  case ModelFormat::CONFIG: return "config";
  default: return "binary";
  }
}

}  // namespace model
}  // namespace MLLib
