#include "MLLib/model/model_io.hpp"
#include "MLLib/layer/activation/relu.hpp"
#include "MLLib/layer/activation/sigmoid.hpp"
#include "MLLib/layer/activation/tanh.hpp"
#include "MLLib/layer/dense.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/stat.h>

namespace MLLib {
namespace model {

// Utility functions
std::string model_type_to_string(ModelType type) {
  switch (type) {
  case ModelType::SEQUENTIAL: return "Sequential";
  case ModelType::AUTOENCODER_DENSE: return "DenseAutoencoder";
  default: return "Unknown";
  }
}

ModelType string_to_model_type(const std::string& type_str) {
  if (type_str == "Sequential") return ModelType::SEQUENTIAL;
  if (type_str == "DenseAutoencoder") return ModelType::AUTOENCODER_DENSE;
  return ModelType::SEQUENTIAL;  // Default
}

// Generic model I/O implementation
bool GenericModelIO::save_model(const ISerializableModel& model,
                                const std::string& filepath,
                                SaveFormat format) {
  std::string actual_filepath = get_filepath_with_extension(filepath, format);

  switch (format) {
  case SaveFormat::BINARY: return save_binary(model, actual_filepath);
  case SaveFormat::JSON: return save_json(model, actual_filepath);
  case SaveFormat::CONFIG: return save_config(model, actual_filepath);
  default: std::cerr << "Unsupported format" << std::endl; return false;
  }
}

std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
GenericModelIO::load_model_data(const std::string& filepath,
                                SaveFormat format) {
  std::string actual_filepath = get_filepath_with_extension(filepath, format);

  switch (format) {
  case SaveFormat::BINARY: return load_binary(actual_filepath);
  case SaveFormat::JSON: return load_json(actual_filepath);
  case SaveFormat::CONFIG:
    // Config only format doesn't contain parameter data
    return std::make_unique<
        std::unordered_map<std::string, std::vector<uint8_t>>>();
  default: std::cerr << "Unsupported format" << std::endl; return nullptr;
  }
}

bool GenericModelIO::save_config(const ISerializableModel& model,
                                 const std::string& filepath) {
  if (!GenericModelIO::ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  auto metadata = model.get_serialization_metadata();
  auto data = model.serialize();

  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  file << "# MLLib Model Configuration\n";
  file << "model_type: " << static_cast<int>(metadata.model_type) << "\n";
  file << "version: " << metadata.version << "\n";
  file << "device: " << (metadata.device == DeviceType::CPU ? "CPU" : "GPU")
       << "\n";

  // Write model-specific configuration
  for (const auto& [key, value] : data) {
    // Skip parameter data in config-only mode
    if (key.find("parameters") == std::string::npos) {
      file << key << ": ";
      // Simple text representation (could be enhanced for complex data)
      for (size_t i = 0; i < value.size() && i < 100; ++i) {
        file << static_cast<int>(value[i]);
        if (i < value.size() - 1 && i < 99) file << ",";
      }
      file << "\n";
    }
  }

  file.close();
  return true;
}

std::unique_ptr<SerializationMetadata>
GenericModelIO::load_metadata(const std::string& filepath) {
  // Try to determine format from extension and load appropriately
  std::string actual_filepath = filepath;

  // If no extension, try to find which file exists
  if (filepath.find('.') == std::string::npos) {
    if (std::ifstream(filepath + ".config").good()) {
      actual_filepath = filepath + ".config";
    } else if (std::ifstream(filepath + ".json").good()) {
      actual_filepath = filepath + ".json";
    } else if (std::ifstream(filepath + ".bin").good()) {
      actual_filepath = filepath + ".bin";
    }
  }

  std::ifstream file(actual_filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << actual_filepath
              << std::endl;
    return nullptr;
  }

  auto metadata = std::make_unique<SerializationMetadata>();
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

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
      metadata->model_type = static_cast<ModelType>(std::stoi(value));
    } else if (key == "version") {
      metadata->version = value;
    } else if (key == "device") {
      metadata->device = (value == "CPU") ? DeviceType::CPU : DeviceType::GPU;
    }
  }

  file.close();
  return metadata;
}

bool GenericModelIO::save_binary(const ISerializableModel& model,
                                 const std::string& filepath) {
  if (!GenericModelIO::ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  // Write magic number and version
  uint32_t magic = 0x4D4C4C47;  // "MLLG" (ML Lib Generic)
  uint32_t version = 1;
  file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));

  // Write metadata
  auto metadata = model.get_serialization_metadata();
  uint32_t model_type = static_cast<uint32_t>(metadata.model_type);
  uint32_t device_type = static_cast<uint32_t>(metadata.device);
  file.write(reinterpret_cast<const char*>(&model_type), sizeof(model_type));
  file.write(reinterpret_cast<const char*>(&device_type), sizeof(device_type));

  // Write version string
  uint32_t version_len = static_cast<uint32_t>(metadata.version.length());
  file.write(reinterpret_cast<const char*>(&version_len), sizeof(version_len));
  file.write(metadata.version.c_str(), version_len);

  // Write serialized data
  auto data = model.serialize();
  uint32_t data_count = static_cast<uint32_t>(data.size());
  file.write(reinterpret_cast<const char*>(&data_count), sizeof(data_count));

  for (const auto& [key, value] : data) {
    // Write key
    uint32_t key_len = static_cast<uint32_t>(key.length());
    file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
    file.write(key.c_str(), key_len);

    // Write value
    uint32_t value_len = static_cast<uint32_t>(value.size());
    file.write(reinterpret_cast<const char*>(&value_len), sizeof(value_len));
    file.write(reinterpret_cast<const char*>(value.data()), value_len);
  }

  file.close();
  return true;
}

bool GenericModelIO::save_json(const ISerializableModel& model,
                               const std::string& filepath) {
  if (!GenericModelIO::ensure_directory_exists(filepath)) {
    std::cerr << "Failed to create directory for: " << filepath << std::endl;
    return false;
  }

  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filepath << std::endl;
    return false;
  }

  auto metadata = model.get_serialization_metadata();
  auto data = model.serialize();

  file << "{\n";
  file << "  \"model_type\": " << static_cast<int>(metadata.model_type)
       << ",\n";
  file << "  \"version\": \"" << metadata.version << "\",\n";
  file << "  \"device\": \""
       << (metadata.device == DeviceType::CPU ? "CPU" : "GPU") << "\",\n";
  file << "  \"data\": {\n";

  bool first = true;
  for (const auto& [key, value] : data) {
    if (!first) file << ",\n";
    first = false;

    file << "    \"" << key << "\": [";
    for (size_t i = 0; i < value.size(); ++i) {
      if (i > 0) file << ", ";
      file << static_cast<int>(value[i]);
    }
    file << "]";
  }

  file << "\n  }\n";
  file << "}\n";

  file.close();
  return true;
}

std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
GenericModelIO::load_binary(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filepath << std::endl;
    return nullptr;
  }

  // Read and verify magic number
  uint32_t magic;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic != 0x4D4C4C47) {
    std::cerr << "Invalid generic model file format" << std::endl;
    return nullptr;
  }

  // Read version
  uint32_t version;
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != 1) {
    std::cerr << "Unsupported file version" << std::endl;
    return nullptr;
  }

  // Skip metadata for now (model type, device type, version string)
  uint32_t model_type, device_type, version_len;
  file.read(reinterpret_cast<char*>(&model_type), sizeof(model_type));
  file.read(reinterpret_cast<char*>(&device_type), sizeof(device_type));
  file.read(reinterpret_cast<char*>(&version_len), sizeof(version_len));
  file.seekg(version_len, std::ios::cur);  // Skip version string

  // Read serialized data
  uint32_t data_count;
  file.read(reinterpret_cast<char*>(&data_count), sizeof(data_count));

  auto data =
      std::make_unique<std::unordered_map<std::string, std::vector<uint8_t>>>();
  for (uint32_t i = 0; i < data_count; ++i) {
    // Read key
    uint32_t key_len;
    file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
    std::string key(key_len, '\0');
    file.read(&key[0], key_len);

    // Read value
    uint32_t value_len;
    file.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
    std::vector<uint8_t> value(value_len);
    file.read(reinterpret_cast<char*>(value.data()), value_len);

    data->emplace(key, std::move(value));
  }

  file.close();
  return data;
}

std::unique_ptr<std::unordered_map<std::string, std::vector<uint8_t>>>
GenericModelIO::load_json(const std::string& filepath) {
  // Simplified JSON parser placeholder
  (void)filepath;
  std::cerr
      << "Generic JSON loading not implemented - use binary format instead"
      << std::endl;
  return nullptr;
}

bool GenericModelIO::ensure_directory_exists(const std::string& filepath) {
  // Extract directory path
  size_t pos = filepath.find_last_of("/\\");
  if (pos == std::string::npos) {
    return true;  // No directory in path
  }

  std::string dir_path = filepath.substr(0, pos);

  // Check if directory exists
  struct stat sb;
  if (stat(dir_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
    return true;
  }

  // Create directory recursively
  return create_directories(dir_path);
}

bool GenericModelIO::create_directories(const std::string& path) {
  if (path.empty()) return true;

  // Check if already exists
  struct stat sb;
  if (stat(path.c_str(), &sb) == 0) {
    return S_ISDIR(sb.st_mode);
  }

  // Create parent first
  size_t pos = path.find_last_of("/\\");
  if (pos != std::string::npos) {
    std::string parent = path.substr(0, pos);
    if (!create_directories(parent)) {
      return false;
    }
  }

  // Create this directory
  return mkdir(path.c_str(), 0755) == 0;
}

std::string
GenericModelIO::get_filepath_with_extension(const std::string& base_filepath,
                                            SaveFormat format) {
  // Check if filepath already has an appropriate extension
  std::string extension;
  switch (format) {
  case SaveFormat::BINARY: extension = ".bin"; break;
  case SaveFormat::JSON: extension = ".json"; break;
  case SaveFormat::CONFIG: extension = ".config"; break;
  default:
    extension = ".bin";  // Default fallback
    break;
  }

  // If the filepath already ends with the correct extension, return as is
  if (base_filepath.length() >= extension.length() &&
      base_filepath.substr(base_filepath.length() - extension.length()) ==
          extension) {
    return base_filepath;
  }

  // Remove any existing extension-like suffix
  std::string clean_filepath = base_filepath;
  size_t last_dot = clean_filepath.find_last_of('.');
  size_t last_slash = clean_filepath.find_last_of("/\\");

  // Only remove extension if the dot is after the last slash (i.e., in
  // filename, not directory)
  if (last_dot != std::string::npos &&
      (last_slash == std::string::npos || last_dot > last_slash)) {
    clean_filepath = clean_filepath.substr(0, last_dot);
  }

  return clean_filepath + extension;
}

// Simple helper functions to replace missing ModelIO methods
static void write_binary_data(std::ofstream& file, const void* data,
                              size_t size) {
  file.write(reinterpret_cast<const char*>(data), size);
}

static bool read_binary_data(std::ifstream& file, void* data, size_t size) {
  file.read(reinterpret_cast<char*>(data), size);
  return file.good();
}

static void write_ndarray(std::ofstream& file, const NDArray& array) {
  // Simplified NDArray writing - TODO: implement proper serialization
  const auto& shape = array.shape();
  size_t ndim = shape.size();
  write_binary_data(file, &ndim, sizeof(ndim));

  for (size_t dim : shape) {
    write_binary_data(file, &dim, sizeof(dim));
  }

  size_t data_size = array.size() * sizeof(float);
  write_binary_data(file, array.data(), data_size);
}

static NDArray read_ndarray(std::ifstream& file) {
  size_t ndim;
  read_binary_data(file, &ndim, sizeof(ndim));

  std::vector<size_t> shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    read_binary_data(file, &shape[i], sizeof(size_t));
  }

  NDArray result(shape);
  size_t data_size = result.size() * sizeof(float);
  read_binary_data(file,
                   const_cast<void*>(static_cast<const void*>(result.data())),
                   data_size);
  return result;
}

// Legacy Sequential model I/O implementation
bool ModelIO::save_model(const Sequential& model, const std::string& filepath,
                         SaveFormat format) {
  std::string actual_filepath = get_filepath_with_extension(filepath, format);

  switch (format) {
  case SaveFormat::BINARY: return save_binary(model, actual_filepath);
  case SaveFormat::JSON: return save_json(model, actual_filepath);
  case SaveFormat::CONFIG: return save_config(model, actual_filepath);
  default: std::cerr << "Unsupported format" << std::endl; return false;
  }
}

std::unique_ptr<Sequential> ModelIO::load_model(const std::string& filepath,
                                                SaveFormat format) {
  std::string actual_filepath = get_filepath_with_extension(filepath, format);

  switch (format) {
  case SaveFormat::BINARY: return load_binary(actual_filepath);
  case SaveFormat::JSON: return load_json(actual_filepath);
  case SaveFormat::CONFIG: return load_config(actual_filepath);
  default: std::cerr << "Unsupported format" << std::endl; return nullptr;
  }
}

bool ModelIO::save_config(const Sequential& model,
                          const std::string& filepath) {
  // Ensure directory exists using ModelIO's own method
  if (!ModelIO::ensure_directory_exists(filepath)) {
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

ModelConfig ModelIO::extract_config(const Sequential& model) {
  ModelConfig config;
  config.device = model.get_device();

  for (const auto& layer : model.get_layers()) {
    auto dense_layer =
        std::dynamic_pointer_cast<const MLLib::layer::Dense>(layer);
    if (dense_layer) {
      LayerInfo layer_info("Dense", dense_layer->get_input_size(),
                           dense_layer->get_output_size(),
                           dense_layer->get_use_bias());
      config.layers.push_back(layer_info);
    } else if (std::dynamic_pointer_cast<const MLLib::layer::activation::ReLU>(
                   layer)) {
      config.layers.push_back(LayerInfo("ReLU"));
    } else if (std::dynamic_pointer_cast<
                   const MLLib::layer::activation::Sigmoid>(layer)) {
      config.layers.push_back(LayerInfo("Sigmoid"));
    } else if (std::dynamic_pointer_cast<const MLLib::layer::activation::Tanh>(
                   layer)) {
      config.layers.push_back(LayerInfo("Tanh"));
    }
  }

  return config;
}

std::unique_ptr<Sequential>
ModelIO::create_from_config(const ModelConfig& config) {
  auto model = std::make_unique<Sequential>(config.device);

  for (const auto& layer_info : config.layers) {
    if (layer_info.type == "Dense") {
      model->add(std::make_shared<layer::Dense>(layer_info.input_size,
                                                layer_info.output_size,
                                                layer_info.use_bias));
    } else if (layer_info.type == "ReLU") {
      model->add(std::make_shared<layer::activation::ReLU>());
    } else if (layer_info.type == "Sigmoid") {
      model->add(std::make_shared<layer::activation::Sigmoid>());
    } else if (layer_info.type == "Tanh") {
      model->add(std::make_shared<layer::activation::Tanh>());
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

  try {
    // Write magic number and version
    uint32_t magic = 0x4D4C4C42;  // "MLLB" in little endian
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write device type
    uint32_t device_type = static_cast<uint32_t>(model.get_device());
    file.write(reinterpret_cast<const char*>(&device_type),
               sizeof(device_type));

    // Write number of layers
    uint32_t num_layers = model.get_layers().size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    // Write layer information (simplified approach)
    for (const auto& layer : model.get_layers()) {
      // Try to identify layer type through dynamic casting
      auto dense_layer = std::dynamic_pointer_cast<MLLib::layer::Dense>(layer);
      if (dense_layer) {
        // Write layer type identifier
        std::string layer_type = "Dense";
        uint32_t type_length = layer_type.length();
        file.write(reinterpret_cast<const char*>(&type_length),
                   sizeof(type_length));
        file.write(layer_type.c_str(), type_length);

        // Write layer dimensions
        uint32_t input_size =
            static_cast<uint32_t>(dense_layer->get_input_size());
        uint32_t output_size =
            static_cast<uint32_t>(dense_layer->get_output_size());
        file.write(reinterpret_cast<const char*>(&input_size),
                   sizeof(input_size));
        file.write(reinterpret_cast<const char*>(&output_size),
                   sizeof(output_size));
      } else {
        // For other layer types, write a generic identifier
        std::string layer_type = "Unknown";
        uint32_t type_length = layer_type.length();
        file.write(reinterpret_cast<const char*>(&type_length),
                   sizeof(type_length));
        file.write(layer_type.c_str(), type_length);
      }
    }

    file.close();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error saving binary model: " << e.what() << std::endl;
    return false;
  }
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
      model->add(std::make_shared<layer::Dense>(layer_info.input_size,
                                                layer_info.output_size,
                                                layer_info.use_bias));
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
  // Ensure directory exists using ModelIO's own method
  if (!ModelIO::ensure_directory_exists(filepath)) {
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
  // Legacy function temporarily disabled
  (void)file;
  (void)data;
  (void)size;
}

bool ModelIO::read_binary_data(std::ifstream& file, void* data, size_t size) {
  // Legacy function temporarily disabled
  (void)file;
  (void)data;
  (void)size;
  return false;
}

void ModelIO::write_ndarray(std::ofstream& file, const NDArray& array) {
  // Legacy function temporarily disabled
  (void)file;
  (void)array;
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

SaveFormat ModelIO::string_to_format(const std::string& format_str) {
  if (format_str == "binary") {
    return SaveFormat::BINARY;
  } else if (format_str == "json") {
    return SaveFormat::JSON;
  } else if (format_str == "config") {
    return SaveFormat::CONFIG;
  } else {
    // Default to binary for unknown formats
    std::cerr << "Unknown format '" << format_str << "', defaulting to binary"
              << std::endl;
    return SaveFormat::BINARY;
  }
}

std::string ModelIO::format_to_string(SaveFormat format) {
  switch (format) {
  case SaveFormat::BINARY: return "binary";
  case SaveFormat::JSON: return "json";
  case SaveFormat::CONFIG: return "config";
  default: return "binary";
  }
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

std::string
ModelIO::get_filepath_with_extension(const std::string& base_filepath,
                                     SaveFormat format) {
  // Check if filepath already has an appropriate extension
  std::string extension;
  switch (format) {
  case SaveFormat::BINARY: extension = ".bin"; break;
  case SaveFormat::JSON: extension = ".json"; break;
  case SaveFormat::CONFIG: extension = ".config"; break;
  default:
    extension = ".bin";  // Default fallback
    break;
  }

  // If the filepath already ends with the correct extension, return as is
  if (base_filepath.length() >= extension.length() &&
      base_filepath.substr(base_filepath.length() - extension.length()) ==
          extension) {
    return base_filepath;
  }

  // Remove any existing extension-like suffix
  std::string clean_filepath = base_filepath;
  size_t last_dot = clean_filepath.find_last_of('.');
  size_t last_slash = clean_filepath.find_last_of("/\\");

  // Only remove extension if the dot is after the last slash (i.e., in
  // filename, not directory)
  if (last_dot != std::string::npos &&
      (last_slash == std::string::npos || last_dot > last_slash)) {
    clean_filepath = clean_filepath.substr(0, last_dot);
  }

  return clean_filepath + extension;
}

}  // namespace model
}  // namespace MLLib