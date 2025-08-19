#include "MLLib.hpp"
#include <filesystem>
#include <iostream>

using namespace MLLib;
using namespace MLLib::model;
using namespace MLLib::model::autoencoder;

int main() {
  printf("=== Debug Model Loading ===\n");

  try {
    printf("1. Checking saved model files...\n");

    std::string load_path = "./saved_models/autoencoder_demo";
    std::string binary_path = load_path + ".bin";

    if (std::filesystem::exists(binary_path)) {
      printf("   Binary file exists: %s\n", binary_path.c_str());
      auto file_size = std::filesystem::file_size(binary_path);
      printf("  ✅ File size: %zu bytes\n", file_size);
    } else {
      printf("  ❌ Binary file not found: %s\n", binary_path.c_str());
      return 1;
    }

    printf("\n2. Testing load_model_data...\n");
    auto model_data =
        GenericModelIO::load_model_data(load_path, SaveFormat::BINARY);
    if (model_data) {
      printf("  ✅ load_model_data successful\n");
      printf("  ✅ Data keys: %zu\n", model_data->size());
      for (const auto& [key, value] : *model_data) {
        printf("    - %s: %zu bytes\n", key.c_str(), value.size());
      }
    } else {
      printf("  ❌ load_model_data failed\n");
      return 1;
    }

    printf("\n3. Testing model creation...\n");
    auto model = std::make_unique<DenseAutoencoder>();
    printf("  ✅ DenseAutoencoder created\n");

    printf("\n4. Testing deserialization...\n");
    if (model->deserialize(*model_data)) {
      printf("  ✅ Deserialization successful\n");
    } else {
      printf("  ❌ Deserialization failed\n");
      return 1;
    }

    printf("\n✅ All tests passed!\n");

  } catch (const std::exception& e) {
    printf("❌ Exception: %s\n", e.what());
    return 1;
  }

  return 0;
}
