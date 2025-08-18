#pragma once

#include "base_model.hpp"
#include <functional>
#include <memory>
#include <unordered_map>

namespace MLLib {
namespace model {

/**
 * @brief Model factory for automatic model creation and registration
 */
class ModelFactory {
public:
  /**
   * @brief Register a model type for automatic creation
   * @tparam ModelClass Model class to register
   * @param type Model type enum
   */
  template <typename ModelClass> static void register_model(ModelType type) {
    creators_[type] = []() -> std::unique_ptr<ISerializableModel> {
      return std::make_unique<ModelClass>();
    };
  }

  /**
   * @brief Create a model instance by type
   * @param type Model type to create
   * @return Model instance or nullptr if type not registered
   */
  static std::unique_ptr<ISerializableModel> create_model(ModelType type) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second();
    }
    return nullptr;
  }

  /**
   * @brief Check if a model type is registered
   * @param type Model type to check
   * @return True if registered
   */
  static bool is_registered(ModelType type) {
    return creators_.find(type) != creators_.end();
  }

  /**
   * @brief Get all registered model types
   * @return Vector of registered model types
   */
  static std::vector<ModelType> get_registered_types() {
    std::vector<ModelType> types;
    for (const auto& [type, creator] : creators_) {
      types.push_back(type);
    }
    return types;
  }

private:
  static std::unordered_map<
      ModelType, std::function<std::unique_ptr<ISerializableModel>()>>
      creators_;
};

/**
 * @brief Registry initializer for automatic model registration
 */
class ModelRegistry {
public:
  /**
   * @brief Initialize all standard model types
   */
  static void initialize_standard_models();

  /**
   * @brief Register custom model type
   * @tparam ModelClass Model class to register
   * @param type Model type enum
   */
  template <typename ModelClass>
  static void register_custom_model(ModelType type) {
    ModelFactory::register_model<ModelClass>(type);
  }
};

}  // namespace model
}  // namespace MLLib
