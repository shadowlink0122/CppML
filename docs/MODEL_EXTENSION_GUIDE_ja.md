# MLLib新規モデル追加ガイド

## 概要
MLLibに新しいモデルタイプを追加し、ModelIOで簡単にファイル入出力を行うためのガイドです。

## 新規モデル追加手順

### 1. 基本設定

#### ModelType列挙型への追加
```cpp
// include/MLLib/model/base_model.hpp
enum class ModelType {
  // ... 既存のタイプ
  YOUR_NEW_MODEL,  // ← 新しいモデルタイプを追加
};
```

#### 文字列変換関数の更新
```cpp
// src/MLLib/model/model_io.cpp
std::string model_type_to_string(ModelType type) {
  switch (type) {
    // ... 既存のケース
    case ModelType::YOUR_NEW_MODEL: return "YourNewModel";
  }
}

ModelType string_to_model_type(const std::string& type_str) {
  // ... 既存の条件
  if (type_str == "YourNewModel") return ModelType::YOUR_NEW_MODEL;
}
```

### 2. モデルクラスの実装

```cpp
// include/MLLib/model/your_new_model.hpp
#pragma once
#include "base_model.hpp"

class YourNewModel : public BaseModel {
public:
    YourNewModel() : BaseModel(ModelType::YOUR_NEW_MODEL) {}
    
    // ISerializableModel インターフェースの実装
    SerializationMetadata get_serialization_metadata() const override {
        SerializationMetadata metadata;
        metadata.model_type = ModelType::YOUR_NEW_MODEL;
        metadata.version = "1.0.0";
        metadata.device = device_;
        // カスタムプロパティがあれば追加
        metadata.custom_properties["specific_param"] = "value";
        return metadata;
    }
    
    std::unordered_map<std::string, std::vector<uint8_t>> serialize() const override {
        std::unordered_map<std::string, std::vector<uint8_t>> data;
        
        // パラメータをバイナリ化して保存
        // 例：weights, biases, config parametersなど
        
        return data;
    }
    
    bool deserialize(const std::unordered_map<std::string, std::vector<uint8_t>>& data) override {
        // dataからパラメータを復元
        return true;
    }
    
    std::string get_config_string() const override {
        // JSON形式の設定文字列を返す
        return "{}";
    }
    
    bool set_config_from_string(const std::string& config_str) override {
        // 設定文字列からパラメータを設定
        return true;
    }
    
    void set_training(bool training) override {
        // トレーニングモードの設定
    }
};
```

### 3. ファクトリーパターンによる自動化

現在のModelIOをより簡単に使えるように、ファクトリーパターンを実装することを推奨します：

```cpp
// include/MLLib/model/model_factory.hpp
#pragma once
#include "base_model.hpp"
#include <memory>

class ModelFactory {
public:
    template<typename ModelClass>
    static void register_model(ModelType type) {
        creators_[type] = []() -> std::unique_ptr<ISerializableModel> {
            return std::make_unique<ModelClass>();
        };
    }
    
    static std::unique_ptr<ISerializableModel> create_model(ModelType type) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
private:
    static std::unordered_map<ModelType, std::function<std::unique_ptr<ISerializableModel>()>> creators_;
};
```

### 4. 使用例

#### 保存
```cpp
// 新しいモデルの使用例
auto model = std::make_unique<YourNewModel>();
// ... モデルの設定・トレーニング

// 簡単に保存
GenericModelIO::save_model(*model, "my_model.bin", SaveFormat::BINARY);
```

#### 読み込み
```cpp
// GenericModelIOのテンプレート関数を使用（実装が必要）
auto loaded_model = GenericModelIO::load_model<YourNewModel>("my_model.bin", SaveFormat::BINARY);
```

## 現在の制限と改善提案

### 制限事項
1. **GenericModelIO::load_model**のテンプレート関数が未実装
2. **JSON形式の読み込み**が未実装
3. **カスタムレイヤー**の自動対応が不完全

### 改善提案

#### 1. テンプレート読み込み関数の実装
```cpp
template <typename ModelClass>
static std::unique_ptr<ModelClass> load_model(const std::string& filepath, SaveFormat format) {
    // メタデータから型を確認
    auto metadata = load_metadata(filepath);
    if (!metadata) return nullptr;
    
    // ファクトリーで空のモデルを作成
    auto base_model = ModelFactory::create_model(metadata->model_type);
    if (!base_model) return nullptr;
    
    // データを読み込んでデシリアライズ
    auto data = load_model_data(filepath, format);
    if (!data || !base_model->deserialize(*data)) return nullptr;
    
    // 指定された型にキャスト
    return std::unique_ptr<ModelClass>(dynamic_cast<ModelClass*>(base_model.release()));
}
```

#### 2. レイヤー登録システム
```cpp
class LayerRegistry {
public:
    template<typename LayerClass>
    static void register_layer(const std::string& type_name) {
        layer_creators_[type_name] = []() {
            return std::make_shared<LayerClass>();
        };
    }
    
    static std::shared_ptr<Layer> create_layer(const std::string& type_name) {
        auto it = layer_creators_.find(type_name);
        return (it != layer_creators_.end()) ? it->second() : nullptr;
    }
};
```

## 結論

**現在の状況**: 基本的な仕組みは整っていますが、新しいモデルを追加するには一定の実装作業が必要です。

**推奨される改善**: 
1. ファクトリーパターンの完全実装
2. テンプレート読み込み関数の実装
3. レイヤー登録システムの追加

これらの改善により、新しいモデルの追加が**大幅に簡素化**され、`ISerializableModel`を継承するだけで自動的にファイルI/Oが使用できるようになります。
