# MLLib モデル## 🔧 型安全なenum形式

MLLib v1.0.0から、型安全なenum形式をサポート：

```cpp
// enum 形式での保存（推奨）
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(model, "model.json", model::ModelFormat::JSON);
model::ModelIO::save_model(model, "model.config", model::ModelFormat::CONFIG);

// 文字列からenum変換（従来コード対応）
auto format = model::ModelIO::string_to_format("binary");
model::ModelIO::save_model(model, "model.bin", format);

// enumから文字列変換
std::string format_name = model::ModelIO::format_to_string(model::ModelFormat::BINARY);
```

## 実用例

```cpp
#include "MLLib/MLLib.hpp"

using namespace MLLib;ディレクトリ管理

この例では、自動ディレクトリ作成機能を持つモデルの保存と読み込み方法を説明します。

## 機能

- **自動ディレクトリ作成**: `mkdir -p` と同等の機能でディレクトリを自動作成
- **複数ファイル形式対応**: バイナリ、JSON、設定ファイル形式をサポート
- **ネストしたディレクトリ対応**: 深い階層のパスも処理可能
- **クロスプラットフォーム**: Windows、macOS、Linuxで動作

## 使用例

```cpp
#include "MLLib/MLLib.hpp"

using namespace MLLib;

// モデルを作成
auto model = std::make_unique<model::Sequential>();
model->add(std::make_shared<layer::Dense>(2, 4, true));
model->add(std::make_shared<layer::activation::ReLU>());
model->add(std::make_shared<layer::Dense>(4, 1, true));
model->add(std::make_shared<layer::activation::Sigmoid>());

// ネストしたディレクトリに完全なモデルを保存（ディレクトリ自動作成）
model::ModelIO::save_model(*model, "models/trained/neural_networks/my_model.bin", 
                          model::ModelFormat::BINARY);

// 異なるコンポーネントを整理されたディレクトリに保存
model::ModelIO::save_config(*model, "configs/architectures/my_model_config.txt");
model::ModelIO::save_parameters(*model, "weights/checkpoints/my_model_params.bin");

// 異なる形式で保存（enum使用）
model::ModelIO::save_model(*model, "exports/binary/my_model.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(*model, "exports/json/my_model.json", model::ModelFormat::JSON);
model::ModelIO::save_model(*model, "exports/config/my_model.config", model::ModelFormat::CONFIG);

// ネストしたディレクトリから読み込み
auto loaded_model = model::ModelIO::load_model("models/trained/neural_networks/my_model.bin", 
                                              model::ModelFormat::BINARY);

// アーキテクチャとパラメータを別々に読み込み
auto config_model = model::ModelIO::load_config("configs/architectures/my_model_config.txt");
model::ModelIO::load_parameters(*config_model, "weights/checkpoints/my_model_params.bin");
```

## ディレクトリ構造例

完全なテストを実行した後、以下のディレクトリ構造が作成されます：

```
├── models/trained/neural_networks/
│   └── test_model.bin
├── exports/
│   ├── binary/
│   │   └── model_v1.bin
│   └── json/
│       └── model_v1.json
├── configs/architectures/
│   └── model_v1_config.txt
├── weights/checkpoints/
│   └── model_v1_params.bin
└── data/projects/ml_experiments/2024/august/neural_networks/xor_solver/
    └── final_model.bin
```

## 設定ファイル形式

設定ファイルは人間が読みやすいYAML形式で保存されます：

```yaml
# MLLib Model Configuration
model_type: Sequential
version: 1.0.0
device: CPU
layers:
  - type: Dense
    input_size: 2
    output_size: 4
    use_bias: true
  - type: ReLU
  - type: Dense
    input_size: 4
    output_size: 1
    use_bias: true
  - type: Sigmoid
```

## ファイル形式

1. **バイナリ形式 (`.bin`)**: マジックナンバーとバージョン管理を含む効率的なストレージ
2. **JSON形式 (`.json`)**: 完全なモデル情報を含む人間が読める形式
3. **設定形式 (`.txt`)**: YAML形式でのアーキテクチャ情報のみ
4. **パラメータ形式 (`.bin`)**: 重みとバイアスのみのバイナリ形式

## エラーハンドリング

- 適切なエラー報告を伴う自動ディレクトリ作成
- ファイル形式の検証
- バージョン互換性チェック
- トラブルシューティング用の包括的なエラーメッセージ

## プラットフォーム対応

- **macOS/Linux**: `mkdir -p` システムコマンドを使用
- **Windows**: 適切なエラーハンドリングを伴う `mkdir` を使用
- **C++17**: 拡張機能として利用可能な場合は `std::filesystem` を使用

## 実用的な使用シナリオ

### 1. 機械学習実験の管理

```cpp
// 実験ごとにディレクトリを整理
std::string experiment_id = "exp_2024_08_14";
model::ModelIO::save_model(*model, 
    "experiments/" + experiment_id + "/models/final_model.bin", "binary");
model::ModelIO::save_config(*model, 
    "experiments/" + experiment_id + "/configs/architecture.txt");
```

### 2. チェックポイント保存

```cpp
// エポックごとのチェックポイント
for (int epoch = 0; epoch < 1000; epoch += 100) {
    std::string checkpoint_path = "checkpoints/epoch_" + std::to_string(epoch) + ".bin";
    model::ModelIO::save_parameters(*model, checkpoint_path);
}
```

### 3. モデルのデプロイメント

```cpp
// 本番環境用にモデルをエクスポート
model::ModelIO::save_model(*model, "deploy/production/model_v1_0_0.bin", "binary");
model::ModelIO::save_model(*model, "deploy/staging/model_v1_0_0.json", "json");
```

## ベストプラクティス

- **バージョン管理**: ファイル名にバージョン番号を含める
- **日付管理**: ディレクトリ名に日付を含める
- **環境分離**: 開発、ステージング、本番環境を分ける
- **バックアップ**: 重要なモデルは複数の場所に保存
- **文書化**: 各モデルの用途と設定をドキュメント化
