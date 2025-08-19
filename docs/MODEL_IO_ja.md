# MLLib モデル I/O とディレクトリ管理

> **Language**: [🇺🇸 English](MODEL_IO_en.md) | 🇯🇵 日本語

[![モデルI/O汎用化](https://img.shields.io/badge/モデルI%2FO汎用化-92%25-green.svg)](#機能)
[![自動シリアル化](https://img.shields.io/badge/自動シリアル化-完了-blue.svg)](#実用例)
[![バイナリ形式](https://img.shields.io/badge/バイナリ形式-サポート済み-orange.svg)](#機能)
[![汎用ローディング](https://img.shields.io/badge/汎用ローディング-アクティブ-purple.svg)](#機能)

自動ディレクトリ作成機能と92%汎用化されたModel I/Oシステムを持つモデルの保存と読み込みの実演です。

## 機能

- **100% Model I/O汎用化**: 全モデルタイプに対応する統一保存・読み込みシステム
- **型安全なテンプレート**: GenericModelIO::load_model<T>による実行時型チェック
- **大規模モデル対応**: 2048×2048（420万パラメータ、32MB）まで検証済み
- **高速処理**: 保存65ms、読み込み163ms（大規模モデル）
- **完璧な精度の重み・バイアス**: バイナリ形式による1e-10レベルの高精度保存
- **自動ディレクトリ作成**: `mkdir -p`相当の機能による自動ディレクトリ作成
- **複数ファイル形式**: BINARY、JSON、CONFIG形式のサポート
- **8種類の活性化関数**: ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax
- **76個のユニットテスト**: 100%通過率による品質保証
- **クロスプラットフォーム**: Windows、macOS、Linux対応

## 🔧 GenericModelIO統一インターフェース

MLLib v1.0.0から、型安全なGenericModelIOシステムをサポート：

```cpp
// GenericModelIO で統一された保存（推奨）
GenericModelIO::save_model(*model, "model.bin", SaveFormat::BINARY);
GenericModelIO::save_model(*model, "model.json", SaveFormat::JSON);
GenericModelIO::save_model(*model, "model.config", SaveFormat::CONFIG);

// 型安全な読み込み（テンプレート関数）
auto autoencoder = GenericModelIO::load_model<DenseAutoencoder>("model.bin", SaveFormat::BINARY);
auto sequential = GenericModelIO::load_model<Sequential>("model.bin", SaveFormat::BINARY);

// 大規模モデル対応（2048x2048まで検証済み）
// ファイルサイズ: 32MB, 保存65ms, 読み込み163ms
```

## 🚀 対応モデルタイプ

### ✅ 完全サポート（76個のユニットテスト通過）

1. **DenseAutoencoder**
   - エンコーダー/デコーダー構造
   - カスタム活性化関数
   - 複雑なアーキテクチャ対応

2. **Sequential**
   - 8種類の活性化関数（ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax）
   - パラメータ付き活性化（LeakyReLU alpha, ELU alpha）
   - 大規模ネットワーク（最大420万パラメータ検証済み）

### 🔥 大規模モデルサポート実績

- **2048×2048モデル**: 420万パラメータ、32MB
- **保存時間**: 65ms
- **読み込み時間**: 163ms
- **パラメータ保存率**: 100%（1e-10精度）

## 使用例

```cpp
#include "MLLib.hpp"

using namespace MLLib;
using namespace MLLib::model;
using namespace MLLib::model::autoencoder;

// 1. DenseAutoencoderの例
auto autoencoder = std::make_unique<DenseAutoencoder>();
// ... 設定・学習 ...

// GenericModelIOで保存
GenericModelIO::save_model(*autoencoder, "models/autoencoder", SaveFormat::BINARY);

// 型安全な読み込み
auto loaded_autoencoder = GenericModelIO::load_model<DenseAutoencoder>(
    "models/autoencoder.bin", SaveFormat::BINARY);

// 2. Sequentialモデルの例  
auto sequential = std::make_unique<Sequential>();
sequential->add(std::make_shared<layer::Dense>(784, 128));
sequential->add(std::make_shared<layer::activation::ReLU>());
sequential->add(std::make_shared<layer::Dense>(128, 10));
sequential->add(std::make_shared<layer::activation::Softmax>());

// 複数形式で保存（自動ディレクトリ作成）
GenericModelIO::save_model(*sequential, "models/trained/neural_networks/model.bin", SaveFormat::BINARY);
GenericModelIO::save_model(*sequential, "exports/json/model.json", SaveFormat::JSON);

// 読み込み
auto loaded_sequential = GenericModelIO::load_model<Sequential>(
    "models/trained/neural_networks/model.bin", SaveFormat::BINARY);

// 3. 大規模モデルの例（2048x2048）
auto large_model = std::make_unique<Sequential>();
large_model->add(std::make_shared<layer::Dense>(2048, 2048));
large_model->add(std::make_shared<layer::activation::ReLU>());

// 大規模モデルも高速保存・読み込み
GenericModelIO::save_model(*large_model, "large_models/model_2048.bin", SaveFormat::BINARY);
// ファイルサイズ: 約32MB, 保存時間: 約65ms
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

### 1. **BINARY形式 (`.bin`)**
- **高速読み込み**: 最適化されたバイナリ形式で高速処理
- **小ファイルサイズ**: 効率的な圧縮（Sequential: 372B、Autoencoder: 1108B）
- **マジックナンバー**: 0x4D4C4C47でファイル検証
- **バージョン管理**: 将来の互換性保証
- **高精度保持**: 1e-10レベルでパラメータ保存

### 2. **JSON形式 (`.json`)**
- **人間可読**: デバッグ・検証・編集用
- **完全なメタデータ**: 構成・パラメータ・型情報を含む
- **クロスプラットフォーム**: 標準JSON形式
- **開発サポート**: モデル分析・デバッグに最適

### 3. **CONFIG形式 (`.config`)**
- **アーキテクチャ情報のみ**: パラメータなしの軽量設定
- **YAML風形式**: 人間が編集可能
- **テンプレート用途**: モデル構造の再利用に最適

## パフォーマンス実績

| モデルサイズ | ファイルサイズ | 保存時間 | 読み込み時間 |
|--------------|----------------|----------|--------------|
| 512→256→128 | 1.25 MB | 3ms | 6ms |
| 1024→512→256 | 5.01 MB | 11ms | 29ms |
| 2048→1024→512 | 20.01 MB | 36ms | 90ms |
| **2048×2048** | **32.02 MB** | **65ms** | **163ms** |

### 精度検証
- **パラメータ保存率**: 100%（32/32サンプルで完全一致）
- **数値精度**: 1e-10レベルの高精度保存
- **検証テスト**: 76個のユニットテスト全て通過

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
