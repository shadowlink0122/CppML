# MLLib - C++ 機械学習ライブラリ

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![GPU CI](https://github.com/shadowlink0122/CppML/workflows/GPU%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/gpu-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-76%2F76_unit_tests-brightgreen.svg)](#-テスト)
[![Integration Tests](https://img.shields.io/badge/integration-3429%2F3429_assertions-brightgreen.svg)](#-テスト)
[![GPU Tests](https://img.shields.io/badge/GPU_tests-145_assertions-blue.svg)](#-gpuサポート)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25_CI_success-brightgreen.svg)](#-テスト)

> **言語**: [English](README.md) | [日本語](README_ja.md)

ニューラルネットワークの訓練とモデル永続化のための包括的なテストとパフォーマンス監視を備えたモダンなC++17機械学習ライブラリです。

## ✨ 特徴

- **🧠 ニューラルネットワーク**: カスタマイズ可能なSequentialモデル
- **📊 レイヤー**: Dense（全結合）、ReLU、Sigmoid、Tanh活性化関数
- **🎯 訓練**: MSE損失関数とSGD最適化器
- **🎯 マルチGPUサポート**: NVIDIA CUDA、AMD ROCm、Intel oneAPI、Apple Metalの自動検出対応
- **�💾 モデル I/O**: バイナリ、JSON、設定ファイル形式での保存・読み込み
- **📁 自動ディレクトリ作成**: `mkdir -p` 相当の機能
- **🔧 型安全性**: enum ベースの形式指定で信頼性向上
- **⚡ パフォーマンス**: NDArray バックエンドによる最適化されたC++17実装
- **🧪 テスト**: 包括的なユニット（76/76）と結合テスト（3429/3429アサーション）
- **🖥️ GPU テスト**: 145個のGPU専用アサーションとフォールバック検証
- **🔄 クロスプラットフォーム**: Linux、macOS、Windows対応
- **🎯 CI/CD対応**: 本番デプロイメント用100%テスト成功率

## 🚀 クイックスタート

### 前提条件

- C++17対応コンパイラ (GCC 7+, Clang 5+, MSVC 2017+)
- Make

### ビルドとテスト

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make                         # 自動依存関係管理でビルド
make test                    # 全テスト実行（単体 + 結合テスト）

# 代替ビルドオプション：
make setup-deps              # 依存関係のダウンロードのみ
make minimal                 # JSON読み込み機能なしでビルド（依存関係不要）
make json-support            # 完全なJSON I/Oサポートでビルド
make deps-check              # 依存関係ステータス確認

make unit-test               # 単体テスト実行（76/76 通過）
make integration-test        # 結合テスト実行（3429/3429 アサーション）
make simple-integration-test # シンプル結合テスト実行

# CI最適化テスト（高速実行用）
make build-tests             # テスト実行ファイルのみビルド（CIアーティファクト用）
make unit-test-run-only      # ビルド済み実行ファイルで単体テスト実行
make integration-test-run-only # ビルド済み実行ファイルで結合テスト実行

make samples                 # 全サンプルプログラムをビルド
make xor                     # XORニューラルネットワークサンプルを実行
make device-detection        # GPUデバイス検出サンプルを実行
make gpu-vendor-detection    # GPUベンダー検出サンプルを実行
```

### 🌍 クロスプラットフォーム対応

MLLibは全プラットフォームで依存関係を自動管理します：

#### クイックスタート（全プラットフォーム）
```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make          # 依存関係を自動ダウンロードしてビルド
make test     # 完全テストスイート
```

#### プラットフォーム固有の注意点
- **Linux**: `curl` または `wget` が必要（通常プリインストール）
- **macOS**: 内蔵の `curl` を使用（追加設定不要）
- **Windows**: Git Bash、MSYS2、またはWSLで動作

#### オフライン/企業環境
```bash
# 依存関係を手動ダウンロード：
tools/setup-deps.sh        # クロスプラットフォーム設定スクリプト
# または依存関係なしの最小ビルド：
make minimal               # JSON保存対応、読み込み無効
```

詳細な手順は [Dependencies Setup Guide](docs/DEPENDENCIES_SETUP.md) をご覧ください。

### 基本的な使用方法

```cpp
#include "MLLib.hpp"
using namespace MLLib;

// XORニューラルネットワークを作成
model::Sequential model;

// レイヤーを追加
model.add(std::make_shared<layer::Dense>(2, 4));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(4, 1));
model.add(std::make_shared<layer::activation::Sigmoid>());

// 訓練データ
std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

// モデルを訓練
loss::MSE loss;
optimizer::SGD optimizer(0.1);
model.train(X, Y, loss, optimizer, [](int epoch, double loss) {
    printf("エポック %d, 損失: %f\n", epoch, loss);
}, 1000);

// 予測を実行（複数の構文をサポート）
auto pred1 = model.predict(std::vector<double>{1.0, 0.0});  // ベクター構文
auto pred2 = model.predict({1.0, 0.0});                     // 初期化子リスト構文（C++17）

// モデルを保存
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
```

## 💾 モデル I/O

MLLibはGenericModelIOによる統一モデルシリアライゼーションをサポートしています：

### モデルの保存

```cpp
using namespace MLLib;
using namespace MLLib::model;

// GenericModelIO統一インターフェース（型安全）
GenericModelIO::save_model(*model, "model.bin", SaveFormat::BINARY);
GenericModelIO::save_model(*model, "model.json", SaveFormat::JSON);
GenericModelIO::save_model(*model, "model.config", SaveFormat::CONFIG);

// 自動ディレクトリ作成対応
GenericModelIO::save_model(*model, "models/trained/my_model.bin", SaveFormat::BINARY);
```

### モデルの読み込み

```cpp
// 型安全なテンプレート読み込み
auto sequential = GenericModelIO::load_model<Sequential>("model.bin", SaveFormat::BINARY);
auto autoencoder = GenericModelIO::load_model<autoencoder::DenseAutoencoder>("model.bin", SaveFormat::BINARY);

// 予測精度が完全保持されることを確認（1e-10レベル）
auto original_pred = model.predict({0.5, 0.5});
auto loaded_pred = sequential->predict({0.5, 0.5});
// original_pred ≈ loaded_pred (1e-10精度)
```

### 対応モデルタイプ

- **Sequential**: 8種類の活性化関数対応（ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax）
- **DenseAutoencoder**: エンコーダー・デコーダー構造
- **大規模モデル**: 2048×2048（420万パラメータ、32MB）まで検証済み

### 対応形式

- **BINARY**: 高速・コンパクト（保存65ms、読み込み163ms）
- **JSON**: 人間可読・デバッグ用
- **CONFIG**: アーキテクチャ情報のみ

### 自動ディレクトリ作成

```cpp
// ネストしたディレクトリを自動作成（mkdir -p相当）
GenericModelIO::save_model(*model, "models/experiment_1/epoch_100.bin", 
                          SaveFormat::BINARY);
```

## 🧪 テスト

MLLibはパフォーマンス監視機能付きの包括的テストフレームワークを含んでいます：

### テストカバレッジ

- **単体テスト**: 全コンポーネントで76/76テスト通過
- **結合テスト**: エンドツーエンドワークフローと複雑なシナリオ
- **パフォーマンステスト**: 実行時間監視とベンチマーク
- **エラーハンドリング**: 包括的エラー条件テスト

### テストカテゴリ

```bash
# 実行時間監視付き単体テスト
make unit-test

# サンプル出力:
# Running test: ConfigConstantsTest
# ✅ ConfigConstantsTest PASSED (10 assertions, 0.03ms)
# Running test: NDArrayMatmulTest  
# ✅ NDArrayMatmulTest PASSED (11 assertions, 0.02ms)
# 
# Total test execution time: 0.45ms
# Total suite time (including overhead): 0.89ms
```

### テストコンポーネント

- **単体テスト（76/76）**: Config、NDArray、Dense Layer、活性化関数、Sequential Model、Model I/O
- **結合テスト（3429/3429 アサーション）**: XOR問題学習と予測精度検証
- **シンプル結合テスト**: 基本機能検証

### 結合テスト

```bash
make integration-test          # 包括的結合テスト（3429/3429 アサーション）
make simple-integration-test   # シンプル結合テスト（基本機能）

# 包括的結合テストカバレッジ：
# 🎯 XORモデルテスト: 基本機能 + 学習収束（CI安全）
# 🔧 最適化器統合: SGD + Adamフォールバックテスト
# 📊 損失関数統合: MSE + CrossEntropy検証
# 💻 バックエンド統合: CPUバックエンド包括テスト（601アサーション）
# 🔗 レイヤー統合: Denseレイヤー + 活性化関数
# 🛠️ ユーティリティ統合: Matrix、Random、Validation（504アサーション）
# 📱 デバイス統合: CPUデバイス操作（2039アサーション）
# 📁 データ統合: 読み込み、バッチ処理、検証（157アサーション）
# ⚡ パフォーマンステスト: 安定性 + 実行時間監視

# テスト結果サマリー:
# ✅ 100% CI成功率（3429/3429 アサーション通過）
# ✅ 全テストが決定的でCI対応
# ✅ 包括的なコンポーネント統合カバレッジ
```

### ビルド

```bash
# ライブラリをビルド
make

# デバッグビルド
make debug

# サンプルをビルドして実行
make samples
make run-sample SAMPLE=xor           # 汎用ランナーで特定サンプル実行
make xor                            # XORニューラルネットワーク（エイリアス）
make device-detection               # GPUデバイス検出（エイリアス）
make gpu-vendor-detection           # GPUベンダー検出（エイリアス）

# クリーン（訓練出力を削除）
make clean
```

### 開発ツールのインストール

```bash
make install-tools
```

### ビルド

```bash
# ライブラリをビルド
make all

# デバッグビルド
make debug

# サンプルをビルド
make examples

## 🛠️ 開発

### テスト

```bash
# 全テスト実行
make test                     # 完全なテストスイート（単体 + 結合テスト）
make unit-test                # 単体テスト実行（76/76 通過）
make integration-test         # 包括的結合テスト（3429/3429 アサーション）
make simple-integration-test  # シンプル結合テスト（基本機能）

# テスト出力には実行時間監視が含まれます：
# ✅ BasicXORModelTest PASSED (5 assertions, 0.17ms)
# ✅ AdamOptimizerIntegrationTest PASSED (10 assertions, 1.04ms)
# ✅ BackendPerformanceIntegrationTest PASSED (551 assertions, 43.54ms)
# 🎉 ALL INTEGRATION TESTS PASSED! (3429/3429 assertions, 100% success rate)
```

### コード品質

```bash
# コードをフォーマット
make fmt

# フォーマットをチェック
make fmt-check

# リントを実行
make lint

# 静的解析
make check

# 全品質チェック
make lint-all
```

MLLibは100%のテスト成功率を持つ包括的なCI/CDパイプラインを特徴としています：

### GitHub Actionsパイプライン

```yaml
# 完全なCIパイプラインワークフロー:
Format Validation → Linting → Static Analysis
         ↓
    Build Test → Unit Tests → Integration Tests
         ↓
     CI Summary
```

## 🔄 CI/CD と品質保証

MLLibは100%のテスト成功率を持つ包括的なCI/CDパイプラインを特徴としています：

### GitHub Actionsパイプライン

```yaml
# 完全なCIパイプラインワークフロー:
Format Validation → Linting → Static Analysis
         ↓
    Build Test → Unit Tests → Integration Tests
         ↓
     CI Summary
```

### テストカバレッジ

- **単体テスト**: 76/76 通過（100%）
- **結合テスト**: 3429/3429 アサーション通過（100%）
- **シンプル結合テスト**: 基本機能検証
- **CI要件**: 100%決定的テスト成功率

### 品質チェック

```bash
make fmt-check    # コードフォーマット検証
make lint         # Clang-tidyリンティング
make check        # Cppcheck静的解析
make lint-all     # 全品質チェック統合
```

### CI機能

- ✅ **決定的テスト**: CI信頼性のために設計された全テスト
- ✅ **包括的カバレッジ**: エンドツーエンドコンポーネント統合テスト
- ✅ **パフォーマンス監視**: 全テストの実行時間追跡
- ✅ **マルチプラットフォーム対応**: Ubuntu、macOS、Windows互換性
- ✅ **プロダクション対応**: デプロイメント信頼性のための100%テスト成功率

## 📁 プロジェクト構造

```
CppML/
├── include/MLLib/         # ヘッダーファイル
│   ├── config.hpp         # ライブラリ設定
│   ├── ndarray.hpp        # テンソル演算
│   ├── device/            # デバイス管理
│   ├── layer/             # ニューラルネットワークレイヤー
│   ├── loss/              # 損失関数
│   ├── optimizer/         # 最適化アルゴリズム
│   └── model/             # モデルアーキテクチャとI/O
├── src/MLLib/            # 実装ファイル
├── samples/              # サンプルプログラム
├── tests/                # テストファイル
├── docs/                 # ドキュメント
└── README.md             # このファイル
```

### サンプル実行

XORニューラルネットワークサンプル：

```bash
make xor  # XORニューラルネットワークサンプルを実行
```

### サンプル実行

全サンプルプログラムをテスト：

```bash
make samples               # 全サンプルプログラムをビルド
make run-sample            # 利用可能なサンプル一覧を表示
make run-sample SAMPLE=xor # 特定サンプルを実行
make xor                   # XORニューラルネットワークサンプル実行（エイリアス）
```

## ⚠️ 現在の制限事項

- **コア機能**: ニューラルネットワーク学習と推論
- **GPU/CPUバックエンド**: GPU・CPU両対応の計算バックエンド
- **レイヤー実装**: 現在はDense層のみ（CNN、RNN層は将来リリース予定）
- **最適化器実装**: 現在はSGD完全実装（Adam等は部分実装）

## 🛠️ ロードマップ

### 完了済み ✅
- [x] 包括的単体テストフレームワーク（76/76テスト）
- [x] パフォーマンス監視付き結合テスト
- [x] 実行時間測定とベンチマーク
- [x] 活性化関数付きDense層（ReLU、Sigmoid、Tanh）
- [x] Sequentialモデルアーキテクチャ
- [x] MSE損失とSGD最適化器
- [x] モデル保存・読み込み（バイナリ、設定形式）

### 進行中 🚧
- [ ] GPU加速サポート
- [ ] JSONモデル読み込み実装
- [ ] 高度なエラーハンドリングと検証

### 計画中 📋
- [ ] 追加レイヤータイプ（Convolutional、LSTM、Dropout）
- [ ] より多くの最適化器（Adam、RMSprop、AdaGrad）
- [ ] 高度な損失関数（CrossEntropy、Huber）
- [ ] モデル量子化と最適化
- [ ] Pythonバインディング
- [ ] マルチスレッディングサポート

## 🎯 サンプル

### XORニューラルネットワーク

付属のXORサンプルでは以下を実演しています：
- モデルアーキテクチャ定義
- コールバック関数付き訓練
- エポックベースモデル保存
- 複数ファイル形式サポート

```bash
make xor  # XORニューラルネットワークサンプルを実行
```

### モデル形式テスト

全モデルI/O形式とサンプルプログラムをテスト：

```bash
make samples               # 全サンプルプログラムをビルド
make run-sample            # 利用可能なサンプル一覧
make run-sample SAMPLE=xor # 特定サンプルを実行
make xor                   # XORニューラルネットワークサンプル（エイリアス）
```

## ❓ FAQ

### よくある質問

#### Q: どのC++標準を使用していますか？
A: C++17を使用しています。現代的な言語機能と適切なコンパイラサポートを活用しています。

#### Q: predict()で初期化子リスト（{}構文）を使用できますか？
A: はい！以下のように使用できます：
```cpp
auto result = model.predict({1.0, 2.0, 3.0});
```

#### Q: カスタムレイヤーや損失関数を作成できますか？
A: はい、提供されているベースクラスを継承してカスタムコンポーネントを実装できます。

#### Q: GPUサポートはありますか？
A: はい、NVIDIA CUDA、AMD ROCm、Intel oneAPI、Apple Metalの包括的なマルチGPUサポートを提供しています。ライブラリはデフォルトで全GPUベンダーに対応し、GPU使用不可時は自動的にCPUにフォールバックします。

#### Q: 大規模なデータセットを処理できますか？
A: はい、効率的なメモリ管理とバッチ処理に対応しています。

#### Q: TensorFlowやPyTorchモデルをインポートできますか？
A: 現在はネイティブ形式のみサポートしていますが、将来的な機能として検討中です。

## 🎯 GPU サポート

MLLibは包括的なマルチGPUベンダーサポートと自動検出・フォールバック機能を提供します：

### 機能

- **🌍 マルチベンダー**: NVIDIA CUDA、AMD ROCm、Intel oneAPI、Apple Metal
- **� 自動検出**: 実行時のGPU検出と選択
- **� デフォルトサポート**: 全GPUベンダーをデフォルトで有効化（ライブラリ設計）
- **🛡️ CPU フォールバック**: GPU利用不可時のシームレスなCPU実行
- **⚠️ スマート警告**: GPU状態に関する情報的メッセージ
- **🧪 完全テスト**: ユニットと統合テストでの145個のGPUアサーション

### サポートGPUベンダー

| ベンダー | API | ハードウェアサポート |
|---------|-----|-------------------|
| **NVIDIA** | CUDA, cuBLAS | GeForce、Quadro、Tesla、RTX |
| **AMD** | ROCm, HIP, hipBLAS | Radeon Instinct、Radeon Pro |
| **Intel** | oneAPI, SYCL, oneMKL | Arc、Iris Xe、UHD Graphics |
| **Apple** | Metal, MPS | M1、M1 Pro/Max/Ultra、M2 |

### GPU ビルドオプション

```bash
# デフォルトビルド（全GPUベンダー有効）
make

# 特定GPUサポートの無効化
make DISABLE_CUDA=1           # NVIDIA CUDA無効化
make DISABLE_ROCM=1           # AMD ROCm無効化
make DISABLE_ONEAPI=1         # Intel oneAPI無効化
make DISABLE_METAL=1          # Apple Metal無効化

# CPUのみビルド
make DISABLE_CUDA=1 DISABLE_ROCM=1 DISABLE_ONEAPI=1 DISABLE_METAL=1

# 環境制御によるGPUテスト
FORCE_CPU_ONLY=1 make test    # CPUのみ強制テスト
GPU_SIMULATION=1 make test    # GPUシミュレーションモード有効化
```

### 使用方法

```cpp
#include "MLLib.hpp"

int main() {
    MLLib::model::Sequential model;
    
    // GPUデバイス設定（自動ベンダー検出）
    model.set_device(MLLib::DeviceType::GPU);
    // ライブラリ出力: ✅ GPU device successfully configured
    // または警告: ⚠️ WARNING: GPU device requested but no GPU found!
    
    // ニューラルネットワーク構築
    model.add_layer(new MLLib::layer::Dense(784, 128));
    model.add_layer(new MLLib::layer::activation::ReLU());
    model.add_layer(new MLLib::layer::Dense(128, 10));
    
    // 訓練は自動的に最適GPUを使用
    model.train(train_X, train_Y, loss, optimizer);
    
    return 0;
}
```

### GPU状態確認

```cpp
// GPU利用可能性チェック
if (MLLib::Device::isGPUAvailable()) {
    // 検出されたGPUを表示
    auto gpus = MLLib::Device::detectGPUs();
    for (const auto& gpu : gpus) {
        printf("GPU: %s (%s)\n", gpu.name.c_str(), gpu.api_support.c_str());
    }
}
```

### GPUドキュメント

- **📖 [マルチGPUサポートガイド (日本語)](docs/MULTI_GPU_SUPPORT_ja.md)**
- **📖 [Multi-GPU Support Guide (English)](docs/MULTI_GPU_SUPPORT_en.md)**
- **⚙️ [GPU CI設定ガイド](docs/GPU_CI_SETUP_ja.md)**

## 📚 ドキュメント

詳細なドキュメントは [`docs/`](docs/) ディレクトリで利用できます：

- **📖 [概要](docs/README.md)** - ドキュメント索引とクイックスタート
- **🇺🇸 [English Documentation](docs/README_en.md)** - 英語での完全ガイド
- **🇯🇵 [日本語ドキュメント](docs/README_ja.md)** - 日本語での完全ガイド
- **🇺🇸 [Model I/O Guide (English)](docs/MODEL_IO_en.md)** - 完全モデルシリアライゼーションガイド
- **🇯🇵 [モデル I/O ガイド (日本語)](docs/MODEL_IO_ja.md)** - 日本語でのモデル保存・読み込み解説
- **🇺🇸 [GPU CI Setup Guide (English)](docs/GPU_CI_SETUP_en.md)** - GPUテスト環境設定
- **🇯🇵 [GPU CI 設定ガイド (日本語)](docs/GPU_CI_SETUP_ja.md)** - GPU テスト環境の設定方法
- **🇺🇸 [Testing Guide (English)](docs/TESTING_en.md)** - テストフレームワークドキュメント
- **🇯🇵 [テストガイド (日本語)](docs/TESTING_ja.md)** - テストフレームワークの説明

### APIコンポーネント

- **🧠 モデル**: Sequential、model I/O
- **📊 レイヤー**: Dense（全結合）、活性化関数（ReLU、Sigmoid、Tanh）
- **🎯 最適化器**: SGD（完全実装）、Adam（フォールバック実装）
- **📉 損失関数**: MSE、CrossEntropy
- **⚡ バックエンド**: CPUバックエンド（NDArray）
- **🛠️ ユーティリティ**: Matrix、Random、Validation、I/O utilities

### 実装サンプル

```cpp
#include "MLLib.hpp"
using namespace MLLib;

// シンプルなニューラルネットワークを作成
model::Sequential model;
model.add(std::make_shared<layer::Dense>(128, 64));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(64, 10));
model.add(std::make_shared<layer::activation::Sigmoid>());

// 訓練データを準備
std::vector<std::vector<double>> X, Y;
// ... データを設定 ...

// モデルを訓練
loss::MSE loss;
optimizer::SGD optimizer(0.01);
model.train(X, Y, loss, optimizer, nullptr, 100);

// 予測実行
auto prediction = model.predict({/* 入力データ */});
```

### アーキテクチャ

```
MLLib/
├── NDArray          # 多次元配列（テンソル操作）
├── Device           # CPU/GPUデバイス管理
├── Layer/           # ニューラルネットワーク層
│   ├── Dense        # 全結合層
│   └── Activation/  # 活性化関数（ReLU、Sigmoid）
├── Loss/            # 損失関数（MSE）
├── Optimizer/       # 最適化アルゴリズム（SGD）
└── Model/           # モデル定義とI/O
    ├── Sequential   # Sequentialモデルアーキテクチャ
    └── ModelIO      # モデルシリアライゼーション（Binary/JSON/Config）
```

## 🤝 コントリビューション

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/素晴らしい機能`)
3. 変更を実装
4. 品質チェックを実行 (`make lint-all`)
5. 変更をコミット (`git commit -m '素晴らしい機能を追加'`)
6. ブランチにプッシュ (`git push origin feature/素晴らしい機能`)
7. プルリクエストを作成

### コードスタイル

このプロジェクトはK&Rスタイルのフォーマットを使用しています。提出前に `make fmt` を実行してください。

## � トラブルシューティング

### よくある問題と解決方法

#### コンパイルエラー
- **C++17対応**: コンパイラがC++17をサポートしていることを確認してください
- **依存関係**: 必要なヘッダーファイルが正しくインクルードされていることを確認してください

#### テストの実行
```bash
# 単体テストのみ実行
make test

# 包括的な結合テスト
make integration-test

# 軽量な結合テスト
make simple-integration-test
```

#### パフォーマンスの最適化
- デバッグビルドではなくリリースビルドを使用してください
- バッチサイズを調整してメモリ使用量を最適化してください

## ❓ FAQ

### よくある質問

#### Q: どのC++標準を使用していますか？
A: C++17を使用しています。現代的な言語機能と適切なコンパイラサポートを活用しています。

#### Q: predict()で初期化子リスト（{}構文）を使用できますか？
A: はい！以下のように使用できます：
```cpp
auto result = model.predict({1.0, 2.0, 3.0});
```

#### Q: カスタムレイヤーや損失関数を作成できますか？
A: はい、提供されているベースクラスを継承してカスタムコンポーネントを実装できます。

#### Q: GPUサポートはありますか？
A: はい、NVIDIA CUDA、AMD ROCm、Intel oneAPI、Apple Metalの包括的なマルチGPUサポートを提供しています。ライブラリはデフォルトで全GPUベンダーに対応し、GPU使用不可時は自動的にCPUにフォールバックします。

#### Q: 大規模なデータセットを処理できますか？
A: はい、効率的なメモリ管理とバッチ処理に対応しています。

#### Q: TensorFlowやPyTorchモデルをインポートできますか？
A: 現在はネイティブ形式のみサポートしていますが、将来的な機能として検討中です。

## 🤝 コントリビューション

### よくある問題と解決方法

#### コンパイルエラー
- **C++17対応**: コンパイラがC++17をサポートしていることを確認してください
- **依存関係**: 必要なヘッダーファイルが正しくインクルードされていることを確認してください

#### テストの実行
```bash
# 単体テストのみ実行
make test

# 包括的な結合テスト
make integration-test

# 軽量な結合テスト
make simple-integration-test
```

#### パフォーマンスの最適化
- デバッグビルドではなくリリースビルドを使用してください
- バッチサイズを調整してメモリ使用量を最適化してください

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します！開発に参加する前に、以下のガイドラインに従ってください：

### コントリビューションガイドライン

1. **テスト**: 新機能には適切なテストを追加してください
2. **ドキュメント**: APIの変更は適切にドキュメント化してください
3. **コードスタイル**: 既存のコードスタイルに従ってください
4. **CI**: すべてのCIテストが成功することを確認してください

### 開発環境のセットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd CppML

# 依存関係を確認
make check-deps

# 開発用ビルド
make debug

# 包括的なテストを実行
make integration-test
```

### 報告とフィードバック

- **バグ報告**: GitHubのIssuesタブでバグを報告してください
- **機能リクエスト**: 新機能のアイデアやリクエストをお寄せください
- **ディスカッション**: 実装に関する議論や質問はDiscussionsタブをご利用ください

### コードスタイル

このプロジェクトはK&Rスタイルのフォーマットを使用しています。提出前に `make fmt` を実行してください。

## 📄 ライセンス

このプロジェクトは商業利用制限付きBSD 3-Clauseライセンスの下でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 🙏 謝辞
