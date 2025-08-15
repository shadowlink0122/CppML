# MLLib - C++ 機械学習ライブラリ

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![GPU CI](https://github.com/shadowlink0122/CppML/workflows/GPU%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/gpu-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_unit_tests-brightgreen.svg)](#-テスト)
[![Integration Tests](https://img.shields.io/badge/integration-3429%2F3429_assertions-brightgreen.svg)](#-テスト)
[![GPU Tests](https://img.shields.io/badge/GPU_tests-145_assertions-blue.svg)](#-gpuサポート)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25_CI_success-brightgreen.svg)](#-テスト)

> **言語**: [English](README.md) | [日本語](README_ja.md)

パフォーマンスと使いやすさを重視したモダンなC++17機械学習ライブラリです。

## ✨ 特徴

- **🧠 ニューラルネットワーク**: カスタマイズ可能なSequentialモデル
- **📊 レイヤー**: Dense（全結合）、ReLU、Sigmoid、Tanh活性化関数
- **🎯 訓練**: MSE損失関数とSGD最適化器
- **🎯 マルチGPUサポート**: NVIDIA CUDA、AMD ROCm、Intel oneAPI、Apple Metalの自動検出対応
- **�💾 モデル I/O**: バイナリ、JSON、設定ファイル形式での保存・読み込み
- **📁 自動ディレクトリ作成**: `mkdir -p` 相当の機能
- **🔧 型安全性**: enum ベースの形式指定で信頼性向上
- **⚡ パフォーマンス**: NDArray バックエンドによる最適化されたC++17実装
- **🧪 テスト**: 包括的なユニット（21/21）と結合テスト（3429/3429アサーション）
- **🖥️ GPU テスト**: 145個のGPU専用アサーションとフォールバック検証
- **🔄 クロスプラットフォーム**: Linux、macOS、Windows対応
- **🎯 CI/CD対応**: 本番デプロイメント用100%テスト成功率

## 🚀 クイックスタート

### 前提条件

- C++17対応コンパイラ (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+ (オプション)
- Make

### ビルドとテスト

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make                         # ライブラリをビルド
make test                    # 全テスト実行（単体 + 結合テスト）
make unit-test              # 単体テスト実行（21/21 通過）
make integration-test       # 結合テスト実行（3429/3429 アサーション）
make simple-integration-test # シンプル結合テスト実行
make xor                    # XORニューラルネットワークサンプル実行
```

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

## � モデル I/O

MLLibは複数のモデルシリアライゼーション形式をサポートしています：

### モデルの保存

```cpp
using namespace MLLib;

// enum ベースの形式指定（型安全）
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(model, "model.json", model::ModelFormat::JSON);
model::ModelIO::save_model(model, "model.config", model::ModelFormat::CONFIG);
```

### モデルの読み込み

```cpp
// 保存したモデルを読み込み
auto loaded_model = model::ModelIO::load_model("model.bin", model::ModelFormat::BINARY);

// 予測精度が保持されることを確認
auto original_pred = model.predict({0.5, 0.5});
auto loaded_pred = loaded_model->predict({0.5, 0.5});
// original_pred ≈ loaded_pred
```

### 対応形式

- **BINARY**: 高速でコンパクトなバイナリ形式
- **JSON**: 人間が読める形式、デバッグ用
- **CONFIG**: 設定ファイル形式、パラメータ調整用

## �🛠️ 開発

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

### ビルド

```bash
# ライブラリをビルド
make all

# デバッグビルド
make debug

# サンプルをビルド
make examples

### テスト

```bash
# 全テスト実行
make test                      # 完全なテストスイート（単体 + 結合テスト）
make unit-test                # 単体テスト実行（21/21 通過）
make integration-test         # 包括的結合テスト（3429/3429 アサーション）
make simple-integration-test  # シンプル結合テスト（基本機能）

# テスト出力には実行時間監視が含まれます：
# ✅ BasicXORModelTest PASSED (5 assertions, 0.17ms)
# ✅ AdamOptimizerIntegrationTest PASSED (10 assertions, 1.04ms)
# ✅ BackendPerformanceIntegrationTest PASSED (551 assertions, 43.54ms)
# 🎉 ALL INTEGRATION TESTS PASSED! (3429/3429 assertions, 100% success rate)
```

### テストコンポーネント

- **単体テスト（21/21）**: Config、NDArray、Dense Layer、活性化関数、Sequential Model
- **結合テスト（3429/3429 アサーション）**: XOR問題の学習と予測精度検証
- **シンプル結合テスト**: 基本機能検証

### 結合テスト

```bash
make integration-test           # 包括的結合テスト（3429/3429 アサーション）
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

- **単体テスト**: 21/21 通過（100%）
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

### モデル形式テスト

全モデルI/O形式をテスト：

```bash
make model-format-test  # enumベース形式システムをテスト
```

## ⚠️ 現在の制限事項

- - ニューラルネットワーク学習と推論のCore機能
- GPU/CPU両対応の計算バックエンド
- 現在はDense層のみ実装（CNN、RNN層は将来実装予定）
- 現在はSGD最適化器のみ完全実装（Adam等は部分実装）

### 開発ツールのインストール

```bash
make install-tools
```

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

# CPUのみテスト
FORCE_CPU_ONLY=1 make test    # CPUのみ強制テスト
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

詳細なドキュメントは `docs/` フォルダをご覧ください：

- [テストドキュメント](docs/TESTING_ja.md) - 包括的なテストシステムガイド
- [モデルI/O](docs/MODEL_IO_ja.md) - モデル保存・読み込みガイド
- [GPU CI 設定ガイド](docs/GPU_CI_SETUP_ja.md) - GPU テスト環境の設定方法
- [GPU CI Setup Guide (English)](docs/GPU_CI_SETUP_en.md) - GPU testing environment configuration

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

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します！開発に参加する前に、以下のガイドラインに従ってください：

### 貢献ガイドライン

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

## �📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 🙏 謝辞

- モダンなMLフレームワークからインスピレーションを得ています
- モダンC++のベストプラクティスで構築
- パフォーマンスと使いやすさを重視して設計

## 📖 詳細ドキュメント

### ライブラリの構成

#### コアコンポーネント
- **設定管理**: システム全体の設定とコンフィグレーション
- **多次元配列**: 効率的な数値計算のためのNDArray実装
- **デバイス管理**: CPU/GPU計算の抽象化

#### データ処理
- **前処理**: データの正規化、標準化、変換
- **バッチ処理**: 効率的なミニバッチ学習
- **データローダー**: 各種フォーマットからのデータ読み込み

#### ニューラルネットワーク
- **レイヤー**: 全結合、畳み込み、プーリング、ドロップアウト等
- **活性化関数**: ReLU、Sigmoid、Tanh等
- **損失関数**: 平均二乗誤差、クロスエントロピー等
- **オプティマイザー**: SGD、Adam等の最適化アルゴリズム

#### モデル構築
- **シーケンシャル**: 層を順次積み重ねるモデル
- **関数型**: 複雑なネットワーク構造対応
- **カスタム**: 独自モデルの実装サポート

### 使用例

#### 基本的な使用方法

```cpp
#include "MLLib.hpp"
#include <vector>

int main() {
    // ライブラリ初期化
    MLLib::initialize();
    
    // データの準備
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // シンプルな線形回帰モデル
    MLLib::Sequential model;
    model.add(new MLLib::Dense(4, 1));  // 入力4次元、出力1次元
    
    // モデルのコンパイル
    model.compile(
        MLLib::SGD(0.01),        // 学習率0.01のSGD
        MLLib::MSE()             // 平均二乗誤差
    );
    
    // クリーンアップ
    MLLib::cleanup();
    return 0;
}
```

#### 多層ニューラルネットワーク

```cpp
#include "MLLib.hpp"

int main() {
    MLLib::initialize();
    
    // 分類用の多層ニューラルネットワーク
    MLLib::Sequential model;
    
    // 入力層からの全結合層
    model.add(new MLLib::Dense(784, 128));  // MNIST用 (28x28=784)
    model.add(new MLLib::ReLU());           // ReLU活性化
    
    // 隠れ層
    model.add(new MLLib::Dense(128, 64));
    model.add(new MLLib::ReLU());
    model.add(new MLLib::Dropout(0.5));     // ドロップアウト
    
    // 出力層
    model.add(new MLLib::Dense(64, 10));    // 10クラス分類
    model.add(new MLLib::Softmax());        // ソフトマックス
    
    // コンパイル
    model.compile(
        MLLib::Adam(0.001),                 // Adam最適化
        MLLib::CrossEntropy()               // クロスエントロピー損失
    );
    
    MLLib::cleanup();
    return 0;
}
```

### パフォーマンス最適化

#### メモリ管理
- 効率的なメモリプールの使用
- 不要なコピーの削減
- RAII（Resource Acquisition Is Initialization）パターンの採用

#### 計算最適化
- SIMD命令の活用
- 並列処理のサポート
- キャッシュフレンドリーなデータ構造

### トラブルシューティング

#### よくある問題

**Q: コンパイルエラーが発生する**
```bash
# 必要なツールがインストールされているか確認
make install-tools

# コンパイラのバージョンを確認
g++ --version
clang++ --version
```

**Q: フォーマットエラーが発生する**
```bash
# 自動フォーマットを実行
make fmt

# フォーマットを確認
make fmt-check
```

**Q: ライブラリが見つからない**
```bash
# ライブラリをビルド
make clean
make all

# インクルードパスを確認
# -I./include オプションが必要
```

### 開発ガイドライン

#### コード規約
- K&Rスタイルのブレース配置
- 2スペースのインデント
- 80文字の行長制限
- const correctnessの徹底

#### テスト戦略
- 単体テストの作成
- 統合テストの実装
- パフォーマンステストの追加

#### ドキュメント
- Doxygenコメントの記述
- サンプルコードの提供
- APIリファレンスの維持
