# MLLib - C++ 機械学習ライブラリ 完全ガイド

> **Language**: [🇺🇸 English](GUIDE_en.md) | 🇯🇵 日本語

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_passing-brightgreen.svg)](#-テスト)

> **言語**: [English](../README.md) | [日本語](README_ja.md)

包括的なテストシステムと実行時間監視機能を備えた、ニューラルネットワーク訓練とモデル永続化のためのモダンC++17機械学習ライブラリです。

## ✨ 機能

- **🧠 ニューラルネットワーク**: カスタマイズ可能なアーキテクチャのSequentialモデル
- **📊 レイヤー**: Dense（全結合）、ReLU、Sigmoid、Tanh活性化関数
- **🎯 訓練**: MSE損失関数とSGD最適化器
- **💾 モデル I/O**: バイナリ、JSON、コンフィグ形式でのモデル保存・読み込み
- **📁 自動ディレクトリ**: `mkdir -p` 機能による自動ディレクトリ作成
- **🔧 型安全性**: 信頼性向上のためのenum形式指定
- **⚡ パフォーマンス**: NDArrayバックエンドによる最適化されたC++17実装
- **🧪 テスト**: 実行時間監視付き包括的単体・統合テスト
- **🔄 クロスプラットフォーム**: Linux、macOS、Windows対応
- **📊 ベンチマーク**: リアルタイムパフォーマンス測定と実行時間追跡

## 🚀 クイックスタート

### 前提条件

- C++17対応コンパイラ（GCC 7+、Clang 5+、MSVC 2017+）
- Make

### ビルドとテスト

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make                    # ライブラリのビルド
make unit-test         # 単体テスト実行（21/21通過）
make integration-test  # 統合テスト実行
make xor              # XORニューラルネットワーク実行
```

### 基本的な使用方法

```cpp
#include "MLLib.hpp"
using namespace MLLib;

// XORニューラルネットワークの作成
model::Sequential model;

// レイヤーの追加
model.add(std::make_shared<layer::Dense>(2, 4));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(4, 1));
model.add(std::make_shared<layer::activation::Sigmoid>());

// 訓練データ
std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

// モデル訓練
loss::MSE loss;
optimizer::SGD optimizer(0.1);
model.train(X, Y, loss, optimizer, [](int epoch, double loss) {
    std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
}, 1000);

// モデル保存
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
```

## 💾 モデル I/O

MLLibは複数のモデルシリアライゼーション形式をサポートします：

### モデル保存

```cpp
using namespace MLLib;

// enum形式指定（型安全）
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(model, "model.json", model::ModelFormat::JSON);
model::ModelIO::save_model(model, "model.config", model::ModelFormat::CONFIG);

// 文字列形式（レガシーサポート）
auto format = model::ModelIO::string_to_format("binary");
model::ModelIO::save_model(model, "model.bin", format);
```

### モデル読み込み

```cpp
// パラメータ付き完全モデルの読み込み
auto model = model::ModelIO::load_model("model.bin", model::ModelFormat::BINARY);

// 設定のみの読み込み（訓練済みパラメータなし）
auto empty_model = model::ModelIO::load_config("model.config");
```

### 自動ディレクトリ作成

```cpp
// 自動的にネストしたディレクトリを作成（mkdir -p相当）
model::ModelIO::save_model(model, "models/experiment_1/epoch_100.bin", 
                          model::ModelFormat::BINARY);
```

## 🧪 テスト

MLLibはパフォーマンス監視付きの包括的テストフレームワークを含んでいます：

### テストカバレッジ

- **単体テスト**: 全コンポーネントで21/21テスト通過
- **統合テスト**: エンドツーエンドワークフローと複雑なシナリオ
- **パフォーマンステスト**: 実行時間監視とベンチマーク
- **エラーハンドリング**: 包括的エラー条件テスト

### テストカテゴリー

```bash
# 実行時間監視付き単体テスト
make unit-test

# 出力例：
# Running test: ConfigConstantsTest
# ✅ ConfigConstantsTest PASSED (10 assertions, 0.03ms)
# Running test: NDArrayMatmulTest  
# ✅ NDArrayMatmulTest PASSED (11 assertions, 0.02ms)
# 
# Total test execution time: 0.45ms
# Total suite time (including overhead): 0.89ms
```

### テストコンポーネント

- **Config Module (3/3)**: ライブラリ設定と定数
- **NDArray Module (6/6)**: 多次元配列操作
- **Dense Layer (4/4)**: 全結合層機能
- **Activation Functions (7/7)**: ReLU、Sigmoid、Tanhとエラーハンドリング
- **Sequential Model (1/1)**: モデルアーキテクチャと予測

### 統合テスト

```bash
make integration-test

# テスト内容：
# - XOR問題のエンドツーエンド訓練
# - モデルI/Oワークフロー（保存・読み込み）
# - マルチレイヤーアーキテクチャ検証
# - パフォーマンスと安定性テスト
```

## 🛠️ 開発

### テスト

```bash
# 全テスト実行
make test                   # 単体＋統合テスト
make unit-test             # 単体テスト実行（21/21通過）
make integration-test      # エンドツーエンド統合テスト

# テスト出力には実行時間監視が含まれます：
# ✅ NDArrayConstructorTest PASSED (14 assertions, 0.01ms)
# Total test execution time: 0.45ms
```

### コード品質

```bash
# コードフォーマット
make fmt

# フォーマットチェック
make fmt-check

# リンティング実行
make lint

# 静的解析
make check

# 全品質チェック
make lint-all
```

### ビルド

```bash
# ライブラリビルド
make

# デバッグビルド
make debug

# サンプルビルドと実行
make samples               # 全サンプルプログラムをビルド
make run-sample            # 利用可能なサンプル一覧を表示
make xor                   # XORニューラルネットワーク（エイリアス）
make device-detection      # GPUデバイス検出サンプル（エイリアス）
make gpu-vendor-detection  # GPUベンダー検出サンプル（エイリアス）

# クリーン（訓練出力削除）
make clean
```

### 開発ツールインストール

```bash
make install-tools
```

## 📚 ドキュメント

[`docs/`](../docs/) ディレクトリに包括的なドキュメントが用意されています：

- **📖 [概要](README.md)** - ドキュメントインデックスとクイックスタート
- **🇺🇸 [English Documentation](README_en.md)** - 英語での完全ガイド
- **🇯🇵 [日本語ドキュメント](README_ja.md)** - 日本語での完全ガイド
- **🇺🇸 [Model I/O Guide (English)](MODEL_IO_en.md)** - 完全なモデルシリアライゼーションガイド
- **🇯🇵 [モデル I/O ガイド (日本語)](MODEL_IO_ja.md)** - 日本語でのモデル保存・読み込み解説
- **🇺🇸 [Testing Guide (English)](TESTING_en.md)** - テストフレームワークドキュメント
- **🇯🇵 [テストガイド (日本語)](TESTING_ja.md)** - テストフレームワークの説明

### APIコンポーネント

- **Core**: `NDArray`（テンソル操作）、`DeviceType`（CPU/GPU管理）
- **Layers**: `Dense`（全結合）、`ReLU`、`Sigmoid`活性化関数
- **Models**: `Sequential`（レイヤー積み重ね）、`ModelIO`（シリアライゼーション）
- **Training**: `MSELoss`（平均二乗誤差）、`SGD`（確率的勾配降下）
- **Utils**: 自動ディレクトリ作成、クロスプラットフォームファイルシステム操作

### アーキテクチャ

```
MLLib/
├── NDArray          # 多次元配列（テンソル操作）
├── Device           # CPU/GPUデバイス管理
├── Layer/           # ニューラルネットワークレイヤー
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
2. 機能ブランチの作成（`git checkout -b feature/amazing-feature`）
3. 変更を実装
4. 品質チェック実行（`make lint-all`）
5. 変更をテスト（`make test`）
6. 変更をコミット（`git commit -m 'Add amazing feature'`）
7. ブランチにプッシュ（`git push origin feature/amazing-feature`）
8. プルリクエストを開く

### コードスタイル

このプロジェクトはK&Rスタイルフォーマットを使用しています。提出前に`make fmt`を実行してください。

## 🎯 例

### XORニューラルネットワーク

含まれているXOR例では以下を実演します：
- モデルアーキテクチャ定義
- コールバック関数による訓練
- エポックベースのモデル保存
- 複数ファイル形式サポート

```bash
make xor  # XORニューラルネットワーク例の実行
```

### モデル形式テスト

全モデルI/O形式のテスト：

```bash
make model-format-test  # enumベース形式システムのテスト
```

## 📁 プロジェクト構造

```
CppML/
├── include/MLLib/         # ヘッダーファイル
│   ├── config.hpp         # ライブラリ設定
│   ├── ndarray.hpp        # テンソル操作
│   ├── device/            # デバイス管理
│   ├── layer/             # ニューラルネットワークレイヤー
│   ├── loss/              # 損失関数
│   ├── optimizer/         # 最適化アルゴリズム
│   └── model/             # モデルアーキテクチャとI/O
├── src/MLLib/            # 実装ファイル
├── sample/               # 例プログラム
├── docs/                 # ドキュメント
└── README.md             # このファイル
```

## ⚠️ 現在の制限

- **GPU対応**: 現在CPU専用（GPU対応は将来のリリースで予定）
- **レイヤータイプ**: Denseと基本活性化関数に制限
- **最適化器**: SGDのみ実装（Adam、RMSprop予定）
- **JSON読み込み**: JSON形式は保存をサポートしていますが読み込みは未実装

## 🛠️ ロードマップ

### 完了済み ✅
- [x] 包括的単体テストフレームワーク（21/21テスト）
- [x] パフォーマンス監視付き統合テスト
- [x] 実行時間測定とベンチマーク
- [x] 活性化関数付きDenseレイヤー（ReLU、Sigmoid、Tanh）
- [x] Sequentialモデルアーキテクチャ
- [x] MSE損失とSGD最適化器
- [x] モデル保存・読み込み（バイナリ、コンフィグ形式）

### 進行中 🚧
- [ ] GPU加速サポート
- [ ] JSON モデル読み込み実装
- [ ] 高度エラーハンドリングと検証

### 計画中 📋
- [ ] 追加レイヤータイプ（Convolutional、LSTM、Dropout）
- [ ] 追加最適化器（Adam、RMSprop、AdaGrad）
- [ ] 高度損失関数（CrossEntropy、Huber）
- [ ] モデル量子化と最適化
- [ ] Pythonバインディング
- [ ] マルチスレッドサポート

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](../LICENSE)ファイルをご覧ください。

## 🙏 謝辞

- PyTorchやTensorFlowなどのモダンMLフレームワークからインスパイア
- モダンC++ベストプラクティスで構築
- 教育・研究目的で設計
