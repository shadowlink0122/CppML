# MLLib ドキュメントインデックス

> **Language**: [🇺🇸 English](README.md) | 🇯🇵 日本語

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_passing-brightgreen.svg)](#testing)
[![GPU最適化](https://img.shields.io/badge/GPUカーネル削減-97%25-brightgreen.svg)](#gpu戦略)
[![モデルI/O汎用化](https://img.shields.io/badge/モデルI%2FO汎用化-92%25-green.svg)](#モデル-io-ガイド)

MLLib C++機械学習ライブラリの包括的なドキュメントコレクションへようこそ。

## 📚 メインドキュメント

### 🌐 言語版別完全ガイド

| 言語 | ファイル | 説明 |
|------|----------|------|
| 🇺🇸 **English** | [GUIDE_en.md](GUIDE_en.md) | Complete guide in English |
| 🇯🇵 **日本語** | [GUIDE_ja.md](GUIDE_ja.md) | 日本語での完全ガイド |
| 🏠 **プロジェクト** | [../README.md](../README.md) | プロジェクトのメインREADME |

## 🔧 技術ドキュメント

### 💾 モデル I/O ガイド

| 言語 | ファイル | 説明 |
|------|----------|------|
| 🇺🇸 **English** | [MODEL_IO_en.md](MODEL_IO_en.md) | Complete model serialization guide |
| 🇯🇵 **日本語** | [MODEL_IO_ja.md](MODEL_IO_ja.md) | モデル保存・読み込みの完全ガイド |

### 🧪 テストガイド

| 言語 | ファイル | 説明 |
|------|----------|------|
| 🇺🇸 **English** | [TESTING_en.md](TESTING_en.md) | Testing framework documentation |
| 🇯🇵 **日本語** | [TESTING_ja.md](TESTING_ja.md) | テストフレームワークの説明 |

### 🚀 GPU ドキュメント

| 言語 | ファイル | 説明 |
|------|----------|------|
| 🇺🇸 **English** | [GPU_STRATEGY_en.md](GPU_STRATEGY_en.md) | GPU backend strategy and roadmap |
| 🇯🇵 **日本語** | [GPU_STRATEGY_ja.md](GPU_STRATEGY_ja.md) | GPU バックエンド戦略とロードマップ |
| 🇺🇸 **English** | [MULTI_GPU_SUPPORT_en.md](MULTI_GPU_SUPPORT_en.md) | Multi-GPU vendor support |
| 🇯🇵 **日本語** | [MULTI_GPU_SUPPORT_ja.md](MULTI_GPU_SUPPORT_ja.md) | マルチGPUベンダーサポート |
| 🇺🇸 **English** | [GPU_CI_SETUP_en.md](GPU_CI_SETUP_en.md) | GPU testing environment setup |
| 🇯🇵 **日本語** | [GPU_CI_SETUP_ja.md](GPU_CI_SETUP_ja.md) | GPU テスト環境セットアップ |
| 🇺🇸 **English** | [GPU_DETECTION_GUIDE_en.md](GPU_DETECTION_GUIDE_en.md) | GPU detection verification guide |
| 🇯🇵 **日本語** | [GPU_DETECTION_GUIDE_ja.md](GPU_DETECTION_GUIDE_ja.md) | GPU検出機能の検証ガイド |

## 🚀 クイックアクセス

### 📖 すぐに始める

- **🏃‍♂️ [クイックスタート（日本語）](GUIDE_ja.md#-クイックスタート)** - 5分で始めるMLLib
- **🏃‍♀️ [Quick Start (English)](GUIDE_en.md#-quick-start)** - Get started with MLLib in 5 minutes
- **🔧 [ビルド手順](GUIDE_ja.md#ビルドとテスト)** - プロジェクトのビルド方法
- **🎯 [基本的な使用例](GUIDE_ja.md#基本的な使用方法)** - XORネットワークの実装例

### 🧪 テスト関連

- **📊 [テスト実行方法](TESTING_ja.md)** - 単体・統合テストの実行
- **⏱️ [実行時間監視](TESTING_ja.md#実行時間監視)** - パフォーマンス測定機能
- **✅ [テストカバレッジ](TESTING_ja.md#テストカバレッジ)** - 21/21テスト通過状況

### 💾 モデル操作

- **💿 [モデル保存](MODEL_IO_ja.md#モデル保存)** - バイナリ、JSON、コンフィグ形式
- **📥 [モデル読み込み](MODEL_IO_ja.md#モデル読み込み)** - 訓練済みモデルの利用
- **🔒 [型安全性](MODEL_IO_ja.md#型安全性)** - enum形式による安全な操作

### 🚀 GPU サポート

- **⚡ [GPU戦略](GPU_STRATEGY_ja.md)** - 全GPUベンダー対応計画
- **🔧 [マルチGPUサポート](MULTI_GPU_SUPPORT_ja.md)** - NVIDIA、AMD、Intel、Apple GPU対応
- **🧪 [GPU CI セットアップ](GPU_CI_SETUP_ja.md)** - 自動GPU テスト構成
- **🔍 [GPU検出ガイド](GPU_DETECTION_GUIDE_ja.md)** - GPU検出機能の検証方法

## 🏗️ アーキテクチャ

```
MLLib/
├── NDArray          # 多次元配列（テンソル操作）
├── Device           # CPU/GPUデバイス管理
├── Layer/           # ニューラルネットワークレイヤー
│   ├── Dense        # 全結合層
│   └── Activation/  # 活性化関数（ReLU、Sigmoid、Tanh）
├── Loss/            # 損失関数（MSE）
├── Optimizer/       # 最適化アルゴリズム（SGD）
└── Model/           # モデル定義とI/O
    ├── Sequential   # Sequentialモデルアーキテクチャ
    └── ModelIO      # モデルシリアライゼーション
```

## 📁 ドキュメント構造

```
docs/
├── README.md           # このファイル（ドキュメントインデックス）
├── README_en.md        # 英語版完全ガイド
├── README_ja_guide_backup.md        # 日本語版完全ガイド
├── MODEL_IO_en.md      # モデルI/Oガイド（英語）
├── MODEL_IO_ja.md      # モデルI/Oガイド（日本語）
├── TESTING_en.md       # テストガイド（英語）
└── TESTING_ja.md       # テストガイド（日本語）
```

## 🎯 使用例とサンプル

### 💡 基本例

```cpp
// XORニューラルネットワークの簡単な例
#include "MLLib.hpp"
using namespace MLLib;

model::Sequential model;
model.add(std::make_shared<layer::Dense>(2, 4));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(4, 1));
model.add(std::make_shared<layer::activation::Sigmoid>());

// 詳細は README_ja_guide_backup.md を参照
```

### 🔧 実行可能サンプル

```bash
# プロジェクトルートで実行
make samples               # 全サンプルプログラムをビルド
make run-sample            # 利用可能なサンプル一覧を表示
make xor                   # XORネットワーク例（エイリアス）
make device-detection      # GPUデバイス検出サンプル（エイリアス）
make gpu-vendor-detection  # GPUベンダー検出サンプル（エイリアス）
make unit-test             # 全単体テスト実行
```

## 🛠️ 開発者向け情報

### 📋 要件

- **コンパイラ**: C++17対応（GCC 7+、Clang 5+、MSVC 2017+）
- **ビルドシステム**: Make
- **プラットフォーム**: Linux、macOS、Windows

### 🔍 品質保証

- **🧪 テスト**: 21/21テスト通過（実行時間監視付き）
- **📊 カバレッジ**: 包括的単体・統合テスト
- **🔧 静的解析**: cppcheck、clang-tidy対応
- **📝 コードスタイル**: K&Rスタイル

## 🤝 コントリビューション

1. **📖 [コントリビューションガイド](README_ja_guide_backup.md#-コントリビューション)** - 参加方法
2. **🎨 [コードスタイル](README_ja_guide_backup.md#コードスタイル)** - K&Rフォーマット
3. **✅ [品質チェック](README_ja_guide_backup.md#コード品質)** - リンティングとテスト

## 📧 サポート

- **🐛 [Issues](https://github.com/shadowlink0122/CppML/issues)** - バグ報告・機能要望
- **💬 [Discussions](https://github.com/shadowlink0122/CppML/discussions)** - 質問・ディスカッション
- **📋 [Projects](https://github.com/shadowlink0122/CppML/projects)** - 開発ロードマップ

## 📄 ライセンス

このプロジェクトは[商業利用制限付きBSD 3-Clauseライセンス](../LICENSE)の下で公開されています。

---

**💡 ヒント**: 初めての方は[日本語版完全ガイド](GUIDE_ja.md)または[English Complete Guide](GUIDE_en.md)から始めることをお勧めします。
