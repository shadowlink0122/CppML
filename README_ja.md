# MLLib - C++ 機械学習ライブラリ

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)

> **言語**: [English](README.md) | [日本語](README_ja.md)

パフォーマンスと使いやすさを重視したモダンなC++17機械学習ライブラリです。

## ✨ 特徴

- **🧠 ニューラルネットワーク**: カスタマイズ可能なSequentialモデル
- **📊 レイヤー**: Dense（全結合）、ReLU、Sigmoid活性化関数
- **🎯 訓練**: MSE損失関数とSGD最適化器
- **💾 モデル I/O**: バイナリ、JSON、設定ファイル形式での保存・読み込み
- **📁 自動ディレクトリ作成**: `mkdir -p` 相当の機能
- **🔧 型安全性**: enum ベースの形式指定で信頼性向上
- **⚡ パフォーマンス**: NDArray バックエンドによる最適化されたC++17実装
- **🔄 クロスプラットフォーム**: Linux、macOS、Windows対応

## 🚀 クイックスタート

### 前提条件

- C++17対応コンパイラ (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+ (オプション)
- Make

### ビルド

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make all
```

### 使用方法

```cpp
#include "MLLib.hpp"

int main() {
    MLLib::initialize();
    
    // ここにMLコードを記述
    
    MLLib::cleanup();
    return 0;
}
```

## 🛠️ 開発

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

# テストを実行
make test

# クリーン
make clean
```

### 開発ツールのインストール

```bash
make install-tools
```

## 📚 ドキュメント

### APIコンポーネント

- **コア**: `MLLib/config.hpp`, `MLLib/ndarray.hpp`
- **データ**: 前処理、バッチ処理、読み込み
- **レイヤー**: 全結合、畳み込み、プーリング、活性化関数
- **モデル**: シーケンシャル、関数型、カスタム
- **オプティマイザー**: SGD、Adam
- **ユーティリティ**: 文字列、数学、I/O、システムユーティリティ

### サンプルコード

```cpp
#include "MLLib.hpp"

// シンプルなニューラルネットワークを作成
MLLib::Sequential model;
model.add(new MLLib::Dense(128, 64));
model.add(new MLLib::ReLU());
model.add(new MLLib::Dense(64, 10));
model.compile(MLLib::Adam(), MLLib::CrossEntropy());
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

## 📄 ライセンス

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
