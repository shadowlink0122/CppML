# MLLib ドキュメント

MLLib machine learning library の包括的なドキュメント集です。

## 📚 ドキュメント一覧

### 🔧 技術ドキュメント
- 🇺🇸 [Model I/O (English)](MODEL_IO_en.md)
- 🇯🇵 [モデル I/O (日本語)](MODEL_IO_ja.md)

### 🧪 テストドキュメント
- 🇺🇸 [Testing Guide (English)](TESTING_en.md)
- 🇯🇵 [テストガイド (日本語)](TESTING_ja.md)

### 📖 言語版
- 🇺🇸 [Documentation (English)](README_en.md)
- 🇯🇵 [ドキュメント (日本語)](README_ja.md)

## 📖 概要

MLLibは、C++17で書かれた機械学習ライブラリです。包括的なテストシステムと実行時間監視機能を備えています。

### ✨ 主要機能

- **ニューラルネットワーク**: Sequential モデルによる多層パーセプトロン
- **レイヤー**: Dense（全結合）、ReLU、Sigmoid、Tanh活性化
- **損失関数**: Mean Squared Error (MSE)
- **最適化器**: Stochastic Gradient Descent (SGD)
- **モデル I/O**: バイナリ、JSON、設定ファイル形式でのモデル保存・読み込み
- **自動ディレクトリ作成**: `mkdir -p` 相当の機能
- **型安全性**: enum ベースの形式指定システム
- **テストフレームワーク**: 実行時間監視付き包括的テストシステム
- **パフォーマンス監視**: マイクロ秒精度の実行時間測定

### 🧪 テストシステム

MLLibは包括的なテストシステムを提供します：

```bash
# 単体テスト（21/21通過、実行時間表示）
make unit-test

# 出力例：
# ✅ ConfigConstantsTest PASSED (10 assertions, 0.03ms)
# ✅ NDArrayMatmulTest PASSED (11 assertions, 0.02ms)
# Total test execution time: 0.45ms
# Total suite time (including overhead): 0.89ms

# 統合テスト
make integration-test

# 全テスト実行
make test
```

#### テストカバレッジ

- **Config Module (3/3)**: ライブラリ設定と定数
- **NDArray Module (6/6)**: 多次元配列操作
- **Dense Layer (4/4)**: 全結合層機能
- **Activation Functions (7/7)**: ReLU、Sigmoid、Tanhとエラーハンドリング
- **Sequential Model (1/1)**: モデル構築と予測

#### 統合テスト内容

- **XOR問題**: エンドツーエンドの訓練と予測
- **モデルI/O**: 保存・読み込みワークフロー
- **マルチレイヤー**: 複雑なアーキテクチャ検証
- **パフォーマンス**: 安定性とパフォーマンステスト

### 🏗️ アーキテクチャ

```
MLLib/
├── NDArray          # 多次元配列（テンソル操作）
├── Device           # CPU/GPU デバイス管理
├── Layer/           # ニューラルネットワークレイヤー
│   ├── Dense        # 全結合層
│   └── Activation/  # 活性化関数
├── Loss/            # 損失関数
├── Optimizer/       # 最適化アルゴリズム
└── Model/           # モデル定義とI/O
```

### 🎯 使用例

```cpp
#include "MLLib/MLLib.hpp"

using namespace MLLib;

// XORネットワークの例（エポック保存付き）
auto model = std::make_unique<model::Sequential>();
model->add(std::make_shared<layer::Dense>(2, 4));
model->add(std::make_shared<layer::activation::ReLU>());  
model->add(std::make_shared<layer::Dense>(4, 1));
model->add(std::make_shared<layer::activation::Sigmoid>());

// 訓練データ
std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

// 損失関数と最適化器
loss::MSE loss_fn;
optimizer::SGD optimizer(0.1);

// エポック保存コールバック
auto save_callback = [&model](int epoch, double loss) {
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        
        // 型安全なenum形式でモデル保存
        std::string path = "models/training_xor/epoch_" + std::to_string(epoch);
        model::ModelIO::save_model(*model, path + ".bin", model::ModelFormat::BINARY);
        model::ModelIO::save_model(*model, path + ".json", model::ModelFormat::JSON);
    }
};

// 訓練
model->train(X, Y, loss_fn, optimizer, save_callback, 1000);

// 最終モデル保存（3つの形式すべて）
model::ModelIO::save_model(*model, "models/final.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(*model, "models/final.json", model::ModelFormat::JSON);
model::ModelIO::save_model(*model, "models/final.config", model::ModelFormat::CONFIG);
```

## 🔧 ビルドとテスト

```bash
# ライブラリのビルド
make

# サンプルのビルドと実行
make xor                    # XORネットワークの実行
make model-format-test      # モデルI/O形式テスト

# コード品質チェック
make fmt                    # コードフォーマット
make lint                   # 静的解析
make check                  # cppcheck実行
```

## 📁 プロジェクト構造

```
CppML/
├── include/MLLib/         # ヘッダーファイル
├── src/MLLib/            # 実装ファイル
├── samples/              # サンプルコード
├── tests/                # テストコード
├── docs/                 # ドキュメント
├── .github/workflows/    # CI/CD設定
└── README.md             # プロジェクト概要
```

## 🚀 始め方

1. **クローン**: `git clone https://github.com/shadowlink0122/CppML.git`
2. **ビルド**: `make`
3. **テスト**: `make test`
4. **ドキュメント**: [モデルI/O](MODEL_IO_ja.md)を参照

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🤝 コントリビューション

プルリクエスト、イシュー報告、機能提案を歓迎します。

---

**Note**: このライブラリは教育目的および研究目的で開発されています。本番環境での使用前に十分なテストを行ってください。
