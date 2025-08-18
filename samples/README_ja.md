# MLLib サンプルプログラム

> **Language**: [🇺🇸 English](README.md) | 🇯🇵 日本語

このディレクトリには、MLLib機械学習ライブラリの機能を実演するサンプルプログラムが含まれています。各サンプルは、基本的なデバイス検出から高度なGPU加速トレーニングまで、ライブラリの様々な側面を紹介しています。

## ディレクトリ構造

```
samples/
├── gpu/          # GPU関連サンプル（デバイス検出、パフォーマンステスト）
├── nn/           # ニューラルネットワークサンプル（トレーニング、推論）
├── README.md     # English版
└── README_ja.md  # このファイル
```

## 利用可能なサンプル

### GPUサンプル (`gpu/`)

#### 1. デバイス検出 (`device_detection.cpp`)

**目的**: GPU検出とデバイス管理機能の実演。

**機能**:
- 異なるベンダー（NVIDIA、AMD、Intel、Apple）のGPU検出
- GPU仕様の表示（メモリ、計算能力、APIサポート）
- 主要GPUベンダー検出のテスト
- ベンダー可用性の確認
- デバイス設定と検証のデモンストレーション

**使用方法**:
```bash
make device-detection
# または
make run-sample SAMPLE=gpu/device_detection
```

#### 2. GPU使用量テスト (`gpu_usage_test.cpp`)

**目的**: 行列演算でGPU vs CPUのパフォーマンスをベンチマーク。

**機能**:
- 異なるサイズでの行列積パフォーマンステスト
- GPUとCPUの実行時間比較
- パフォーマンス速度向上率の計算
- GPUとCPU結果間の数値精度検証

**使用方法**:
```bash
make run-sample SAMPLE=gpu/gpu_usage_test
```

#### 3. GPU検出デバッグ (`debug_gpu_detection.cpp`)

**目的**: 包括的なGPU検出デバッグとシステム情報表示。

**機能**:
- 詳細なGPUベンダー検出
- システムプロファイラー統合
- ハードウェア互換性チェック
- トラブルシューティング用デバッグ出力

**使用方法**:
```bash
make gpu-vendor-detection
# または
make run-sample SAMPLE=gpu/debug_gpu_detection
```

#### 4. 軽量GPU検出 (`debug_gpu_detection_light.cpp`)

**目的**: 迅速なシステムチェック用の軽量GPU検出。

**使用方法**:
```bash
make run-sample SAMPLE=gpu/debug_gpu_detection_light
```

#### 5. GPUシナリオテスト (`test_gpu_scenarios.cpp`, `verify_gpu_scenarios.cpp`)

**目的**: 様々なGPU使用シナリオのテストと検証。

**使用方法**:
```bash
make run-sample SAMPLE=gpu/test_gpu_scenarios
make run-sample SAMPLE=gpu/verify_gpu_scenarios
```

### ニューラルネットワークサンプル (`nn/`)

#### 1. XORニューラルネットワーク (`xor.cpp`)

**目的**: XOR問題による完全な機械学習ワークフローのデモンストレーション。

**機能**:
- Denseレイヤーと活性化関数によるニューラルネットワーク構築
- トレーニングでのGPU加速使用
- 進行状況監視のためのトレーニングコールバック実装
- トレーニング中のモデルチェックポイント保存（10エポック毎）
- 複数モデル形式対応（Binary、JSON、Config）

**ネットワーク構造**:
```
入力層:  2ニューロン（XOR入力）
隠れ層: 4ニューロン + ReLU活性化
出力層: 1ニューロン + Sigmoid活性化
```

**使用方法**:
```bash
make xor
# または
make run-sample SAMPLE=nn/xor
```
## サンプルのビルドと実行

### 前提条件

1. **最初にMLLibライブラリをビルド**:
   ```bash
   make all
   ```

2. **GPUサポートが利用可能か確認**（GPU サンプルをテストする場合）:
   ```bash
   make gpu-check
   ```

### 全サンプルをビルド

```bash
make samples
```

これにより全サンプルがビルドされ、以下に実行ファイルが配置されます：
- `build/samples/gpu/` - GPU関連サンプル
- `build/samples/nn/` - ニューラルネットワークサンプル

### 個別サンプルの実行

```bash
# エイリアスを使用
make xor                    # XORニューラルネットワークサンプル実行
make device-detection       # デバイス検出サンプル実行
make gpu-vendor-detection   # GPUベンダー検出サンプル実行

# run-sampleとフルパスを使用
make run-sample SAMPLE=nn/xor
make run-sample SAMPLE=gpu/device_detection
make run-sample SAMPLE=gpu/gpu_usage_test
```

### 利用可能サンプル一覧表示

```bash
make run-sample
```

出力：
```
Usage: make run-sample SAMPLE=<sample_name>
       make run-sample SAMPLE=<subdir>/<sample_name>
Available samples:
  gpu/:
    gpu/debug_gpu_detection
    gpu/debug_gpu_detection_light
    gpu/device_detection
    gpu/gpu_usage_test
    gpu/test_gpu_scenarios
    gpu/verify_gpu_scenarios
  nn/:
    nn/xor
```

## GPUバックエンドサポート

サンプルは利用可能な最適なGPUバックエンドを自動検出・使用します：

| プラットフォーム | 主要バックエンド | フォールバック |
|----------|----------------|----------|
| **Apple Silicon (M1/M2/M3)** | Metal Performance Shaders | CPU |
| **NVIDIA GPU** | CUDA | CPU |
| **AMD GPU** | ROCm/HIP | CPU |
| **Intel GPU** | oneAPI/SYCL | CPU |

## サンプル出力例

### デバイス検出出力
```
=== MLLib Device Detection Sample ===
--- GPU Detection ---
Detected 1 GPU(s):
  GPU 1:
    Vendor: Apple
    Name: Apple M1 GPU
    Memory: 12288 MB
    Compute Capable: Yes
    API Support: Metal

--- Primary GPU Vendor ---
Primary GPU vendor: Apple
```

### XORトレーニング出力
```
✅ GPU device successfully configured
Epoch 0 loss: 0.241477
Model saved at epoch 0 to samples/training_xor/epoch_0
Epoch 10 loss: 0.238898
...
0,0 => 0.468069  # 0.0に近づくべき
0,1 => 0.468069  # 1.0に近づくべき
1,0 => 0.648697  # 1.0に近づくべき
1,1 => 0.468069  # 0.0に近づくべき
```

## パフォーマンス期待値

### デバイス検出
- **実行時間**: 1秒未満
- **メモリ使用量**: 最小限
- **目的**: 迅速なハードウェア能力評価

### GPUパフォーマンステスト
- **小行列 (256x256)**: GPUオーバーヘッドによりCPUの方が高速
- **中行列 (512x512)**: パフォーマンス均衡
- **大行列 (1024x1024以上)**: GPUが2-10倍の高速化を示す
- **メモリ使用量**: 行列サイズに比例 (O(n²))

### XORトレーニング
- **トレーニング時間**: 1-10秒 (150エポック)
- **収束**: 通常 < 0.01 lossを達成
- **GPU高速化**: CPUより2-5倍 (ハードウェアによる)
- **メモリ使用量**: 最小限（小ネットワーク）

## 出力ファイル

### XORトレーニングファイル (`samples/training_xor/`)
- **バイナリモデル** (`.bin`): コンパクトで高速読み込み形式
- **JSONモデル** (`.json`): 人間が読める、デバッグ可能形式
- **設定ファイル** (`.config`): モデルアーキテクチャ設定

## トラブルシューティング

### よくある問題

1. **GPU が検出されない**:
   ```bash
   make run-sample SAMPLE=gpu/device_detection
   ```

2. **ビルド失敗**:
   ```bash
   make clean && make all && make samples
   ```

3. **GPUパフォーマンス問題**:
   ```bash
   make run-sample SAMPLE=gpu/gpu_usage_test
   ```

### デバッグ情報

```bash
# ライブラリビルド確認
make all

# GPU機能チェック
make gpu-check

# 利用可能サンプル一覧
make run-sample
```

## 次のステップ

これらのサンプルを探索した後：

1. **ドキュメントを読む** `docs/` ディレクトリ内
2. **テストスイートを実行** `make test`
3. **GPU統合テストを探索** `make gpu-integration-test`
4. **オートエンコーダーの例を実装** (将来の更新で追加予定)

詳細情報は以下を参照：
- [GPU戦略ドキュメント](../docs/GPU_STRATEGY_ja.md)
- [テストドキュメント](../docs/TESTING_ja.md)
- [モデルI/Oドキュメント](../docs/MODEL_IO_ja.md)
```bash
make samples
./build/samples/xor
```

**学習出力**:
学習プロセスは`samples/training_xor/`ディレクトリにモデルを保存します：
- `epoch_X.bin` / `epoch_X.json`: 10エポックごとに保存されるモデル
- `final_model.*`: 複数形式での最終学習済みモデル

**期待される結果**:
```
0,0 => ~0.0  (False XOR False = False)
0,1 => ~1.0  (False XOR True = True)
1,0 => ~1.0  (True XOR False = True)  
1,1 => ~0.0  (True XOR True = False)
```

## サンプルのビルドと実行

### 前提条件

1. **MLLibライブラリを最初にビルド**:
   ```bash
   make all
   ```

2. **GPU サポートが利用可能であることを確認**（GPU サンプルをテストする場合）:
   ```bash
   make gpu-check
   ```

### 全サンプルのビルド

```bash
make samples
```

これにより、samplesディレクトリ内のすべての`.cpp`ファイルがビルドされ、実行ファイルが`build/samples/`に配置されます。

### 個別サンプルの実行

```bash
# 特定のサンプル実行
make run-sample SAMPLE=device_detection
make run-sample SAMPLE=gpu_usage_test
make run-sample SAMPLE=xor

# または直接実行
./build/samples/device_detection
./build/samples/gpu_usage_test
./build/samples/xor
```

### 利用可能サンプルの一覧表示

```bash
make run-sample
```

## GPU バックエンドサポート

サンプルは利用可能な最適なGPU バックエンドを自動検出して使用します：

| プラットフォーム | 主要バックエンド | フォールバック |
|----------|----------------|----------|
| **Apple Silicon (M1/M2/M3)** | Metal Performance Shaders | CPU |
| **NVIDIA GPUs** | CUDA | CPU |
| **AMD GPUs** | ROCm/HIP | CPU |
| **Intel GPUs** | oneAPI/SYCL | CPU |

## サンプル出力ファイル

### XOR 学習出力 (`training_xor/`)

XORサンプルは学習成果物を生成します：

- **バイナリモデル** (`.bin`): コンパクトで高速読み込み形式
- **JSONモデル** (`.json`): 人間が読めるデバッグ可能形式  
- **設定ファイル** (`.config`): モデル構造設定

これらのファイルは以下の用途に使用できます：
- チェックポイントからの学習再開
- 学習進捗の分析
- 他のアプリケーションでの事前学習済みモデル読み込み

## パフォーマンス期待値

### デバイス検出
- **実行時間**: 1秒未満
- **メモリ使用量**: 最小限
- **目的**: 迅速なハードウェア能力評価

### GPU パフォーマンステスト
- **小行列 (256x256)**: GPUオーバーヘッドによりCPUが高速な場合が多い
- **中行列 (512x512)**: パフォーマンスが拮抗
- **大行列 (1024x1024以上)**: GPUが2-10倍の速度向上を示す
- **メモリ使用量**: 行列サイズと共にスケール（O(n²)）

### XOR 学習
- **学習時間**: 1-10秒（150エポック）
- **収束**: 通常0.01未満の損失を達成
- **GPU 速度向上**: CPU比で2-5倍（ハードウェアに依存）
- **メモリ使用量**: 最小限（小規模ネットワーク）

## トラブルシューティング

### 一般的な問題

1. **GPU が検出されない**:
   ```
   No GPUs detected
   GPU Available: No
   ```
   **解決法**: GPUドライバーとハードウェア互換性を確認。

2. **GPU パフォーマンスがCPUより遅い**:
   ```
   CPU is 1.5x faster than GPU
   ```
   **解決法**: 小行列では正常です。より大きなサイズを試すか、GPUバックエンド選択を確認。

3. **ビルドエラー**:
   ```
   ❌ Some samples failed to build
   ```
   **解決法**: 最初に`make all`でMLLibライブラリがビルドされていることを確認。

4. **Metal 精度差**:
   ```
   Max difference: 0.000123
   ```
   **解決法**: これは期待される動作 - MetalはFloat32、CPUはdouble64を使用。

### デバッグ情報

追加のデバッグには以下を確認：
```bash
# ライブラリビルドの確認
make all

# GPU能力の確認  
make gpu-check

# 詳細なビルド出力の表示
make samples VERBOSE=1
```

## プロジェクトへの統合

これらのサンプルは、あなた自身のプロジェクトにMLLibを統合するためのテンプレートとして機能します：

1. **関連するサンプルコードをコピー**して開始点として使用
2. **ネットワーク構造を変更**してあなたの用途に適用
3. **データ読み込みを適応**させてあなたのデータセットに対応  
4. **学習コールバックをカスタマイズ**してあなたの監視ニーズに対応
5. **適切なモデル形式を選択**してあなたのデプロイメントに対応

## 次のステップ

これらのサンプルを探索した後：

1. **ドキュメントを読む**（`docs/`ディレクトリ）
2. **テストスイートを実行**（`make test`）
3. **GPU統合テストを探索**（`make gpu-integration-test`）
4. **高度な例を確認**（testディレクトリ内）

詳細情報については以下を参照：
- [GPU戦略ドキュメント](../docs/GPU_STRATEGY_ja.md)
- [テストドキュメント](../docs/TESTING_ja.md)
- [モデルI/Oドキュメント](../docs/MODEL_IO_ja.md)
