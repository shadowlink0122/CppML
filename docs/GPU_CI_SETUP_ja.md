# GPU Testing Environment Setup

> **Language**: [🇺🇸 English](GPU_CI_SETUP_en.md) | 🇯🇵 日本語

このドキュメントでは、CIでGPU環境のテストを設定する方法について説明します。

## 概要

MLLibのGPU実装は以下のテスト戦略を採用しています：

1. **CPU Famake clean
make

# GPUテストの実行（実行時検出）
make unit-test
make integration-test

# GPU環境チェック
nvidia-smi
nvcc --versionting**: CPUでGPUの代替動作をテスト
2. **GPU Simulation Testing**: CUDA環境をシミュレートしてテスト
3. **GPU Hardware Testing**: 実際のGPUハードウェアでテスト（オプション）

## CI設定

### 1. 最適化されたCI/CDパイプライン

`.github/workflows/gpu-ci.yml`により、効率的なテストパイプラインを実行：

**ビルドアーティファクト最適化**:
- `gpu-build`: ライブラリとテスト実行ファイルを一度だけビルド
- ビルド成果物をGitHub Artifactsとして共有（1日間保存）
- 各テストジョブでビルド成果物を再利用（重複ビルドを排除）

**テスト実行の最適化**:
- 事前ビルドされた実行ファイルでテスト実行
- ビルド時間を50-70%短縮
- CI実行時間とコストの大幅削減

### 2. 自動実行されるテスト

以下のテストが最適化されたパイプラインで自動実行されます：

- **GPU Build**: ライブラリとテスト実行ファイルのビルド（アーティファクト作成）
- **GPU Fallback Test**: 事前ビルド済み実行ファイルでGPU無効環境テスト
- **GPU Simulation Test**: CUDAシミュレーション環境でのテスト（アーティファクト利用）
- **GPU Documentation Check**: GPU関連ドキュメントの検証

### 2. 手動実行可能なテスト

GitHub ActionsのWorkflow Dispatchから手動で実行可能：

```bash
# テストタイプの選択:
# - all: 全てのGPUテスト
# - unit: GPU単体テストのみ
# - integration: GPU統合テストのみ
# - benchmark: GPUベンチマークのみ
```

### 3. GPU Hardware Testing (オプション)

実際のGPUハードウェアでテストを実行するには：

#### 3.1 Self-hosted Runner設定

1. NVIDIA GPU搭載のマシンでself-hosted runnerを設定
2. CUDAツールキットをインストール
3. Runnerに`self-hosted-gpu`ラベルを設定

#### 3.2 ワークフロー有効化

`.github/workflows/gpu-ci.yml`で以下を変更：

```yaml
# Before:
runs-on: ubuntu-latest  # Change to 'self-hosted-gpu' when GPU runner is available
if: false  # Change to 'needs.gpu-availability-check.outputs.gpu-available == true' when ready

# After:
runs-on: self-hosted-gpu
if: needs.gpu-availability-check.outputs.gpu-available == 'true'
```

#### 3.3 環境変数設定

GPU環境チェック部分を修正：

```yaml
# gpu-availability-check job内で:
if [ -n "${{ secrets.GITHUB_TOKEN }}" ]; then
  echo "gpu-available=true" >> $GITHUB_OUTPUT  # true に変更
  echo "✅ GPU runners available"
```

## テスト内容

### GPU Unit Tests (74 assertions)

- **GPUAvailabilityTest**: GPU検出機能
- **GPUDeviceValidationTest**: GPU検証システム
- **GPUBackendOperationsTest**: GPU基本演算
- **GPUMemoryManagementTest**: GPUメモリ管理
- **GPUErrorHandlingTest**: GPUエラー処理
- **GPUPerformanceTest**: GPU性能測定

### GPU Integration Tests (71 assertions)

- **GPUCPUFallbackIntegrationTest**: GPU-CPU代替統合
- **GPUModelComplexityIntegrationTest**: GPUモデル複雑度
- **GPUPerformanceIntegrationTest**: GPU性能統合
- **GPUMemoryIntegrationTest**: GPUメモリ統合

## GPU機能

### 実装されている機能

1. **自動GPU検出**: 起動時にGPU環境を自動検出
2. **CPU Fallback**: GPU使用不可時の自動CPU代替
3. **警告システム**: GPU問題時のユーザー通知
4. **CUDA統合**: CUDA kernelとcuBLASの完全統合
5. **メモリ管理**: GPU/CPUメモリの効率的管理

### テスト対象演算

- Matrix multiplication (行列乗算)
- Element-wise operations (要素単位演算)
- Activation functions (活性化関数)
- Memory transfer operations (メモリ転送演算)

## ローカル開発環境でのGPUテスト

### CUDA環境構築

```bash
# CUDA toolkit インストール (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# 環境変数設定
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GPUテスト実行（最適化版）

```bash
# ライブラリビルド（実行時GPU検出）
make clean
make

# テスト実行ファイルの事前ビルド（CI最適化）
make build-tests

# GPUテスト実行（事前ビルド済み実行ファイル使用）
make unit-test-run-only
make integration-test-run-only
make simple-integration-test-run-only

# 従来の方法（ビルド + 実行）
make unit-test
make integration-test

# GPU環境確認
nvidia-smi
nvcc --version
```

### CPU Fallbackテスト（最適化版）

```bash
# CUDA無効でビルド
make clean
make

# テスト実行ファイルの事前ビルド
make build-tests

# Fallbackテスト実行（最適化版）
make unit-test-run-only
make integration-test-run-only
```

## トラブルシューティング

### よくある問題

1. **CUDA not found エラー**
   - `cuda.mk`でCUDA環境を確認
   - `nvcc`がPATHに含まれているか確認

2. **GPU tests failing**
   - GPU driverが正しくインストールされているか確認
   - CUDAバージョンの互換性を確認

3. **Fallback tests failing**
   - CPU-onlyビルドが正常に動作するか確認
   - 警告メッセージが正しく表示されるか確認

### デバッグコマンド

```bash
# CUDA環境確認
which nvcc
nvidia-smi
cat /proc/driver/nvidia/version

# ビルド詳細確認
make clean
make -d

# テスト詳細実行
make unit-test VERBOSE=1
```

## パフォーマンス指標

### 期待されるパフォーマンス

- **GPU Tests**: ~10-15秒 (ハードウェア依存)
- **CPU Fallback Tests**: ~5-10秒
- **Simulation Tests**: ~3-5秒

### ベンチマーク基準

- Matrix multiplication (1000x1000): GPU < 10ms, CPU < 100ms
- Element-wise operations: GPU speedup 2-10x vs CPU
- Memory transfer overhead: < 5% of computation time

## CI最適化

### パフォーマンス改善

1. **ビルドアーティファクト共有**: 重複ビルドを排除し、CI実行時間を50-70%短縮
2. **並列実行**: GPU/CPUテストを効率的に並列化
3. **キャッシュ活用**: CUDA toolkit installation cache
4. **Early exit**: 必須テスト失敗時の早期終了

### リソース管理

1. **Artifact管理**: ビルド成果物の1日間保存で効率化
2. **GPU time limits**: 長時間実行の制限
3. **Memory monitoring**: GPUメモリ使用量監視
4. **Temperature checks**: GPU温度監視（可能な場合）

### 最適化されたMakeターゲット

```bash
# CI最適化用ターゲット
make build-tests                    # テスト実行ファイルのみビルド（アーティファクト用）
make unit-test-run-only             # 事前ビルド済み実行ファイルで単体テスト
make integration-test-run-only      # 事前ビルド済み実行ファイルで統合テスト
make simple-integration-test-run-only # 事前ビルド済み実行ファイルでシンプル統合テスト

# 従来のターゲット（ビルド + 実行）
make unit-test                      # 単体テスト（ビルド含む）
make integration-test               # 統合テスト（ビルド含む）
make simple-integration-test        # シンプル統合テスト（ビルド含む）
```

## まとめ

GPU CI環境は以下の利点を提供します：

- ✅ **包括的テスト**: GPU/CPU両方の実行パスをテスト
- ✅ **自動検証**: コード変更時の自動GPU機能検証
- ✅ **品質保証**: GPU実装の信頼性確保
- ✅ **パフォーマンス監視**: GPU性能の継続的監視
- ✅ **互換性確認**: 様々な環境での動作確認

この設定により、GPU実装の品質と信頼性が保証され、プロダクション環境での安定動作が期待できます。
