# GPU Testing Environment Setup

このドキュメントでは、CIでGPU環境のテストを設定する方法について説明します。

## 概要

MLLibのGPU実装は以下のテスト戦略を採用しています：

1. **CPU Fallback Testing**: CPUでGPUの代替動作をテスト
2. **GPU Simulation Testing**: CUDA環境をシミュレートしてテスト
3. **GPU Hardware Testing**: 実際のGPUハードウェアでテスト（オプション）

## CI設定

### 1. 自動実行されるテスト

`.github/workflows/gpu-ci.yml`により、以下のテストが自動実行されます：

- **GPU Fallback Test**: GPU無効環境での動作テスト
- **GPU Simulation Test**: CUDAシミュレーション環境でのテスト
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

### GPUテスト実行

```bash
# GPU有効でビルド
make clean
make WITH_CUDA=1

# GPUテスト実行
make unit-test WITH_CUDA=1
make integration-test WITH_CUDA=1

# GPU環境確認
nvidia-smi
nvcc --version
```

### CPU Fallbackテスト

```bash
# CUDA無効でビルド
make clean
make

# Fallbackテスト実行
make unit-test
make integration-test
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
make -d WITH_CUDA=1

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

1. **並列実行**: GPU/CPUテストを並列化
2. **キャッシュ活用**: CUDA toolkit installation cache
3. **Early exit**: 必須テスト失敗時の早期終了

### リソース管理

1. **GPU time limits**: 長時間実行の制限
2. **Memory monitoring**: GPUメモリ使用量監視
3. **Temperature checks**: GPU温度監視（可能な場合）

## まとめ

GPU CI環境は以下の利点を提供します：

- ✅ **包括的テスト**: GPU/CPU両方の実行パスをテスト
- ✅ **自動検証**: コード変更時の自動GPU機能検証
- ✅ **品質保証**: GPU実装の信頼性確保
- ✅ **パフォーマンス監視**: GPU性能の継続的監視
- ✅ **互換性確認**: 様々な環境での動作確認

この設定により、GPU実装の品質と信頼性が保証され、プロダクション環境での安定動作が期待できます。
