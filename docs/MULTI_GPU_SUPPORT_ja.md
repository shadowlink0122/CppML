# マルチGPUサポート

CppMLは複数のGPUベンダーをサポートし、ユーザーの環境に応じて最適なGPUアクセラレーションを提供します。

## サポートされているGPUベンダー

### 1. NVIDIA GPU (CUDA)
- **API**: CUDA Runtime API、cuBLAS
- **サポート範囲**: GeForce、Quadro、Tesla、RTXシリーズ
- **要件**: 
  - CUDA Toolkit 11.0以上
  - NVIDIA Driver 450.80.02以上
  - Compute Capability 6.0以上推奨

### 2. AMD GPU (ROCm/HIP)
- **API**: ROCm、HIP、hipBLAS
- **サポート範囲**: Radeon Instinct、Radeon Pro、一部のRadeonシリーズ
- **要件**:
  - ROCm 4.0以上
  - サポート対象GPU (gfx900以上)
  - Linux環境推奨

### 3. Intel GPU (oneAPI)
- **API**: Intel oneAPI、SYCL、oneMKL
- **サポート範囲**: Intel Arc、Intel Iris Xe、Intel UHD Graphics
- **要件**:
  - Intel oneAPI Toolkit 2023.0以上
  - Intel GPU Driver (最新版推奨)
  - OpenCL 2.1以上サポート

### 4. Apple Silicon GPU (Metal)
- **API**: Metal Performance Shaders (MPS)
- **サポート範囲**: M1、M1 Pro、M1 Max、M1 Ultra、M2シリーズ
- **要件**:
  - macOS 12.0以上
  - Xcode 13.0以上
  - Metal 3サポート

## ライブラリの設計思想

### デフォルト全GPU対応
CppMLはライブラリとして設計されており、**デフォルトで全てのGPUベンダーをサポート**します：

```makefile
# デフォルトで有効化
CXXFLAGS += -DWITH_CUDA -DWITH_ROCM -DWITH_ONEAPI -DWITH_METAL
```

### 条件付きコンパイル
利用可能でないGPUライブラリがある場合、自動的にスタブ実装にフォールバックします：

```cpp
#ifdef WITH_CUDA
    // 実際のCUDA実装
#else
    // スタブ実装（CPU処理）
#endif
```

## ビルドオプション

### 基本ビルド（推奨）
```bash
# 全GPUサポートを有効にしてビルド
make clean && make
```

### 特定GPUの無効化
必要に応じて特定のGPUサポートを無効にできます：

```bash
# CUDAサポートを無効化
make DISABLE_CUDA=1

# 複数のGPUサポートを無効化
make DISABLE_CUDA=1 DISABLE_ROCM=1

# CPUのみでビルド
make DISABLE_CUDA=1 DISABLE_ROCM=1 DISABLE_ONEAPI=1 DISABLE_METAL=1
```

### GPU優先順位
複数のGPUが利用可能な場合の優先順位：
1. **NVIDIA CUDA** (最高優先度)
2. **AMD ROCm**
3. **Intel oneAPI**
4. **Apple Metal**

## 使用方法

### 基本的な使用例
```cpp
#include "MLLib.hpp"

int main() {
    MLLib::model::Sequential model;
    
    // GPUデバイスを指定（自動検出・選択）
    model.set_device(MLLib::DeviceType::GPU);
    
    // レイヤー構築
    model.add_layer(new MLLib::layer::Dense(784, 128));
    model.add_layer(new MLLib::layer::activation::ReLU());
    model.add_layer(new MLLib::layer::Dense(128, 10));
    
    // 学習実行（GPU自動利用）
    model.train(train_X, train_Y, loss, optimizer);
    
    return 0;
}
```

### GPU状況の確認
```cpp
// GPU利用可能性チェック
if (MLLib::Device::isGPUAvailable()) {
    printf("GPU available!\n");
    
    // 利用可能GPUの詳細表示
    auto gpus = MLLib::Device::detectGPUs();
    for (const auto& gpu : gpus) {
        printf("GPU: %s (%s)\n", gpu.name.c_str(), gpu.api_support.c_str());
    }
}
```

## 警告システム

### 自動警告表示
GPUが要求されたが利用できない場合、ライブラリが自動的に警告を表示します：

```
⚠️  WARNING: GPU device requested but no GPU found!
   Falling back to CPU device for computation.
   Possible causes:
   - No NVIDIA GPU installed
   - CUDA driver not installed or incompatible
   - GPU is being used by another process
```

### 成功メッセージ
GPUが正常に設定された場合：
```
✅ GPU device successfully configured
```

## 環境変数による制御

### 強制CPU実行
```bash
# テスト用にGPUを無効化
FORCE_CPU_ONLY=1 ./your_program
```

## パフォーマンスの最適化

### メモリ管理
- **統一メモリ**: CUDA Unified Memoryを使用（NVIDIA GPU）
- **ゼロコピー**: Apple MetalのManagedBufferを活用
- **非同期転送**: ROCm/HIPの非同期メモリ転送

### 並列実行
```cpp
// バッチ処理での並列実行
model.train(large_dataset, batch_size=256);  // GPU並列処理
```

## トラブルシューティング

### CUDA関連
```bash
# CUDA情報確認
nvidia-smi
nvcc --version

# CUDA環境変数設定
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### ROCm関連
```bash
# ROCm情報確認
rocm-smi
hipcc --version

# ROCm環境変数設定
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
```

### oneAPI関連
```bash
# oneAPI環境の初期化
source /opt/intel/oneapi/setvars.sh
sycl-ls  # SYCL デバイス一覧
```

### Metal関連
```bash
# Metal機能確認
system_profiler SPDisplaysDataType
```

## 開発者向け情報

### 新しいGPUベンダーの追加
1. `include/MLLib/backend/` にヘッダファイル追加
2. `src/MLLib/backend/gpu/` に実装ファイル追加
3. `Device::detectGPUs()` に検出ロジック追加
4. Makefileに条件コンパイル設定追加

### スタブ実装
GPU未対応環境でもコンパイルエラーにならないよう、全GPUバックエンドにスタブ実装が含まれています：

```cpp
#ifdef HIP_AVAILABLE
    // 実際のROCm実装
    hipMalloc(&device_ptr, size);
#else
    // スタブ実装（CPU処理）
    device_ptr = malloc(size);
#endif
```

## ライセンスと注意事項

- **CUDA**: NVIDIA CUDA Toolkit End User License Agreement
- **ROCm**: MIT License
- **oneAPI**: Intel oneAPI License
- **Metal**: Apple Developer License

各GPUベンダーのライセンスとドキュメントを必ず確認してください。

## サポートとコミュニティ

- **Issues**: [GitHub Issues](https://github.com/shadowlink0122/CppML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shadowlink0122/CppML/discussions)
- **Wiki**: [CppML Wiki](https://github.com/shadowlink0122/CppML/wiki)

---

**最終更新**: 2025年8月15日  
**対応バージョン**: CppML v1.0.0+
