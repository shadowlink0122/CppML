# GPU Detection Verification Guide

このガイドは、MLLibのGPU検出機能が異なるMac環境で正しく動作することを確認するためのものです。

## 期待される結果

### Apple Silicon Mac (M1/M2/M3)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: Apple
    Name: Apple M1 GPU (または M2/M3)
    Memory: 12288 MB (統合メモリ - システムRAMの75%)
    API Support: Metal

Primary GPU vendor: Apple
```

### Intel Mac + AMD GPU (例: Radeon Pro 5700)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: AMD
    Name: AMD Radeon Pro 5700
    Memory: 8192 MB (8GB 専用VRAM)
    API Support: OpenCL/Metal

Primary GPU vendor: AMD
```

### Intel Mac + NVIDIA GPU (例: GeForce RTX 3080)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: NVIDIA
    Name: NVIDIA GeForce RTX 3080
    Memory: 10240 MB (10GB 専用VRAM)
    API Support: CUDA (CUDAがインストール済みの場合)
    または API Support: OpenCL/Metal (CUDAなしの場合)

Primary GPU vendor: NVIDIA
```

### Intel Mac + Intel内蔵GPU (例: Iris Plus Graphics)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: Intel
    Name: Intel Iris Plus Graphics
    Memory: 4096 MB (システムRAMの25%を共有)
    API Support: oneAPI/OpenCL

Primary GPU vendor: Intel
```

## 重要な注意点

### GPUメモリの検出方法
MLLibでは、GPU種類に応じて異なるメモリ検出方式を使用します：

#### Apple Silicon (M1/M2/M3)
- **統合メモリアーキテクチャ**: CPUとGPUがシステムメモリを共有
- **検出方法**: `sysctl hw.memsize` でシステムメモリを取得
- **GPU利用可能メモリ**: システムメモリの75%として計算
- **利点**: 大容量メモリへのアクセス、メモリコピーオーバーヘッドなし

#### AMD/NVIDIA 離散GPU
- **専用VRAM**: GPU専用のグラフィックスメモリ
- **検出方法**: `system_profiler SPDisplaysDataType` でVRAM情報を解析
- **表示例**: "VRAM (Total): 8 GB" → 8192 MB として検出
- **利点**: 高帯域幅、CPU処理との独立性

#### Intel統合GPU
- **共有メモリアーキテクチャ**: システムメモリの一部をGPUが利用
- **検出方法**: システムメモリの25%として保守的に計算
- **動的割り当て**: 実際の使用時はより多くのメモリも利用可能

### MetalとGPUベンダーの区別
- **Metal API サポート ≠ Apple GPUベンダー**
- Intel Mac + AMD GPU環境では：
  - ✅ **ベンダー**: AMD (正しい - 実際のハードウェアベンダー)
  - ✅ **API Support**: OpenCL/Metal (正しい - MacでのAPI支援)
  - ❌ **ベンダー**: Apple (間違い - Apple GPUではない)

### よくある混同
```
誤解: "MetalサポートがあるからApple GPU"
正解: "MacではAMD/NVIDIA GPUでもMetalが使える"

誤解: "API Support: Metal → ベンダー: Apple"
正解: "AMD GPU + API Support: OpenCL/Metal → ベンダー: AMD"
```

## 検証手順

### 1. システム情報の確認
```bash
# アーキテクチャ確認
sysctl hw.optional.arm64
# 1 = Apple Silicon, 0 または存在しない = Intel Mac

# GPU情報確認
system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model"
```

### 2. MLLib GPU検出の実行
```bash
cd CppML
make clean && make samples
make run-sample SAMPLE=device_detection
```

### 3. 詳細デバッグ情報（必要に応じて）
```bash
# 詳細テストスクリプト実行
./test_gpu_detection.sh
```

## 報告が必要な問題

以下の場合は問題報告をお願いします：

### ❌ 間違った検出例
- Intel Mac + AMD Radeon Pro → ベンダー: Apple と表示
- Intel Mac + NVIDIA RTX → ベンダー: Apple と表示
- Apple Silicon Mac → ベンダー: AMD/NVIDIA と表示

### ✅ 正常な検出例
- Intel Mac + AMD Radeon Pro → ベンダー: AMD, API Support: OpenCL/Metal
- Intel Mac + NVIDIA RTX → ベンダー: NVIDIA, API Support: CUDA/OpenCL/Metal
- Apple Silicon Mac → ベンダー: Apple, API Support: Metal

## トラブルシューティング

### Intel Mac + 離散GPUで "Apple" と表示される場合

1. system_profilerの出力を確認：
```bash
system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model"
```

2. 期待される出力例 (AMD GPU):
```
Chipset Model: AMD Radeon Pro 5700
```

3. 期待される出力例 (NVIDIA GPU):
```
Chipset Model: NVIDIA GeForce RTX 3080
```

4. 実際の検出結果と比較してください

### システム情報収集
問題報告時は以下の情報も含めてください：
```bash
# システム情報
uname -a
sysctl hw.model
sysctl hw.optional.arm64
system_profiler SPDisplaysDataType | grep -A 10 "Chipset Model\|Metal Support"

# MLLib検出結果
make run-sample SAMPLE=device_detection
```

## 期待される動作の技術的詳細

### GPU検出の優先順位
1. **NVIDIA GPU** (離散GPU、高性能)
2. **AMD GPU** (離散GPU、高性能)
3. **Apple GPU** (Apple Silicon統合GPU)
4. **Intel GPU** (Intel統合GPU、省電力)

### API支援の決定
- **CUDA**: NVIDIA GPU + CUDAドライバーがインストール済み
- **ROCm**: AMD GPU + ROCmがインストール済み
- **Metal**: macOSで利用可能な全GPU (AMD, NVIDIA, Apple, Intel)
- **OpenCL**: ほぼ全てのGPU

この設計により、Intel Mac + AMD GPU環境では：
- ベンダー: AMD（正しいハードウェアベンダー）
- API Support: OpenCL/Metal（macOSで利用可能なAPI）
となります。
