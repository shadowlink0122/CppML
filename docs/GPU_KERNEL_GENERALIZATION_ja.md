# GPU実装の汎用化設計

## 🎯 概要

従来のGPU実装では、各activation関数ごとに同じボイラープレートコードが重複していました。新しい汎用的な設計により、保守性、拡張性、開発効率が大幅に向上しました。

## 📋 従来の問題点

### ❌ **問題点**
```cpp
// 従来: 各関数で同じコードパターンの重複
void MetalBackend::relu(const double* input, double* output, size_t size) {
    // 100行以上の同じパターンのコード
    // - double→float変換
    // - バッファ作成
    // - カーネル文字列定義  
    // - コンパイル
    // - 実行
    // - float→double変換
}

void MetalBackend::sigmoid(const double* input, double* output, size_t size) {
    // 同じパターンのコード（95%重複）
    // カーネル文字列だけが違う
}

void MetalBackend::tanh_activation(const double* input, double* output, size_t size) {
    // 同じパターンのコード（95%重複）
    // カーネル文字列だけが違う
}
```

### 🔴 **維持管理の課題**
- 新しいactivation関数追加時に100行以上のコピペ
- バグ修正時に全ての関数を更新する必要
- パラメータ付き関数（LeakyReLU等）の実装が困難
- Metal固有のコードがactivation層に散在

## ✅ 新しい汎用設計

### 🏗️ **アーキテクチャ**

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│  MetalBackend::relu(), sigmoid(), etc.  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      ActivationKernelRegistry           │
│   • 関数登録・実行の統一インターフェース      │
│   • 数式からカーネル自動生成              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       GPUKernelManager                  │
│   • 汎用カーネル実行エンジン               │  
│   • バッファ管理の統一                   │
│   • Metal/CUDA/ROCm対応                │
└─────────────────────────────────────────┘
```

### 🎯 **核心的な改善**

#### 1. **宣言的な関数定義**
```cpp
// 新しい方法: 1行で関数定義
ActivationKernelRegistry::registerActivation({
    "leaky_relu", 
    "input > 0.0f ? input : alpha * input",  // GPU式だけ記述
    {"alpha"},                               // パラメータ名
    true                                     // パラメータ有無
});
```

#### 2. **自動カーネル生成**
```cpp
// 内部で自動生成されるMetal kernel
kernel void leaky_relu_kernel(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              device const float* alpha [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    float alpha = alpha[0];
    output[index] = input > 0.0f ? input : alpha * input;
}
```

#### 3. **統一されたAPI**
```cpp
// すべての関数で同じインターフェース
MetalBackend::relu(input, output, size);                    // パラメータなし
MetalBackend::leaky_relu(input, output, size, 0.01);        // パラメータあり
MetalBackend::elu(input, output, size, 1.0);               // パラメータあり

// カスタム関数も同じAPI
ActivationKernelRegistry::executeActivation("custom_func", input, output, size, {param1, param2});
```

## 🚀 利用方法

### 新しいActivation関数の追加

#### **Step 1: 関数登録**
```cpp
ActivationKernelRegistry::registerActivation({
    "mish",                                      // 関数名
    "input * tanh(log(1.0f + exp(input)))",     // GPU数式
    {},                                          // パラメータなし
    false                                        // パラメータフラグ
});
```

#### **Step 2: Backend APIに追加**
```cpp
// metal_backend.hpp
static void mish(const double* input, double* output, size_t size);

// metal_backend.mm  
void MetalBackend::mish(const double* input, double* output, size_t size) {
    ActivationKernelRegistry::executeActivation("mish", input, output, size);
}
```

#### **Step 3: 使用**
```cpp
// レイヤーで使用
MetalBackend::mish(input_data, output_data, size);
```

### パラメータ付き関数の例

```cpp
// Swish-Beta (パラメータ付き)
ActivationKernelRegistry::registerActivation({
    "swish_beta",
    "input / (1.0f + exp(-beta * input))",  // beta倍のSwish
    {"beta"},
    true
});

// 使用
MetalBackend::swish_beta(input, output, size, 1.5);  // beta=1.5
```

## 📊 メリット比較

| 項目 | 従来の実装 | 新しい汎用実装 |
|------|------------|----------------|
| **新関数追加** | 100+行のコピペ | 1-3行の定義 |
| **保守性** | 各関数を個別修正 | 中央集約管理 |
| **バグ修正** | 全関数で修正必要 | 一箇所で修正 |
| **パラメータ対応** | 手動実装 | 自動対応 |
| **コード重複** | 95%重複 | 0%重複 |
| **テスト工数** | 各関数個別テスト | 統一テスト |

## 🔧 実装詳細

### GPUKernelManager
- **役割**: 汎用的なGPUカーネル実行基盤
- **機能**: 
  - バッファ管理の統一
  - double⇔float変換の自動化
  - エラーハンドリングの標準化
  - Metal/CUDA/ROCm対応の抽象化

### ActivationKernelRegistry
- **役割**: Activation関数の管理・実行
- **機能**:
  - 数式からカーネルコード自動生成
  - パラメータ付き関数の自動対応
  - 型安全なAPI提供

### MetalBackend
- **役割**: ユーザー向けAPI
- **変更**: 重複コード削除、汎用システム使用

## 🎯 今後の拡張

### 1. **他のGPUバックエンド対応**
```cpp
// CUDA版も同じインターフェース
CUDAKernelManager::executeUnaryKernel("relu", input, output, size);
```

### 2. **複合関数サポート**
```cpp
// 複数ステップの関数
registerActivation({
    "gelu_precise",
    "0.5f * input * (1.0f + erf(input / 1.4142135623f))",  // 正確なGELU
    {},
    false
});
```

### 3. **最適化カーネル**
```cpp
// プラットフォーム特化最適化
registerOptimizedKernel("relu", "optimized_relu_metal.metal");
```

## 💡 設計原則

1. **DRY (Don't Repeat Yourself)**: コード重複の完全排除
2. **単一責任原則**: 各クラスが明確な役割を持つ
3. **拡張性**: 新機能追加が容易
4. **型安全性**: コンパイル時エラー検出
5. **性能**: ランタイムオーバーヘッド最小化

## 🧪 テスト戦略

### 統一テストフレームワーク
```cpp
template<typename ActivationFunc>
void test_activation_gpu(const std::string& name, ActivationFunc cpu_func) {
    // GPU vs CPU比較テスト
    // 数値精度テスト
    // パフォーマンステスト
}

// 使用例
test_activation_gpu("relu", [](double x) { return std::max(0.0, x); });
test_activation_gpu("sigmoid", [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
```

この汎用化により、MLLibのGPU実装は格段に保守しやすく、拡張しやすくなりました！
