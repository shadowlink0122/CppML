# MLLib Testing Documentation

> **Language**: [🇺🇸 English](TESTING_en.md) | 🇯🇵 日本語

MLLibの包括的なテストシステムのドキュメントです。

## 🧪 テスト概要

MLLibは高品質なコードを保証するため、以下のテストシステムを提供します：

- **単体テスト**: 21個のテストケース、全て通過（100%）
- **結合テスト**: 3429個のアサーション、全て通過（100%）
- **シンプル結合テスト**: 基本機能検証
- **実行時間監視**: マイクロ秒精度のパフォーマンス測定
- **エラーハンドリング**: 例外条件の包括的テスト
- **CI/CD対応**: 100%の成功率でプロダクション環境に対応

## 🚀 テスト実行方法

### 基本的な実行

```bash
# 全テスト実行（単体 + 結合テスト）
make test

# 単体テストのみ（21/21 通過）
make unit-test

# 包括的結合テスト（3429/3429 アサーション）
make integration-test

# シンプル結合テスト（基本機能）
make simple-integration-test
```

### 結合テストカバレッジ

結合テストは以下のコンポーネントを包括的にテストします：

- **🎯 XORモデルテスト**: 基本機能 + 学習収束（CI安全）
- **🔧 最適化器統合**: SGD + Adamフォールバックテスト
- **📊 損失関数統合**: MSE + CrossEntropy検証
- **💻 バックエンド統合**: CPUバックエンド包括テスト（601アサーション）
- **🔗 レイヤー統合**: Denseレイヤー + 活性化関数
- **🛠️ ユーティリティ統合**: Matrix、Random、Validation（504アサーション）
- **📱 デバイス統合**: CPUデバイス操作（2039アサーション）
- **📁 データ統合**: 読み込み、バッチ処理、検証（157アサーション）
- **⚡ パフォーマンステスト**: 安定性 + 実行時間監視

### 出力例

```bash
$ make integration-test
=== MLLib Integration Test Suite ===
Testing end-to-end functionality and workflows
Output capture enabled for clean test reporting

=== Running Test Suite: XOR Model Tests ===
Running test: BasicXORModelTest
✅ BasicXORModelTest PASSED (5 assertions, 0.17ms)

=== Running Test Suite: Optimizer Integration Tests ===
Running test: SGDOptimizerIntegrationTest
✅ SGDOptimizerIntegrationTest PASSED (11 assertions, 2.42ms)
Running test: AdamOptimizerIntegrationTest
✅ AdamOptimizerIntegrationTest PASSED (10 assertions, 1.04ms)

=== Running Test Suite: Backend Integration Tests ===
Running test: BackendPerformanceIntegrationTest
✅ BackendPerformanceIntegrationTest PASSED (551 assertions, 43.54ms)

============================================================
INTEGRATION TEST SUMMARY
============================================================
🎉 ALL INTEGRATION TESTS PASSED! 🎉
MLLib components work together correctly.
📊 Total: 3429/3429 assertions (100% success rate)
============================================================
✅ Integration tests passed
```

### シンプル結合テスト出力例

```bash
$ make simple-integration-test
Running simple integration tests...
Test 1: Basic model creation...
✓ Model created successfully
Test 2: Prediction with vector...
✓ Vector prediction completed successfully
Test 3: Prediction with initializer list...
✓ Initializer list prediction completed successfully

🎉 Basic integration tests passed (including {} syntax)!
✅ Simple integration tests completed successfully

--- NDArray Module Tests ---
Running test: NDArrayConstructorTest
✅ NDArrayConstructorTest PASSED (14 assertions, 0.01ms)
Running test: NDArrayMatmulTest
✅ NDArrayMatmulTest PASSED (11 assertions, 0.02ms)

============================================================
FINAL TEST SUMMARY
============================================================
Total individual tests: 21
Passed tests: 21
Failed tests: 0
Total test execution time: 0.45ms
Total suite time (including overhead): 0.89ms

🎉 ALL UNIT TESTS PASSED! 🎉
MLLib is ready for production use.
```

## 📊 テストカバレッジ

### 単体テスト (21/21)

#### Config Module (3テスト)
- **ConfigConstantsTest**: ライブラリ定数とバージョン情報
- **ConfigUsageTest**: 数学関数の使用例
- **ConfigMathTest**: 数学計算の正確性

#### NDArray Module (6テスト)
- **NDArrayConstructorTest**: 配列の構築とメモリ管理
- **NDArrayAccessTest**: インデックスアクセスと境界チェック
- **NDArrayOperationsTest**: fill、reshape、copyなどの操作
- **NDArrayArithmeticTest**: 加算、減算、乗算
- **NDArrayMatmulTest**: 行列乗算の機能性
- **NDArrayErrorTest**: エラーハンドリング

#### Dense Layer (4テスト)
- **DenseConstructorTest**: レイヤー構築とパラメータ初期化
- **DenseForwardTest**: 順伝播計算（2Dバッチ処理対応）
- **DenseBackwardTest**: 逆伝播とグラディエント計算
- **DenseParameterTest**: 重みとバイアスへのアクセス

#### Activation Functions (7テスト)
- **ReLUTest**: ReLU活性化関数の順伝播
- **ReLUBackwardTest**: ReLU活性化関数の逆伝播
- **SigmoidTest**: Sigmoid活性化関数の順伝播
- **SigmoidBackwardTest**: Sigmoid活性化関数の逆伝播
- **TanhTest**: Tanh活性化関数の順伝播
- **TanhBackwardTest**: Tanh活性化関数の逆伝播
- **ActivationErrorTest**: 活性化関数のエラー条件テスト

#### Sequential Model (1テスト)
- **SequentialModelTests**: モデル構築とpredict機能

### 統合テスト

#### XOR Integration Test
- エンドツーエンドの訓練プロセス
- XOR問題の学習と予測精度検証
- 訓練安定性の確認

#### Model I/O Integration Test
- 複数形式でのモデル保存・読み込み
- バイナリ、JSON、設定ファイル形式
- 予測精度の保持確認

#### Multi-Layer Integration Test
- 複雑なアーキテクチャの検証
- 深いネットワークの順伝播
- バッチ予測機能

#### Performance Integration Test
- 大規模データでの訓練安定性
- 数値的安定性の確認
- バッチ予測のパフォーマンス

## ⏱️ パフォーマンス監視

### 実行時間測定

MLLibのテストシステムは、各テストケースの実行時間をマイクロ秒精度で測定します：

```cpp
// テストケース内での自動計測
class TestCase {
private:
    double execution_time_ms_;  // ミリ秒単位の実行時間
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    double getExecutionTimeMs() const { return execution_time_ms_; }
};
```

### パフォーマンス指標

現在の基準値（参考）：
- **設定テスト**: 0.01-0.03ms
- **NDArray操作**: 0.01-0.02ms
- **Dense層テスト**: 0.01-0.02ms
- **活性化関数**: 0.00-0.15ms（エラーテストは例外処理のため時間がかかる）
- **全体**: 21テストで約0.45ms

## 🔧 テストフレームワーク

### TestCase基底クラス

```cpp
class TestCase {
public:
    explicit TestCase(const std::string& name);
    bool run();  // 実行時間測定付き
    
    // アサーションメソッド
    void assertTrue(bool condition, const std::string& message = "");
    void assertEqual(const T& expected, const T& actual, const std::string& message);
    void assertNear(double expected, double actual, double tolerance, const std::string& message);
    void assertThrows<ExceptionType>(Func func, const std::string& message);
    
    // 実行時間取得
    double getExecutionTimeMs() const;
};
```

### OutputCapture機能

テスト中の標準出力をキャプチャして、クリーンなテスト結果を表示：

```cpp
class OutputCapture {
    // 一時的にstdout/stderrをリダイレクト
    // テスト出力がコンソールに表示されないようにする
};
```

## 🛠️ カスタムテスト作成

### 新しいテストケースの追加

```cpp
class MyCustomTest : public MLLib::test::TestCase {
public:
    MyCustomTest() : TestCase("MyCustomTest") {}
    
protected:
    void test() override {
        // テスト実装
        assertEqual(2, 1+1, "Basic arithmetic should work");
        assertTrue(true, "True should be true");
        
        // 例外テスト
        assertThrows<std::runtime_error>(
            []() { throw std::runtime_error("test"); },
            "Should throw runtime_error"
        );
    }
};
```

### テストの登録と実行

```cpp
int main() {
    auto runTest = [&](std::unique_ptr<TestCase> test) {
        bool result = test->run();
        std::cout << "Execution time: " << test->getExecutionTimeMs() << "ms" << std::endl;
        return result;
    };
    
    runTest(std::make_unique<MyCustomTest>());
    return 0;
}
```

## 🔍 デバッグとトラブルシューティング

### テスト失敗時の対応

1. **アサーション失敗**: エラーメッセージを確認
2. **例外発生**: スタックトレースを確認
3. **実行時間異常**: パフォーマンス回帰の可能性

### ログ出力

```cpp
// テスト中のログ出力
std::cout << "Debug info: " << value << std::endl;  // OutputCaptureによりキャプチャされる
```

## 📈 継続的改善

### 新機能追加時のテスト

1. 単体テストを追加
2. 関連する統合テストを更新
3. パフォーマンス基準値を確認
4. 全テストの実行を確認

### パフォーマンス回帰検出

実行時間の大幅な増加が検出された場合：

1. 最適化の確認
2. アルゴリズムの見直し
3. メモリ使用量の確認
4. コンパイラ最適化オプションの確認

---

MLLibのテストシステムは、高品質なコードと安定したパフォーマンスを保証するために設計されています。新機能の追加や変更時は、必ずテストを実行して品質を維持してください。
