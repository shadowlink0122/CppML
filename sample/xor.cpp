#include "MLLib.hpp"
#include <iostream>
#include <vector>

// 事前定義のコールバック関数
void on_epoch_end(int epoch, double loss) {
  if (epoch % 10 == 0) {
    std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;
  }
}

int main() {
  // XORデータ
  std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

  // モデル構築
  MLLib::model::Sequential model;
  // デバイス指定（将来GPUにも対応可能）
  model.set_device(MLLib::DeviceType::CPU);
  // レイヤー追加
  // 入力層: 2次元入力
  model.add_layer(new MLLib::layer::Dense(2, 4));
  // 活性化関数: ReLUを使用
  model.add_layer(new MLLib::layer::activation::ReLU());
  // 隠れ層: 4ユニット、出力層: 1ユニット
  model.add_layer(new MLLib::layer::Dense(4, 1));
  // 活性化関数: Sigmoidを使用
  model.add_layer(new MLLib::layer::activation::Sigmoid());

  // 損失関数と最適化
  MLLib::loss::MSELoss loss;
  MLLib::optimizer::SGD optimizer(0.1);

  // 学習（epochはライブラリ側デフォルト。コールバック関数で途中経過表示）
  model.train(X, Y, loss, optimizer, on_epoch_end, 150);

  // 予測表示
  for (auto& x : X) {
    auto y_pred = model.predict(x);
    std::cout << x[0] << "," << x[1] << " => " << y_pred[0] << std::endl;
  }
}