#include "MLLib.hpp"
#include <iostream>
#include <vector>

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

  // モデル保存用のラムダ関数（モデルをキャプチャ）
  auto save_callback = [&model](int epoch, double loss) {
    if (epoch % 10 == 0) {
      std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;

      // 10エポックごとにモデルを保存
      std::string model_dir = "samples/training_xor";
      std::string model_path = model_dir + "/epoch_" + std::to_string(epoch);

      try {
        // バイナリ形式で保存
        MLLib::model::ModelIO::save_model(model, model_path + ".bin",
                                          MLLib::model::ModelFormat::BINARY);
        // JSON形式でも保存（デバッグ用）
        MLLib::model::ModelIO::save_model(model, model_path + ".json",
                                          MLLib::model::ModelFormat::JSON);
        std::cout << "Model saved at epoch " << epoch << " to " << model_path
                  << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "Failed to save model at epoch " << epoch << ": "
                  << e.what() << std::endl;
      }
    }
  };

  // 学習（epochはライブラリ側デフォルト。コールバック関数で途中経過表示とモデル保存）
  model.train(X, Y, loss, optimizer, save_callback, 150);

  // 予測表示
  for (auto& x : X) {
    auto y_pred = model.predict(x);
    std::cout << x[0] << "," << x[1] << " => " << y_pred[0] << std::endl;
  }

  // 最終モデルを保存
  std::string final_model_dir = "samples/training_xor";
  std::string final_model_path = final_model_dir + "/final_model";

  try {
    MLLib::model::ModelIO::save_model(model, final_model_path + ".bin",
                                      MLLib::model::ModelFormat::BINARY);
    MLLib::model::ModelIO::save_model(model, final_model_path + ".json",
                                      MLLib::model::ModelFormat::JSON);
    MLLib::model::ModelIO::save_model(model, final_model_path + ".config",
                                      MLLib::model::ModelFormat::CONFIG);
    std::cout << "Final model saved to " << final_model_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to save final model: " << e.what() << std::endl;
  }
}