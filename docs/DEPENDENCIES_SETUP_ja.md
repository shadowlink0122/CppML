# MLLib 依存関係セットアップガイド

## 🚀 クイックスタート（自動）

MLLibは依存関係を自動管理します。以下を実行するだけです：

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make setup-deps    # 必要なライブラリを自動ダウンロード
make               # 完全なJSON対応でビルド
```

## 📋 ビルドオプション

### オプション1: フル機能（推奨）
```bash
make               # 自動依存関係ダウンロード + フルビルド
make test          # JSON I/Oを含むすべてのテストを実行
```

### オプション2: 最小ビルド（依存関係なし）
```bash
make minimal       # JSON読み込みサポートなしでビルド
                   # JSON保存は動作します（依存関係不要）
```

### オプション3: 手動依存関係管理
```bash
make deps-check    # 不足している依存関係をチェック
make deps-install  # 不足している依存関係をインストール
make build         # 利用可能な機能でビルド
```

## 🌍 プラットフォーム別手順

### Linux (Ubuntu/Debian)
```bash
# 自動（推奨）
make

# 手動インストール
sudo apt-get update
sudo apt-get install curl wget
make setup-deps
```

### Linux (CentOS/RHEL/Fedora)
```bash
# 自動（推奨）
make

# 手動インストール
sudo yum install curl wget  # CentOS/RHEL
# または
sudo dnf install curl wget  # Fedora
make setup-deps
```

### macOS
```bash
# 自動（推奨）
make

# Homebrewで手動（オプション）
brew install curl wget
make setup-deps

# 組み込みcurlで手動
make setup-deps  # 組み込みcurlを使用
```

### Windows
```bash
# Git Bash / MSYS2 / WSL
make

# 手動ダウンロード（自動が失敗した場合）
curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o include/MLLib/third_party/json.hpp
make build
```

## 🔧 トラブルシューティング

### ネットワークの問題
```bash
# 自動ダウンロードが失敗した場合：
mkdir -p include/MLLib/third_party
wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -O include/MLLib/third_party/json.hpp

# 代替ミラー
curl -L https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o include/MLLib/third_party/json.hpp
```

### オフライン環境
```bash
# 1. インターネット接続されたマシンでダウンロード：
curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o json.hpp

# 2. オフラインマシンにコピー：
mkdir -p include/MLLib/third_party
cp json.hpp include/MLLib/third_party/

# 3. オフラインでビルド：
make build
```

### 企業ファイアウォール
```bash
# 企業プロキシを使用
export https_proxy=your-proxy:port
make setup-deps

# または手動でダウンロードして include/MLLib/third_party/ に配置
```

## 📊 ビルドタイプ別機能マトリクス

| ビルドタイプ | バイナリI/O | JSON保存 | JSON読み込み | 依存関係 |
|--------------|-------------|----------|--------------|----------|
| **フル** | ✅ | ✅ | ✅ | 自動管理 |
| **最小** | ✅ | ✅ | ❌ | なし |
| **カスタム** | ✅ | ✅ | ⚠️ | ユーザー選択 |

## 🧪 検証

セットアップをテスト：
```bash
# クイックテスト
make test-deps     # 依存関係を検証
make quick-test    # 基本機能テスト
make full-test     # 完全なテストスイート

# 機能別テスト
make test-json     # JSON I/Oテスト
make test-binary   # バイナリI/Oテスト
```

## ⚠️ 既知の問題

### 問題: curl/wgetが利用できない
**解決方法:**
```bash
# 最初にcurlまたはwgetをインストール
# Ubuntu/Debian: sudo apt-get install curl
# CentOS/RHEL: sudo yum install curl
# macOS: curlは組み込み
# Windows: Git Bashを使用するかcurlをインストール
```

### 問題: 許可が拒否される
**解決方法:**
```bash
# 書き込み権限があることを確認
sudo chown -R $USER:$USER include/
make setup-deps
```

### 問題: 企業ネットワークがGitHubをブロック
**解決方法:**
```bash
# 代替ソースから手動ダウンロード
curl -L https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp -o include/MLLib/third_party/json.hpp
```

## 🤝 サポート

問題が発生した場合：

1. **ビルドログを確認**: `make verbose`
2. **依存関係を検証**: `make deps-check`
3. **最小ビルドを試行**: `make minimal`
4. **問題を報告**: [GitHub Issues](https://github.com/shadowlink0122/CppML/issues)

## 📄 依存関係ライセンス

- **nlohmann/json**: MITライセンス
- MLLibのBSD 3-Clause with Commercial Use Restrictionと互換性あり
