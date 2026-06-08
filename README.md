# 日経AI予測 — Flask Web App

スマホ・PCどちらでも使える日経225翌日方向性予測ウェブアプリです。

## セットアップ

```bash
cd nikkei_app

# 依存パッケージのインストール
pip install -r requirements.txt

# サーバー起動
python app.py
```

## アクセス

ブラウザで開く:
- ローカル: http://localhost:5000
- スマホ(同じWiFi): http://<PCのIPアドレス>:5000

## 使い方（3ステップ）

1. **STEP 01 — データ取得**  
   「ダウンロード開始」をタップ → Yahoo Financeから8種の市場データを取得（初回のみ）

2. **STEP 02 — 特徴量生成**  
   「特徴量生成」をタップ → RSI、MACD、ボリンジャーバンドなど50以上の特徴量を計算

3. **STEP 03 — AI予測実行**  
   「予測実行」をタップ → LightGBMモデルが翌営業日の方向性を予測

## 画面の説明

- **翌日予測**: 上昇↑ / 下落↓ と確率（UP/DOWN Probability）
- **評価指標**: Accuracy・Precision・Recall・F1（テスト期間 2024年〜）
- **Feature Importance**: モデルが最も重視した特徴量TOP10
- **System Log**: リアルタイム進捗ログ

## 注意事項

- 初回のデータDLは数分かかる場合があります
- 予測は参考情報です。投資判断は自己責任でお願いします
