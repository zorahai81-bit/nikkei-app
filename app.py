import os
import json
import threading
import pandas as pd
from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__)

# --- グローバル状態管理 ---
status = {
    "download": {"state": "idle", "message": ""},
    "dataset": {"state": "idle", "message": ""},
    "prediction": {"state": "idle", "message": "", "result": None},
}

log_lines = []


def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    log_lines.append(f"[{ts}] {msg}")
    if len(log_lines) > 200:
        log_lines.pop(0)


# -------------------
# ルーティング
# -------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(status)


@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": log_lines[-50:]})


# -------------------
# Step 1: データDL
# -------------------

@app.route("/api/download", methods=["POST"])
def api_download():
    if status["download"]["state"] == "running":
        return jsonify({"error": "Already running"}), 400

    def run():
        try:
            status["download"]["state"] = "running"
            status["download"]["message"] = "ダウンロード中..."
            add_log("=== データダウンロード開始 ===")

            import yfinance as yf

            os.makedirs("data", exist_ok=True)

            symbols = {
                "nikkei": "^N225",
                "usdjpy": "JPY=X",
                "eurjpy": "EURJPY=X",
                "sp500": "^GSPC",
                "nasdaq": "^IXIC",
                "vix": "^VIX",
                "us10y": "^TNX",
                "SOX": "^SOX",
            }

            start_date = "2000-01-01"

            for name, ticker in symbols.items():
                add_log(f"Downloading {name} ({ticker})...")
                df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.reset_index(inplace=True)
                df.to_csv(f"data/{name}.csv", index=False)
                add_log(f"  -> {len(df)} rows saved")

            status["download"]["state"] = "done"
            status["download"]["message"] = "ダウンロード完了 ✓"
            add_log("=== ダウンロード完了 ===")

        except Exception as e:
            status["download"]["state"] = "error"
            status["download"]["message"] = f"エラー: {str(e)}"
            add_log(f"ERROR: {str(e)}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


# -------------------
# Step 2: 特徴量生成
# -------------------

@app.route("/api/create_dataset", methods=["POST"])
def api_create_dataset():
    if status["dataset"]["state"] == "running":
        return jsonify({"error": "Already running"}), 400

    def run():
        try:
            status["dataset"]["state"] = "running"
            status["dataset"]["message"] = "特徴量生成中..."
            add_log("=== データセット作成開始 ===")

            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.trend import MACD
            from ta.volatility import BollingerBands, AverageTrueRange

            add_log("CSVファイル読み込み中...")
            nikkei = pd.read_csv("data/nikkei.csv")
            usdjpy = pd.read_csv("data/usdjpy.csv")
            eurjpy = pd.read_csv("data/eurjpy.csv")
            sp500 = pd.read_csv("data/sp500.csv")
            nasdaq = pd.read_csv("data/nasdaq.csv")
            vix = pd.read_csv("data/vix.csv")
            us10y = pd.read_csv("data/us10y.csv")
            sox = pd.read_csv("data/SOX.csv")

            dfs = [nikkei, usdjpy, eurjpy, sp500, nasdaq, vix, us10y, sox]
            for d in dfs:
                d["Date"] = pd.to_datetime(d["Date"])

            usdjpy = usdjpy[["Date", "Close"]].rename(columns={"Close": "USDJPY"})
            eurjpy = eurjpy[["Date", "Close"]].rename(columns={"Close": "EURJPY"})
            sp500 = sp500[["Date", "Close"]].rename(columns={"Close": "SP500"})
            nasdaq = nasdaq[["Date", "Close"]].rename(columns={"Close": "NASDAQ"})
            vix = vix[["Date", "Close"]].rename(columns={"Close": "VIX"})
            us10y = us10y[["Date", "Close"]].rename(columns={"Close": "US10Y"})
            sox = sox[["Date", "Close"]].rename(columns={"Close": "SOX"})

            add_log("データ結合中...")
            df = nikkei.copy()
            for add_df in [usdjpy, eurjpy, sp500, nasdaq, vix, us10y, sox]:
                df = df.merge(add_df, on="Date", how="left")
            df = df.sort_values("Date").ffill()

            add_log("特徴量エンジニアリング中...")
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

            for p in [1, 3, 5, 10, 20]:
                df[f"ret_{p}"] = df["Close"].pct_change(p)
            for p in [5, 10, 25, 50, 75]:
                df[f"ma_{p}"] = df["Close"].rolling(p).mean()
                df[f"ma_bias_{p}"] = df["Close"] / df[f"ma_{p}"] - 1
            for p in [5, 10, 20]:
                df[f"volatility_{p}"] = df["ret_1"].rolling(p).std()
            for p in [1, 5]:
                df[f"usdjpy_ret_{p}"] = df["USDJPY"].pct_change(p)
                df[f"eurjpy_ret_{p}"] = df["EURJPY"].pct_change(p)
                df[f"sp500_ret_{p}"] = df["SP500"].pct_change(p)
                df[f"nasdaq_ret_{p}"] = df["NASDAQ"].pct_change(p)
                df[f"vix_ret_{p}"] = df["VIX"].pct_change(p)
                df[f"sox_ret_{p}"] = df["SOX"].pct_change(p)
            df["us10y_ret"] = df["US10Y"].pct_change()
            df["dayofweek"] = df["Date"].dt.dayofweek
            df["month"] = df["Date"].dt.month

            add_log("テクニカル指標計算中...")
            df["rsi_7"] = RSIIndicator(close=df["Close"], window=7).rsi()
            df["rsi_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
            df["rsi_28"] = RSIIndicator(close=df["Close"], window=28).rsi()

            macd = MACD(close=df["Close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()

            bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = bb.bollinger_wband()

            atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
            df["atr"] = atr.average_true_range()

            stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()

            df = df.dropna()
            df.to_csv("features.csv", index=False)

            add_log(f"features.csv 保存完了: {df.shape[0]} rows × {df.shape[1]} cols")
            status["dataset"]["state"] = "done"
            status["dataset"]["message"] = f"特徴量生成完了 ✓ ({df.shape[0]}行)"
            add_log("=== データセット作成完了 ===")

        except Exception as e:
            status["dataset"]["state"] = "error"
            status["dataset"]["message"] = f"エラー: {str(e)}"
            add_log(f"ERROR: {str(e)}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


# -------------------
# Step 3: 予測
# -------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if status["prediction"]["state"] == "running":
        return jsonify({"error": "Already running"}), 400

    def run():
        try:
            status["prediction"]["state"] = "running"
            status["prediction"]["message"] = "モデル学習・予測中..."
            status["prediction"]["result"] = None
            add_log("=== 予測開始 ===")

            from lightgbm import LGBMClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            add_log("features.csv 読み込み中...")
            df = pd.read_csv("features.csv")
            df["Date"] = pd.to_datetime(df["Date"])

            train = df[df["Date"] < "2021-01-01"]
            valid = df[(df["Date"] >= "2021-01-01") & (df["Date"] < "2024-01-01")]
            test = df[df["Date"] >= "2024-01-01"]

            drop_cols = ["Date", "Target"]
            X_train = train.drop(columns=drop_cols)
            y_train = train["Target"]
            X_test = test.drop(columns=drop_cols)
            y_test = test["Target"]
            X_all = df.drop(columns=drop_cols)
            y_all = df["Target"]

            params = dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)

            add_log(f"学習データ: {len(train)}行 / 検証: {len(valid)}行 / テスト: {len(test)}行")

            add_log("評価モデル学習中 (train only)...")
            model_eval = LGBMClassifier(**params)
            model_eval.fit(X_train, y_train)
            pred = model_eval.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred)
            rec = recall_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            add_log(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

            add_log("本番モデル学習中 (全データ)...")
            model = LGBMClassifier(**params)
            model.fit(X_all, y_all)

            importance = pd.DataFrame({
                "feature": X_all.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            top20 = importance.head(20)[["feature", "importance"]].to_dict(orient="records")

            latest_row = df.iloc[[-1]]
            latest_date = latest_row["Date"].values[0]
            X_latest = latest_row.drop(columns=drop_cols)
            tomorrow_pred = int(model.predict(X_latest)[0])
            tomorrow_prob = model.predict_proba(X_latest)[0].tolist()

            result = {
                "latest_date": str(pd.Timestamp(latest_date).date()),
                "prediction": tomorrow_pred,
                "up_prob": round(tomorrow_prob[1], 4),
                "down_prob": round(tomorrow_prob[0], 4),
                "metrics": {
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                },
                "top20": top20,
            }

            status["prediction"]["state"] = "done"
            status["prediction"]["message"] = "予測完了 ✓"
            status["prediction"]["result"] = result
            add_log("=== 予測完了 ===")

        except Exception as e:
            status["prediction"]["state"] = "error"
            status["prediction"]["message"] = f"エラー: {str(e)}"
            add_log(f"ERROR: {str(e)}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
