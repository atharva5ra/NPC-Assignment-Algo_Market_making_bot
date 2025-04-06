import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(".."))
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import joblib

def fetch_historical_data(symbol="BTCUSDT", interval="1m", limit=1000):
    """Fetch historical OHLCV market data from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_base_volume", "taker_quote_volume", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def compute_indicators(df):
    """Compute RSI, MACD, and Bollinger Bands."""
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = MACD(df["close"]).macd()
    bollinger = BollingerBands(df["close"])
    df["bollinger_h"] = bollinger.bollinger_hband()
    df["bollinger_l"] = bollinger.bollinger_lband()

    # Define trend labels: 1 = Uptrend, 0 = Sideways, -1 = Downtrend
    df["trend"] = np.where(df["close"].shift(-1) > df["close"], 1, -1)
    df["trend"] = np.where(np.abs(df["close"].pct_change()) < 0.002, 0, df["trend"])

    df.dropna(inplace=True)
    return df

def train_trend_model():
    """Train trend prediction model."""
    print("📊 Fetching market data...")
    df = fetch_historical_data()
    if df is None or df.empty:
        print("No market data fetched. Exiting...")
        return

    print("🧮 Extracting features...")
    df = compute_indicators(df)

    feature_cols = ["rsi", "macd", "bollinger_h", "bollinger_l"]
    X = df[feature_cols]
    y = df["trend"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Wrap scaled arrays back into DataFrames with correct feature names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

    print("🧠 Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_df, y_train)

    print("✅ Features used:", model.feature_names_in_)
    print(f"🎯 Model training complete. Accuracy: {model.score(X_test_df, y_test):.4f}")

    # Save the model
    model_path = "C:/Users/91942/Desktop/Market_making_bot/models/trend_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Trend Analysis Model saved at {model_path}")

if __name__ == "__main__":
    train_trend_model()
