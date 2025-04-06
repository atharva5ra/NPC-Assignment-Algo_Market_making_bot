import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils.data_loader import fetch_historical_data
import joblib

class SpreadOptimizer:
    def __init__(self, symbol="BTCUSDT", interval="1m", limit=1000):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

    def fetch_market_data(self):
        """Fetch historical market data."""
        df = fetch_historical_data(self.symbol, self.interval, self.limit)
        if df is None or df.empty:
            print("Error: No market data fetched. Check API connection.")
            return None
        return df

    def extract_features(self, df):
        """Extract features for spread optimization."""
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(window=10).std()
        df["price_depth"] = (df["high"] - df["low"]) / df["close"]
        df.dropna(inplace=True)

        X = df[["volatility", "price_depth"]]
        y = df["volatility"]
        return X, y

    def train_model(self):
        """Train a spread optimization model using real market data."""
        print("📊 Fetching market data...")
        df = self.fetch_market_data()
        if df is None:
            return

        print("🧮 Extracting features...")
        X, y = self.extract_features(df)

        print("🧠 Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Confirm that feature_names_in_ exists
        print("✅ Features used:", model.feature_names_in_)

        print(f"🎯 Model training complete. Test Score: {model.score(X_test, y_test):.4f}")

        # Save the model
        model_path = "C:/Users/91942/Desktop/Market_making_bot/models/spread_model.pkl"
        joblib.dump(model, model_path)
        print(f"💾 Spread Optimization Model saved at {model_path}")

if __name__ == "__main__":
    optimizer = SpreadOptimizer(symbol="BTCUSDT", interval="1m", limit=1000)
    optimizer.train_model()
