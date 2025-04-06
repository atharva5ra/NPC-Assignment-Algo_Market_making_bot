import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils.data_loader import fetch_historical_data
import joblib

class InventoryManager:
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
        """Extract features for inventory management optimization."""
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(window=10).std()
        df["trade_volume"] = df["volume"]
        df["inventory_level"] = np.random.uniform(0.1, 1.0, len(df))  # Placeholder for real inventory data

        df.dropna(inplace=True)

        features = df[["volatility", "trade_volume", "inventory_level"]]
        target = df["inventory_level"]
        return features, target

    def train_model(self):
        """Train an inventory management optimization model using real market data."""
        print("Fetching market data...")
        df = self.fetch_market_data()
        if df is None:
            return

        print("Extracting features...")
        X, y = self.extract_features(df)

        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Manually set feature_names_in_ for compatibility with Algo_market_maker
        model.feature_names_in_ = X.columns.to_numpy()

        print(f"Model training complete. Score: {model.score(X_test, y_test):.4f}")

        # Save model
        model_path = "C:/Users/91942/Desktop/Market_making_bot/models/inventory_model.pkl"
        joblib.dump(model, model_path)
        print(f"Inventory Management Model saved at {model_path}")

if __name__ == "__main__":
    manager = InventoryManager(symbol="BTCUSDT", interval="1m", limit=1000)
    manager.train_model()
