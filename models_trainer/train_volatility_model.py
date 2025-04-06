import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import pickle
from utils.data_loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

def compute_realized_volatility(returns):
    """Compute Realized Volatility (RV) using squared returns."""
    return np.sqrt(np.sum(returns ** 2))

def compute_ewma_volatility(returns, lambda_factor=0.94):
    """Compute Exponentially Weighted Moving Average (EWMA) Volatility."""
    return returns.ewm(span=(2 / (1 - lambda_factor))).std()

def train_volatility_model(symbol="BTCUSDT", interval="1m", limit=500):
    """Train a simple model to estimate market volatility."""
    # Load historical price data
    loader = DataLoader(symbol=symbol, interval=interval, limit=limit)
    df = loader.fetch_data()

    if df is None:
        print("Failed to load data.")
        return

    # Compute log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Compute Realized Volatility (RV)
    df["realized_volatility"] = df["log_return"].rolling(window=10).apply(compute_realized_volatility, raw=True)

    # Compute EWMA Volatility
    df["ewma_volatility"] = compute_ewma_volatility(df["log_return"])

    # Drop NaN values
    df.dropna(inplace=True)

    # ✅ Train on DataFrame with column names — NOT .values
    X = df[["ewma_volatility"]]  # KEEP AS DATAFRAME
    y = df["realized_volatility"]

    model = LinearRegression()
    model.fit(X, y)

    # Save model
    model_path = "C:/Users/91942/Desktop/Market_making_bot/models/volatility_model.pkl"
    joblib.dump(model, model_path)

    print(f"Volatility Model saved to {model_path}")
    print(f"Volatility Model features: {model.feature_names_in_}")

    # Plot volatility estimates
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["realized_volatility"], label="Realized Volatility", color="blue")
    plt.plot(df["timestamp"], df["ewma_volatility"], label="EWMA Volatility", color="red", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.title("Volatility Estimation (RV & EWMA)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_volatility_model()
