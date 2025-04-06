import requests
import pandas as pd

class DataLoader:
    def __init__(self, symbol="BTCUSDT", interval="1m", limit=100):
        """
        Initialize the data loader.
        :param symbol: Trading pair symbol (default: BTCUSDT)
        :param interval: Kline interval (default: 1m)
        :param limit: Number of data points to fetch (default: 100)
        """
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit

    def fetch_data(self):
        """
        Fetch historical candlestick data from Binance API.
        :return: Pandas DataFrame with Open, High, Low, Close, Volume, and Timestamps
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise error for failed requests
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_base_volume", "taker_quote_volume", "ignore"
            ])
            
            # Convert timestamp to readable format
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
            
            # Convert numerical columns to float
            num_cols = ["open", "high", "low", "close", "volume"]
            df[num_cols] = df[num_cols].astype(float)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None


def fetch_historical_data(symbol="BTCUSDT", interval="1m", limit=100):
    """
    Function to fetch historical market data.
    :param symbol: Trading pair symbol (default: BTCUSDT)
    :param interval: Kline interval (default: 1m)
    :param limit: Number of data points to fetch (default: 100)
    :return: Pandas DataFrame with Open, High, Low, Close, Volume, and Timestamps
    """
    loader = DataLoader(symbol=symbol, interval=interval, limit=limit)
    return loader.fetch_data()


if __name__ == "__main__":
    # Example usage
    data = fetch_historical_data(symbol="BTCUSDT", interval="1m", limit=10)
    
    if data is not None:
        print(data.head())  # Print first few rows
