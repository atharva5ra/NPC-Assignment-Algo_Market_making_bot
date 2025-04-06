import logging
from decimal import Decimal
from typing import Dict, List, Any
import numpy as np
import joblib
import pandas_ta as ta
import pandas as pd

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

from scripts.model_loader import load_models

class Algo_market_maker(ScriptStrategyBase):
    # === CONFIGURATION ===
    bid_spread = 0.0006
    ask_spread = 0.0004
    order_refresh_time = 10
    order_amount = 0.05
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 100

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.log_with_clock(logging.INFO, "Initializing Algo_market_maker strategy")

        self.candles = CandlesFactory.get_candle(CandlesConfig(
            connector=self.candle_exchange,
            trading_pair=self.trading_pair,
            interval=self.candles_interval,
            max_records=self.max_records
        ))
        self.candles.start()

        self.create_timestamp = 0

        # === Load models using utility loader ===
        try:
            models = load_models("scripts/models")
            self.spread_model = models["spread_model"]
            self.inventory_model = models["inventory_model"]
            self.trend_model = models["trend_model"]
##            self.orderbook_model = models["orderbook_model"]
##            self.sentiment_model = models["sentiment_model"]
##            self.regime_model = models["regime_model"]
            self.volatility_model = models["volatility_model"]
            print("Spread model expects:", self.spread_model.feature_names_in_)
            print("Inventory model expects:", self.inventory_model.feature_names_in_)
            print("Trend model expects:", self.trend_model.feature_names_in_)
##            print("Orderbook model expects:", self.orderbook_model.feature_names_in_)
##            print("Sentiment model expects:", self.sentiment_model.feature_names_in_)
##            print("Regime model expects:", self.regime_model.feature_names_in_)
            print("Volatility model expects:", self.volatility_model.feature_names_in_)
            self.log_with_clock(logging.INFO, "All models loaded successfully via utils/model_loader.py.")
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Model loading failed: {e}")
            raise

    def on_stop(self):
        self.log_with_clock(logging.INFO, "Stopping strategy and candle feed.")
        self.candles.stop()

    def on_tick(self):
      if not self.ready_to_trade:
        self.log_with_clock(logging.WARNING, "Market not ready.")
        return

      self.log_with_clock(logging.DEBUG, f"Tick received at timestamp {self.current_timestamp}")
      self.log_with_clock(logging.INFO, f"Candles available: {len(self.candles.candles_df)} rows")

      if self.create_timestamp <= self.current_timestamp:
        try:
            self.cancel_all_orders()
            self.log_with_clock(logging.INFO, "Creating new proposal.")
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)

            try:
                if len(proposal_adjusted) < 2:
                  self.notify_hb_app_with_timestamp("⚠️ Not enough proposals to place both orders.")
                  return
                ref_price = float(self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source))
                buy_price = proposal_adjusted[0].price if proposal_adjusted[0].order_side == TradeType.BUY else None
                sell_price = proposal_adjusted[1].price if proposal_adjusted[1].order_side == TradeType.SELL else None
                self.log_with_clock(logging.INFO, f"Mid Price: {ref_price:.2f} | Buy Price: {buy_price} | Sell Price: {sell_price}")
                self.notify_hb_app_with_timestamp(f"✅ Orders Placed | BUY: {buy_price:.2f}, SELL: {sell_price:.2f}")
            except Exception as e:
                self.notify_hb_app_with_timestamp(f"⚠️ Order notification error: {e}")

            self.create_timestamp = int(self.current_timestamp + self.order_refresh_time)

        except Exception as e:
            import traceback
            self.log_with_clock(logging.ERROR, f"❌ Exception during on_tick: {e}")
            self.log_with_clock(logging.ERROR, traceback.format_exc())

    def compute_features(self) -> Dict[str, Any]:
      df = self.candles.candles_df.copy()
      features = {}

      try:
        returns = df["close"].pct_change().dropna()
        features["volatility"] = returns.rolling(window=5).std().iloc[-1]
        features["trend"] = df["close"].rolling(window=5).mean().iloc[-1]
        features["last_price"] = df["close"].iloc[-1]
        features["rsi"] = df.ta.rsi(length=14).iloc[-1]
        df.ta = ta

        base, quote = self.trading_pair.split("-")
        base_bal = self.connectors[self.exchange].get_balance(base)
        quote_bal = self.connectors[self.exchange].get_balance(quote)
        base_bal_f = float(base_bal)
        quote_bal_f = float(quote_bal)
        self.log_with_clock(logging.INFO, f"💰 Balances | {base}: {base_bal}, {quote}: {quote_bal}")
        last_price = float(features["last_price"])
        features["inventory_ratio"] = base_bal_f / (base_bal_f + (quote_bal_f / last_price)) if last_price != 0 else 0.5

        # Fill remaining ML features with dummy or derived values
        features["price_depth"] = 1.0
        features["trade_volume"] = 1000.0
        features["inventory_level"] = features["inventory_ratio"]
        features["price_slope"] = 0.01
        features["momentum"] = 0.01
        features["bid_ask_imbalance"] = 0.0
        features["depth_ratio"] = 1.0
##        features["sentiment_score"] = 0.0
        features["price_std"] = features["volatility"]
        features["volume_std"] = 100.0

      except Exception as e:
        self.log_with_clock(logging.WARNING, f"Feature computation error: {e}")
        # Fallback defaults for all features
        features = {
            "volatility": 0.01,
            "trend": 1000,
            "last_price": 1000,
            "rsi": 50,
            "inventory_ratio": 0.5,
            "price_depth": 1.0,
            "trade_volume": 1000.0,
            "inventory_level": 0.5,
            "price_slope": 0.01,
            "momentum": 0.01,
            "bid_ask_imbalance": 0.0,
            "depth_ratio": 1.0,
            "price_std": 0.01,
            "volume_std": 100.0,
        }

      self.log_with_clock(logging.DEBUG, f"Computed features: {features}")
      return features
    
    def predict_model(self, model, input_df):
      try:
        return float(model.predict(input_df)[0])
      except AttributeError as e:
        self.log_with_clock(logging.WARNING, f"Model prediction failed: {e} — Retrying with NumPy values")
        try:
            return float(model.predict(input_df.values)[0])
        except Exception as ex:
            self.log_with_clock(logging.ERROR, f"Model prediction retry failed: {ex}")
            return 0.0  # Safe fallback
      except Exception as e:
        self.log_with_clock(logging.ERROR, f"Unexpected model prediction error: {e}")
        return 0.0
      
    def create_proposal(self) -> List[OrderCandidate]:
      try:
        features = self.compute_features()
        # Define expected input features manually
        spread_features = ['volatility', 'price_depth']
        inv_features = ['volatility', 'trade_volume', 'inventory_level']
        trend_features = ['price_slope', 'momentum']
##        ob_features = ['bid_ask_imbalance', 'depth_ratio']
##        sentiment_features = ['sentiment_score']
##        regime_features = ['volatility', 'momentum']
        vol_features = ['price_std', 'volume_std']
        
        spread_inputs = pd.DataFrame([[features[col] for col in spread_features]], columns=spread_features)
        inv_inputs = pd.DataFrame([[features[col] for col in inv_features]], columns=inv_features)
        trend_inputs = pd.DataFrame([[features[col] for col in trend_features]], columns=trend_features)
##        ob_inputs = pd.DataFrame([[features[col] for col in ob_features]], columns=ob_features)
##        sentiment_inputs = pd.DataFrame([[features[col] for col in sentiment_features]], columns=sentiment_features)
##        regime_inputs = pd.DataFrame([[features[col] for col in regime_features]], columns=regime_features)
        vol_inputs = pd.DataFrame([[features[col] for col in vol_features]], columns=vol_features)

        # Predict adjustments using the safer helper function
        spread_adj = self.predict_model(self.spread_model, spread_inputs)
        inv_adj = self.predict_model(self.inventory_model, inv_inputs)
        trend_adj = self.predict_model(self.trend_model, trend_inputs)
##        ob_adj = self.predict_model(self.orderbook_model, ob_inputs)
##        sentiment_adj = self.predict_model(self.sentiment_model, sentiment_inputs)
##        regime_adj = self.predict_model(self.regime_model, regime_inputs)
        vol_adj = self.predict_model(self.volatility_model, vol_inputs)

        self.log_with_clock(logging.DEBUG, f"Model outputs - Spread: {spread_adj}, Inv: {inv_adj}, Trend: {trend_adj}, Vol: {vol_adj}")

        # === Raw spread adjustments (asymmetric logic is okay) ===
        raw_bid_adj = spread_adj - inv_adj - trend_adj + vol_adj
        raw_ask_adj = spread_adj + inv_adj + trend_adj + vol_adj

        # ✅ Cap only the *upper limit* to 5%
        capped_bid_adj = min(raw_bid_adj, 0.05)
        capped_ask_adj = min(raw_ask_adj, 0.05)

        # === Final adjusted spreads ===
        final_bid_spread = self.bid_spread * (1 + capped_bid_adj)
        final_ask_spread = self.ask_spread * (1 + capped_ask_adj)

        # === Reference mid price and final prices ===
        ref_price = float(self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source))
        self.log_with_clock(logging.DEBUG, f"📈 Reference price: {ref_price}")

        buy_price = Decimal(ref_price * (1 - final_bid_spread))
        sell_price = Decimal(ref_price * (1 + final_ask_spread))

        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(self.order_amount),
            price=buy_price
        )

        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(self.order_amount),
            price=sell_price
        )

        return [buy_order, sell_order]

      except Exception as e:
        import traceback
        self.log_with_clock(logging.ERROR, f"❌ Exception in create_proposal: {e}")
        self.log_with_clock(logging.ERROR, traceback.format_exc())
        return []

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        try:
            return self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
        except Exception as e:
            self.log_with_clock(logging.WARNING, f"Budget adjustment error: {e}")
            return []

    def place_orders(self, proposal: List[OrderCandidate]):
      for order in proposal:
        try:
            if order.order_side == TradeType.BUY:
                self.buy(self.exchange, order.trading_pair, order.amount, order.order_type, order.price)
            elif order.order_side == TradeType.SELL:
                self.sell(self.exchange, order.trading_pair, order.amount, order.order_type, order.price)
            self.log_with_clock(logging.INFO, f"Placed {order.order_side.name} order at {order.price} for {order.amount}")
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Failed to place {order.order_side.name} order at {order.price} for {order.amount}: {str(e)}")

    def cancel_all_orders(self):
        self.log_with_clock(logging.INFO, "Cancelling all active orders.")
        for order in self.get_active_orders(self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        lines = ["", f"  Strategy: Algo_market_maker @ {self.trading_pair}"]

        if not self.ready_to_trade:
            lines.append("  Status: Market connector not ready.")
            return "\n".join(lines)

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            order_df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in order_df.to_string(index=False).split("\n")])
        except Exception:
            lines.append("  No active orders.")

        try:
            candles_df = self.candles.candles_df.tail(self.candles_length).iloc[::-1]
            lines.extend(["", "-" * 70])
            lines.append(f"  Candles Feed: {self.candles.name} | Interval: {self.candles.interval}")
            lines.extend(["    " + line for line in candles_df.to_string(index=False).split("\n")])
        except Exception as e:
            lines.append(f"  Candle data not available: {e}")

        return "\n".join(lines)