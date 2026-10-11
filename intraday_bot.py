"""Intraday Trading Bot for Indian Equity Market
Author: Cascade AI assistant
Date: 2025-08-03

This script supports two primary modes:
1. Historical back-testing over the last 6 months.
2. Near-real-time signal generation for live deployment.

Replace `PLACE_ORDER_FUNCTION()` with your broker-specific order placement logic
in `live_trading()`.
"""

from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Literal, Optional

import pandas as pd
import numpy as np
# Patch for pandas_ta compatibility with NumPy ≥2.0
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # noqa: N806 – match expected constant name
import pandas_ta as ta
import pytz
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

INDIA_TZ = pytz.timezone("Asia/Kolkata")

###############################################################################
# Helper utilities
###############################################################################

def india_now() -> datetime:
    """Current time in IST as timezone-aware datetime."""
    return datetime.now(INDIA_TZ)


def is_market_open(now: Optional[datetime] = None) -> bool:
    """True if the time is between 09:15 & 15:25 IST on a weekday."""
    now = now or india_now()
    market_start = dtime(9, 15)
    market_end = dtime(15, 25)  # square-off handled separately
    return now.weekday() < 5 and market_start <= now.time() <= market_end

###############################################################################
# Main trading bot class
###############################################################################

class IntradayTradingBot:
    """EMA × RSI × VWAP intraday strategy for NSE equities/indices."""

    def __init__(
        self,
        symbol: str = "RELIANCE.NS",
        interval: Literal["1m", "5m", "1d"] = "5m",
        risk_per_trade: float = 1.0,  # percentage of equity
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------------------
    # Data gathering & indicator calculation
    # ---------------------------------------------------------------------
    def fetch_data(self, period: str = "7d") -> pd.DataFrame:
        """Download OHLCV data via Yahoo Finance."""
        print(f"\n[DEBUG] Fetching {self.interval} data for {self.symbol} (period: {period})...")
        try:
            df = yf.download(
                tickers=self.symbol,
                period=period,
                interval=self.interval,
                auto_adjust=False,
                progress=False,  # Less noisy output
                group_by="column",  # Ensure single-level columns
                threads=False,
            )
            print(f"[DEBUG] Raw data shape: {df.shape}")
            try:
                nlevels = getattr(df.columns, "nlevels", 1)
                print(f"[DEBUG] Columns nlevels: {nlevels} | columns: {list(df.columns)}")
            except Exception:
                pass
            if df.empty:
                print("[ERROR] No data returned from yfinance!")
                return df
                
            # Flatten to single-level columns if MultiIndex is returned
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    tickers_level = df.columns.get_level_values(-1)
                    upper_map = {str(t).upper(): t for t in set(tickers_level)}
                    sym_upper = self.symbol.upper()
                    if sym_upper in upper_map:
                        # Select columns for this ticker (case-insensitive)
                        df = df.xs(upper_map[sym_upper], axis=1, level=-1)
                    else:
                        # Fallback: drop last level and keep OHLCV first-level names
                        df = df.droplevel(-1, axis=1)
                    print(f"[DEBUG] Flattened columns: {list(df.columns)}")
                except Exception as fe:
                    print(f"[WARN] Failed to flatten columns gracefully: {fe}; falling back to first level")
                    df.columns = df.columns.get_level_values(0)

            # Standardize column names and drop rows missing core OHLCV only
            df = df.rename(str.capitalize, axis="columns")
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[ERROR] Missing required columns after fetch: {missing}")
                return pd.DataFrame()
            df = df.dropna(subset=required)
            print(f"[DEBUG] Final columns: {list(df.columns)}")
            
            # `yf.download` returns index in UTC ‑ convert to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(INDIA_TZ)
            else:
                df.index = df.index.tz_convert(INDIA_TZ)
                
            print(f"[DEBUG] After cleaning: {df.shape}")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMAs, VWAP & deviation, RSI, ATR and helper columns."""

        df = df.copy()
        # EMAs
        df["EMA_20"] = ta.ema(df["Close"], length=20)
        df["EMA_50"] = ta.ema(df["Close"], length=50)
        df["EMA_100"] = ta.ema(df["Close"], length=100)

        # VWAP (session-independent cumulative)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["CumTPV"] = (typical_price * df["Volume"]).cumsum()
        df["CumVol"] = df["Volume"].cumsum()
        df["VWAP"] = df["CumTPV"] / df["CumVol"]
        df["VWAP_Slope"] = df["VWAP"].diff()
        df.drop(columns=["CumTPV", "CumVol"], inplace=True)

        # RSI & ATR
        df["RSI"] = ta.rsi(df["Close"], length=14)
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        # Clean initial NaNs
        df.dropna(inplace=True)
        return df

    # ---------------------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------------------
    @staticmethod
    def _make_signal_row(row: pd.Series) -> int:
        """Return 1 for long, ‑1 for short, 0 otherwise, based on row indicators."""
        long_cond = (
            row.EMA_20 > row.EMA_50
            and row.EMA_50 > row.EMA_100
            and row.Close > row.VWAP
            and row.VWAP_Slope > 0
            and row.RSI > 50
        )
        short_cond = (
            row.EMA_20 < row.EMA_50
            and row.EMA_50 < row.EMA_100
            and row.Close < row.VWAP
            and row.VWAP_Slope < 0
            and row.RSI < 50
        )
        if long_cond:
            return 1
        if short_cond:
            return -1
        return 0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Signal"] = df.apply(self._make_signal_row, axis=1)
        return df

    # ---------------------------------------------------------------------
    # Back-testing using `backtesting.py`
    # ---------------------------------------------------------------------
    def backtest_strategy(self, months: int = 6) -> pd.Series:
        # Yahoo restricts intraday (≤5-min) data to ~60 days. Adjust.
        if self.interval in ("1m", "5m") and months > 2:
            print("[WARN] Limiting back-test to 2 months due to Yahoo Finance 5-minute data cap.")
            months = 2
        """Run a back-test and return statistics."""

        class _Strategy(Strategy):
            risk_per_trade = self.risk_per_trade

            def init(self):
                close = self.data.Close
                high, low = self.data.High, self.data.Low
                self.ema20 = self.I(ta.ema, close, 20)
                self.ema50 = self.I(ta.ema, close, 50)
                self.ema100 = self.I(ta.ema, close, 100)
                self.vwap = self.I(lambda h, l, c, v: pd.Series(((((h + l + c) / 3) * v).cumsum()) / v.cumsum()),
                                   high, low, close, self.data.Volume)
                self.vwap_slope = self.I(lambda x: x.diff(), self.vwap)
                self.rsi = self.I(ta.rsi, close, 14)
                self.atr = self.I(ta.atr, high, low, close, 14)

            def next(self):
                if len(self.data) < 100:
                    return

                # Position sizing
                atr_val = self.atr[-1]
                if np.isnan(atr_val) or atr_val == 0:
                    return
                risk_amt = self.equity * (self.risk_per_trade / 100)
                size = risk_amt / atr_val

                # Entry conditions
                long_cond = (
                    crossover(self.ema20, self.ema50)
                    and self.ema50[-1] > self.ema100[-1]
                    and self.data.Close[-1] > self.vwap[-1]
                    and self.vwap_slope[-1] > 0
                    and self.rsi[-1] > 50
                )
                short_cond = (
                    crossover(self.ema50, self.ema20)
                    and self.ema50[-1] < self.ema100[-1]
                    and self.data.Close[-1] < self.vwap[-1]
                    and self.vwap_slope[-1] < 0
                    and self.rsi[-1] < 50
                )

                # Position exits configured via ATR multiples
                if long_cond:
                    sl = self.data.Close[-1] - atr_val
                    tp1 = self.data.Close[-1] + atr_val * 1.2
                    tp2 = self.data.Close[-1] + atr_val * 2.0
                    self.buy(size=size, sl=sl, tp=tp1)
                    # Partial exit handled manually – not natively supported
                elif short_cond:
                    sl = self.data.Close[-1] + atr_val
                    tp1 = self.data.Close[-1] - atr_val * 1.2
                    tp2 = self.data.Close[-1] - atr_val * 2.0
                    self.sell(size=size, sl=sl, tp=tp1)

        # Download approx `months` of data, add margin
        period_days = int(months * 30 + 10)
        if self.interval in ("1m", "5m"):
            period_days = min(period_days, 59)  # Yahoo limit for 5-minute candles
        df = self.fetch_data(period=f"{period_days}d")
        # Diagnostics prior to indicator calc
        try:
            print(f"[DEBUG] Pre-indicator df shape: {df.shape}; columns nlevels: {getattr(df.columns, 'nlevels', 1)}")
            print(f"[DEBUG] Pre-indicator columns: {list(df.columns)[:10]}")
        except Exception:
            pass
        df = self.calculate_indicators(df)
        try:
            print(f"[DEBUG] Post-indicator df shape: {df.shape}")
        except Exception:
            pass

        # Ensure single-level columns before selecting OHLCV
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
                print("[DEBUG] Flattened columns in backtest to first level")
            except Exception as e:
                print(f"[WARN] Could not flatten columns in backtest: {e}")

        # Prepare clean OHLCV for backtesting (timezone-naive)
        bt_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if getattr(bt_df.index, "tz", None) is not None:
            bt_df.index = bt_df.index.tz_localize(None)
        # Ensure numeric dtypes and drop rows with missing OHLCV
        bt_df = bt_df.apply(pd.to_numeric, errors="coerce")
        na_counts = bt_df.isna().sum()
        bt_df = bt_df.dropna()
        print(f"[DEBUG] Backtest DF shape: {bt_df.shape}, cols: {list(bt_df.columns)} | dtypes=\n{bt_df.dtypes}")
        print(f"[DEBUG] OHLCV NaN counts before drop: \n{na_counts}")
        if bt_df.empty:
            print("[ERROR] Prepared OHLCV DataFrame is empty after cleaning. Aborting backtest.")
            return pd.Series()

        bt = Backtest(bt_df, _Strategy, commission=0.0005, cash=100_000, margin=1.0)
        stats = bt.run()
        print(stats)

        # Persist results for inspection even if console output is suppressed
        try:
            stats_path = "backtest_stats.csv"
            stats.to_csv(stats_path)
            print(f"[INFO] Saved stats to {stats_path}")
        except Exception as e:
            print(f"[WARN] Could not save stats CSV: {e}")

        try:
            trades = getattr(stats, "_trades", None)
            if trades is not None:
                trades_path = "backtest_trades.csv"
                trades.to_csv(trades_path, index=False)
                print(f"[INFO] Saved trades to {trades_path}")
        except Exception as e:
            print(f"[WARN] Could not save trades CSV: {e}")

        # Disabled plotting to keep CLI run non-blocking
        # bt.plot(open_browser=False)
        print("[INFO] Backtest completed.")
        return stats

    # ---------------------------------------------------------------------
    # Live trading – prints signals every minute; integrate broker API as needed
    # ---------------------------------------------------------------------
    def live_trading(self) -> None:
        print("[INFO] Starting live signal engine…")
        last_checked = None
        while True:
            now = india_now()
            if not is_market_open(now):
                print("[INFO] Market closed – sleeping 60 s")
                time.sleep(60)
                continue

            if last_checked and (now - last_checked).seconds < 50:
                time.sleep(10)
                continue

            try:
                df = self.fetch_data(period="1d")
                df = self.calculate_indicators(df)
                df = self.generate_signals(df)
                latest = df.iloc[-1]
                signal = latest.Signal
                price = latest.Close
                if signal == 1:
                    print(f"{now}: BUY signal | Price: {price:.2f}")
                    # PLACE_ORDER_FUNCTION('BUY', price)
                elif signal == -1:
                    print(f"{now}: SELL signal | Price: {price:.2f}")
                    # PLACE_ORDER_FUNCTION('SELL', price)
                else:
                    print(f"{now}: No signal | Price: {price:.2f}")
            except Exception as exc:
                print(f"[ERROR] Live loop exception: {exc}")

            last_checked = now
            time.sleep(60)

###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    # Using daily candles for testing since Yahoo's 1m/5m data is unreliable
    bot = IntradayTradingBot(symbol="RELIANCE.NS", interval="1d", risk_per_trade=1.0)

    print("\n======== 6-Month Back-test (Daily Candles) =========")
    bot.backtest_strategy(months=6)

    # Uncomment to start live signal generation
    # print("\n======== Starting Live Signal Generation =========")
    # bot.live_trading()
