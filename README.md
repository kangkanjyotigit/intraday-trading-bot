# Intraday trading bot (Indian equities)

A two-mode intraday system for Indian equity markets: historical backtesting and
near-real-time signal generation.

## Modes

1. **Backtest** — replays the last N months of data through the strategy and
   reports performance.
2. **Live signals** — runs the same strategy against current data and emits
   entry/exit signals. Order placement is deliberately left as a stub
   (`PLACE_ORDER_FUNCTION`) so no broker credentials live in this repo.

## Stack

`pandas` · `numpy` · `pandas-ta` (technical indicators) · `backtesting.py` ·
`yfinance` (data) · `pytz` (`Asia/Kolkata` session handling)

`intraday_strategy.pine` is the same strategy expressed for TradingView, for
visual cross-checking against the Python implementation.

## Run

```bash
pip install -r requirements.txt
python intraday_bot.py
```

## Notes

Research and backtesting code. Not financial advice; no live order routing is
included or configured.
