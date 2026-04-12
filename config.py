"""
Default configuration for the Trading AI Dashboard.
"""

# Default watchlist - popular swing trading stocks
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "DIS",
    "NFLX", "AMD", "PYPL", "BA", "INTC",
]

# Indicator default parameters
INDICATOR_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "volume_multiplier": 1.5,
    "volume_ma_period": 20,
    "atr_period": 14,
    "sl_atr_multiplier": 1.5,
    "tp_atr_multiplier": 2.5,
    "min_signals": 3,
}

# Timeframe options
TIMEFRAMES = {
    "1 Hour": "1h",
    "4 Hours": "4h",      # not available on yfinance for all periods
    "1 Day": "1d",         # default for swing trading
    "1 Week": "1wk",
}

DEFAULT_TIMEFRAME = "1d"

# Data period (how far back to fetch)
DATA_PERIOD = "6mo"
