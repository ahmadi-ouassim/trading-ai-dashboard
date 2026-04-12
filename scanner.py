"""
Stock scanner – fetches market data and runs the signal engine across a watchlist.
"""

import yfinance as yf
import pandas as pd
from indicators import generate_signals, get_latest_signal
from config import INDICATOR_PARAMS, DEFAULT_TIMEFRAME, DATA_PERIOD


def fetch_stock_data(
    ticker: str,
    period: str = DATA_PERIOD,
    interval: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        return pd.DataFrame()
    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def scan_stock(
    ticker: str,
    params: dict | None = None,
    period: str = DATA_PERIOD,
    interval: str = DEFAULT_TIMEFRAME,
) -> dict:
    """Analyse a single stock and return its latest signal summary."""
    params = params or INDICATOR_PARAMS
    df = fetch_stock_data(ticker, period=period, interval=interval)
    if df.empty:
        return {"ticker": ticker, "signal": "NO DATA", "strength": 0}

    data = generate_signals(df, params)
    summary = get_latest_signal(data)
    summary["ticker"] = ticker

    # Fetch company name
    try:
        info = yf.Ticker(ticker).info
        summary["company"] = info.get("shortName") or info.get("longName") or ticker
    except Exception:
        summary["company"] = ticker

    return summary


def scan_watchlist(
    tickers: list[str],
    params: dict | None = None,
    period: str = DATA_PERIOD,
    interval: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Scan every ticker and return a DataFrame sorted by signal strength."""
    results = []
    for t in tickers:
        try:
            result = scan_stock(t, params=params, period=period, interval=interval)
            results.append(result)
        except Exception as exc:
            results.append({"ticker": t, "signal": f"ERROR: {exc}", "strength": 0})

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("strength", ascending=False).reset_index(drop=True)
    return df


def get_analysed_data(
    ticker: str,
    params: dict | None = None,
    period: str = DATA_PERIOD,
    interval: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Return the full DataFrame with indicators and signals for charting."""
    params = params or INDICATOR_PARAMS
    df = fetch_stock_data(ticker, period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()
    return generate_signals(df, params)
