"""
Technical indicator calculations for the Trading AI Dashboard.
Mirrors the logic used in the TradingView Pine Script indicator.
"""

import pandas as pd
import numpy as np


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply all indicators and generate buy/sell signals.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume
    params : dict of indicator parameters (see config.INDICATOR_PARAMS)

    Returns
    -------
    DataFrame with all original columns plus indicator and signal columns.
    """
    data = df.copy()

    # --- EMAs ---
    data["ema_fast"] = calculate_ema(data["Close"], params["ema_fast"])
    data["ema_slow"] = calculate_ema(data["Close"], params["ema_slow"])
    data["ema_bullish"] = data["ema_fast"] > data["ema_slow"]

    # EMA crossovers
    prev_bull = data["ema_bullish"].shift(1).fillna(False)
    data["ema_bull_cross"] = (~prev_bull) & data["ema_bullish"]
    data["ema_bear_cross"] = prev_bull & (~data["ema_bullish"])

    # --- RSI ---
    data["rsi"] = calculate_rsi(data["Close"], params["rsi_period"])
    prev_rsi = data["rsi"].shift(1)

    data["rsi_bull_signal"] = (
        ((prev_rsi <= params["rsi_oversold"]) & (data["rsi"] > params["rsi_oversold"]))
        | (
            (data["rsi"] > params["rsi_oversold"])
            & (data["rsi"] < 50)
            & (data["rsi"] > prev_rsi)
        )
    )
    data["rsi_bear_signal"] = (
        ((prev_rsi >= params["rsi_overbought"]) & (data["rsi"] < params["rsi_overbought"]))
        | (
            (data["rsi"] < params["rsi_overbought"])
            & (data["rsi"] > 50)
            & (data["rsi"] < prev_rsi)
        )
    )

    # --- MACD ---
    macd_df = calculate_macd(
        data["Close"], params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )
    data["macd"] = macd_df["macd"]
    data["macd_signal"] = macd_df["signal"]
    data["macd_hist"] = macd_df["histogram"]

    macd_bull = data["macd"] > data["macd_signal"]
    prev_macd_bull = macd_bull.shift(1).fillna(False)
    data["macd_bull_cross"] = (~prev_macd_bull) & macd_bull
    data["macd_bear_cross"] = prev_macd_bull & (~macd_bull)

    # --- Volume ---
    data["volume_ma"] = data["Volume"].rolling(window=params["volume_ma_period"]).mean()
    data["high_volume"] = data["Volume"] > data["volume_ma"] * params["volume_multiplier"]

    # --- ATR ---
    data["atr"] = calculate_atr(data, params["atr_period"])

    # --- Aggregate signals ---
    data["buy_count"] = (
        data["ema_bull_cross"].astype(int)
        + data["rsi_bull_signal"].astype(int)
        + data["macd_bull_cross"].astype(int)
        + data["high_volume"].astype(int)
    )
    data["sell_count"] = (
        data["ema_bear_cross"].astype(int)
        + data["rsi_bear_signal"].astype(int)
        + data["macd_bear_cross"].astype(int)
        + data["high_volume"].astype(int)
    )

    min_sig = params["min_signals"]
    data["buy_signal"] = data["buy_count"] >= min_sig
    data["sell_signal"] = data["sell_count"] >= min_sig

    data["signal_strength"] = data[["buy_count", "sell_count"]].max(axis=1)

    # --- Stop-loss / Take-profit levels ---
    data["sl_buy"] = data["Close"] - data["atr"] * params["sl_atr_multiplier"]
    data["tp_buy"] = data["Close"] + data["atr"] * params["tp_atr_multiplier"]
    data["sl_sell"] = data["Close"] + data["atr"] * params["sl_atr_multiplier"]
    data["tp_sell"] = data["Close"] - data["atr"] * params["tp_atr_multiplier"]

    return data


def get_latest_signal(data: pd.DataFrame) -> dict:
    """Return a summary of the most recent signal state."""
    if data.empty:
        return {"signal": "NO DATA", "strength": 0}

    last = data.iloc[-1]
    if last["buy_signal"]:
        signal = "STRONG BUY" if last["buy_count"] == 4 else "BUY"
        strength = int(last["buy_count"] * 25)
    elif last["sell_signal"]:
        signal = "STRONG SELL" if last["sell_count"] == 4 else "SELL"
        strength = int(last["sell_count"] * 25)
    else:
        # Score based on how close the dominant side is to triggering
        bull_score = int(last["buy_count"] * 25)
        bear_score = int(last["sell_count"] * 25)
        if bull_score > bear_score:
            signal = "LEANING BULLISH"
            strength = bull_score
        elif bear_score > bull_score:
            signal = "LEANING BEARISH"
            strength = bear_score
        else:
            signal = "NEUTRAL"
            strength = 0

    return {
        "signal": signal,
        "strength": strength,
        "rsi": round(last["rsi"], 2) if pd.notna(last["rsi"]) else None,
        "ema_trend": "Bullish" if last["ema_bullish"] else "Bearish",
        "macd_trend": "Bullish" if last["macd"] > last["macd_signal"] else "Bearish",
        "volume": "High" if last["high_volume"] else "Normal",
        "close": round(last["Close"], 2),
        "atr": round(last["atr"], 2) if pd.notna(last["atr"]) else None,
        "sl_buy": round(last["sl_buy"], 2) if pd.notna(last["sl_buy"]) else None,
        "tp_buy": round(last["tp_buy"], 2) if pd.notna(last["tp_buy"]) else None,
    }
