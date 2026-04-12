"""
Candlestick pattern detection, swing point identification,
trend classification, and support/resistance level detection.
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CANDLE PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect common candlestick patterns.

    Returns a copy of df with boolean columns for each pattern
    and a summary column 'pattern' with the pattern name (or None).
    """
    data = df.copy()
    o, h, l, c = data["Open"], data["High"], data["Low"], data["Close"]

    body = (c - o).abs()
    total_range = h - l
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    bullish = c > o
    bearish = c < o

    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_body = (prev_c - prev_o).abs()
    prev_bullish = prev_c > prev_o
    prev_bearish = prev_c < prev_o
    prev_range = h.shift(1) - l.shift(1)

    # ATR for context
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # ── Doji ──
    data["doji"] = (body <= total_range * 0.1) & (total_range > 0)

    # ── Hammer ──
    data["hammer"] = (lower_wick >= body * 2) & (upper_wick <= body * 0.5) & (total_range > atr * 0.5)

    # ── Inverted Hammer ──
    data["inv_hammer"] = (upper_wick >= body * 2) & (lower_wick <= body * 0.5) & (total_range > atr * 0.5)

    # ── Shooting Star ──
    data["shooting_star"] = (upper_wick >= body * 2) & (lower_wick <= body * 0.5) & (total_range > atr * 0.5)

    # ── Bullish Engulfing ──
    data["bull_engulf"] = prev_bearish & bullish & (c > prev_o) & (o <= prev_c) & (body > prev_body)

    # ── Bearish Engulfing ──
    data["bear_engulf"] = prev_bullish & bearish & (c < prev_o) & (o >= prev_c) & (body > prev_body)

    # ── Morning Star ──
    o2, c2 = o.shift(2), c.shift(2)
    body1 = (c.shift(1) - o.shift(1)).abs()
    range1 = h.shift(1) - l.shift(1)
    data["morning_star"] = (
        (c2 < o2)
        & (body1 < range1 * 0.3)
        & bullish
        & (c > (o2 + c2) / 2)
    )

    # ── Evening Star ──
    data["evening_star"] = (
        (c2 > o2)
        & (body1 < range1 * 0.3)
        & bearish
        & (c < (o2 + c2) / 2)
    )

    # ── Bullish Pin Bar ──
    data["bull_pin"] = (lower_wick >= total_range * 0.6) & (upper_wick <= total_range * 0.15) & (body <= total_range * 0.35)

    # ── Bearish Pin Bar ──
    data["bear_pin"] = (upper_wick >= total_range * 0.6) & (lower_wick <= total_range * 0.15) & (body <= total_range * 0.35)

    # ── Bullish Harami ──
    data["bull_harami"] = prev_bearish & bullish & (o > prev_c) & (c < prev_o) & (body < prev_body * 0.6)

    # ── Bearish Harami ──
    data["bear_harami"] = prev_bullish & bearish & (o < prev_c) & (c > prev_o) & (body < prev_body * 0.6)

    # ── Three White Soldiers ──
    data["three_soldiers"] = (
        (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        & bullish & prev_bullish & (c.shift(2) > o.shift(2))
        & (body > total_range * 0.5) & (prev_body > prev_range * 0.5)
    )

    # ── Three Black Crows ──
    data["three_crows"] = (
        (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        & bearish & prev_bearish & (c.shift(2) < o.shift(2))
        & (body > total_range * 0.5) & (prev_body > prev_range * 0.5)
    )

    # Summary column
    pattern_cols = {
        "doji": ("Doji", "neutral"),
        "hammer": ("Hammer", "bullish"),
        "inv_hammer": ("Inv Hammer", "bullish"),
        "shooting_star": ("Shooting Star", "bearish"),
        "bull_engulf": ("Bull Engulfing", "bullish"),
        "bear_engulf": ("Bear Engulfing", "bearish"),
        "morning_star": ("Morning Star", "bullish"),
        "evening_star": ("Evening Star", "bearish"),
        "bull_pin": ("Bull Pin Bar", "bullish"),
        "bear_pin": ("Bear Pin Bar", "bearish"),
        "bull_harami": ("Bull Harami", "bullish"),
        "bear_harami": ("Bear Harami", "bearish"),
        "three_soldiers": ("3 White Soldiers", "bullish"),
        "three_crows": ("3 Black Crows", "bearish"),
    }

    patterns = []
    biases = []
    for _, row in data.iterrows():
        names = []
        bias = "neutral"
        for col, (name, b) in pattern_cols.items():
            if row.get(col, False):
                names.append(name)
                if b != "neutral":
                    bias = b
        patterns.append(", ".join(names) if names else None)
        biases.append(bias if names else None)

    data["pattern"] = patterns
    data["pattern_bias"] = biases

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SWING POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Identify swing highs and swing lows.

    Returns df with 'swing_high' and 'swing_low' columns
    (price value at pivot, NaN otherwise).
    """
    data = df.copy()
    highs = data["High"].values
    lows = data["Low"].values
    n = len(data)

    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        # Swing high: highest in window
        if highs[i] == max(highs[i - lookback : i + lookback + 1]):
            sh[i] = highs[i]
        # Swing low: lowest in window
        if lows[i] == min(lows[i - lookback : i + lookback + 1]):
            sl[i] = lows[i]

    data["swing_high"] = sh
    data["swing_low"] = sl
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# TREND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_trend(df: pd.DataFrame) -> dict:
    """
    Classify the current trend based on the last two swing highs and lows.

    Returns dict with trend, higher_highs, higher_lows, last values.
    """
    swing_highs = df.loc[df["swing_high"].notna(), "swing_high"]
    swing_lows = df.loc[df["swing_low"].notna(), "swing_low"]

    result = {
        "trend": "Sideways",
        "higher_highs": False,
        "higher_lows": False,
        "lower_highs": False,
        "lower_lows": False,
        "last_swing_high": None,
        "last_swing_low": None,
    }

    if len(swing_highs) >= 2:
        sh1, sh2 = swing_highs.iloc[-1], swing_highs.iloc[-2]
        result["last_swing_high"] = round(sh1, 2)
        result["higher_highs"] = sh1 > sh2
        result["lower_highs"] = sh1 < sh2

    if len(swing_lows) >= 2:
        sl1, sl2 = swing_lows.iloc[-1], swing_lows.iloc[-2]
        result["last_swing_low"] = round(sl1, 2)
        result["higher_lows"] = sl1 > sl2
        result["lower_lows"] = sl1 < sl2

    if result["higher_highs"] and result["higher_lows"]:
        result["trend"] = "Uptrend"
    elif result["lower_highs"] and result["lower_lows"]:
        result["trend"] = "Downtrend"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORT & RESISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def find_support_resistance(
    df: pd.DataFrame,
    tolerance_atr_mult: float = 0.5,
    max_levels: int = 5,
) -> dict:
    """
    Cluster swing highs/lows into support and resistance zones.

    Returns dict with 'support' and 'resistance' lists,
    each containing (level, touch_count) tuples sorted by strength.
    """
    # ATR for tolerance
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    tol = atr * tolerance_atr_mult

    current_price = df["Close"].iloc[-1]

    # Gather all pivot values
    pivots_high = df.loc[df["swing_high"].notna(), "swing_high"].tolist()
    pivots_low = df.loc[df["swing_low"].notna(), "swing_low"].tolist()

    # Cluster pivots
    def cluster(values, tolerance):
        if not values:
            return []
        values = sorted(values)
        clusters = []
        current_cluster = [values[0]]
        for v in values[1:]:
            if v - np.mean(current_cluster) <= tolerance:
                current_cluster.append(v)
            else:
                clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [v]
        clusters.append((np.mean(current_cluster), len(current_cluster)))
        return clusters

    all_pivots = pivots_high + pivots_low
    clusters = cluster(all_pivots, tol)

    # Classify as support (below price) or resistance (above price)
    support = sorted(
        [(round(lvl, 2), cnt) for lvl, cnt in clusters if lvl < current_price],
        key=lambda x: x[1], reverse=True,
    )[:max_levels]

    resistance = sorted(
        [(round(lvl, 2), cnt) for lvl, cnt in clusters if lvl >= current_price],
        key=lambda x: x[1], reverse=True,
    )[:max_levels]

    return {"support": support, "resistance": resistance}
