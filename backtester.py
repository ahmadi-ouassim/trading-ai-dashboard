"""
Backtesting engine — tests how the signal strategy performed historically.
"""

import pandas as pd
import numpy as np
from indicators import generate_signals
from scanner import fetch_stock_data
from config import INDICATOR_PARAMS, DATA_PERIOD


def run_backtest(
    ticker: str,
    params: dict | None = None,
    period: str = "2y",
    interval: str = "1d",
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.02,
) -> dict:
    """
    Backtest the signal strategy on historical data.

    Returns a dict with:
      - trades: list of trade dicts
      - stats: summary statistics
      - equity_curve: Series of portfolio value over time
    """
    params = params or INDICATOR_PARAMS.copy()
    # For backtesting, allow user-set min_signals (default 2 for more trades)
    bt_params = params.copy()

    df = fetch_stock_data(ticker, period=period, interval=interval)
    if df.empty or len(df) < 60:
        return {"trades": [], "stats": {}, "equity_curve": pd.Series(dtype=float)}

    data = generate_signals(df, bt_params)
    data = data.dropna(subset=["atr"])

    capital = initial_capital
    position = 0  # shares held
    entry_price = 0.0
    entry_date = None
    sl_price = 0.0
    tp_price = 0.0
    trades = []
    equity = []

    for i in range(len(data)):
        row = data.iloc[i]
        date = data.index[i]
        price = row["Close"]

        # Track equity
        value = capital + position * price
        equity.append({"date": date, "equity": value})

        # If in a position, check exit conditions
        if position > 0:
            # Stop loss hit
            if row["Low"] <= sl_price:
                exit_price = sl_price
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": position,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / entry_price - 1) * 100, 2),
                    "exit_reason": "Stop Loss",
                })
                position = 0
                continue

            # Take profit hit
            if row["High"] >= tp_price:
                exit_price = tp_price
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": position,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / entry_price - 1) * 100, 2),
                    "exit_reason": "Take Profit",
                })
                position = 0
                continue

            # Sell signal — exit
            if row["sell_signal"]:
                exit_price = price
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": position,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / entry_price - 1) * 100, 2),
                    "exit_reason": "Sell Signal",
                })
                position = 0
                continue

        # If no position, check entry conditions
        if position == 0 and row["buy_signal"]:
            risk_amount = capital * risk_per_trade
            atr_sl = row["atr"] * params.get("sl_atr_multiplier", 1.5)
            if atr_sl > 0:
                shares = int(risk_amount / atr_sl)
                if shares > 0 and shares * price <= capital:
                    entry_price = price
                    entry_date = date
                    sl_price = price - atr_sl
                    tp_price = price + row["atr"] * params.get("tp_atr_multiplier", 2.5)
                    position = shares
                    capital -= shares * price

    # Close any remaining position at last price
    if position > 0:
        last_price = data.iloc[-1]["Close"]
        pnl = (last_price - entry_price) * position
        capital += position * last_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": data.index[-1],
            "entry_price": round(entry_price, 2),
            "exit_price": round(last_price, 2),
            "shares": position,
            "pnl": round(pnl, 2),
            "pnl_pct": round((last_price / entry_price - 1) * 100, 2),
            "exit_reason": "Still Open",
        })

    # Build equity curve
    equity_df = pd.DataFrame(equity).set_index("date")

    # Calculate stats
    stats = _calculate_stats(trades, initial_capital, equity_df)

    return {
        "trades": trades,
        "stats": stats,
        "equity_curve": equity_df,
    }


def _calculate_stats(trades: list, initial_capital: float, equity: pd.DataFrame) -> dict:
    """Calculate backtest summary statistics."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "total_return_pct": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_drawdown_pct": 0,
            "profit_factor": 0,
            "best_trade": 0,
            "worst_trade": 0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    final_equity = equity["equity"].iloc[-1] if not equity.empty else initial_capital

    # Max drawdown
    if not equity.empty:
        peak = equity["equity"].expanding().max()
        drawdown = (equity["equity"] - peak) / peak * 100
        max_dd = drawdown.min()
    else:
        max_dd = 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((final_equity / initial_capital - 1) * 100, 2),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
    }
