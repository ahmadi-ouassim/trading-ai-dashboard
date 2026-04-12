"""
Trading AI Dashboard – Streamlit web app for swing trading analysis.

DISCLAIMER: This tool is for EDUCATIONAL purposes only.
It does NOT constitute financial advice. Always do your own research.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DEFAULT_WATCHLIST, INDICATOR_PARAMS, TIMEFRAMES, DEFAULT_TIMEFRAME, DATA_PERIOD
from scanner import scan_watchlist, get_analysed_data
from candle_patterns import detect_patterns, find_swing_points, classify_trend, find_support_resistance

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading AI - Swing Signal Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Session state defaults ───────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
if "params" not in st.session_state:
    st.session_state.params = INDICATOR_PARAMS.copy()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trading AI")
    st.caption("Swing Trading Signal Engine")

    page = st.radio("Navigate", ["Scanner", "Stock Analysis", "Settings"], label_visibility="collapsed")

    st.divider()
    st.warning(
        "**Disclaimer:** This tool is for educational purposes only. "
        "It does NOT constitute financial advice. Past performance does not "
        "guarantee future results. Always do your own research.",
        icon="⚠️",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Scanner
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Scanner":
    st.header("Stock Scanner")
    st.write("Scanning your watchlist for swing trading signals…")

    tf_label = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=list(TIMEFRAMES.values()).index(DEFAULT_TIMEFRAME))
    interval = TIMEFRAMES[tf_label]

    if st.button("Scan Now", type="primary"):
        with st.spinner("Fetching data & analysing signals…"):
            results = scan_watchlist(
                st.session_state.watchlist,
                params=st.session_state.params,
                interval=interval,
            )

        if results.empty:
            st.error("No results. Check your watchlist in Settings.")
        else:
            # Color the signal column
            def color_signal(val):
                if "BUY" in str(val):
                    return "background-color: #1b5e20; color: white"
                elif "SELL" in str(val):
                    return "background-color: #b71c1c; color: white"
                elif "BULLISH" in str(val):
                    return "background-color: #2e7d32; color: white"
                elif "BEARISH" in str(val):
                    return "background-color: #c62828; color: white"
                return ""

            display_cols = [c for c in ["ticker", "company", "signal", "strength", "close", "rsi", "ema_trend", "macd_trend", "volume", "atr"] if c in results.columns]
            styled = results[display_cols].style.map(color_signal, subset=["signal"])
            st.dataframe(styled, use_container_width=True, height=600)

            # Quick stats
            buy_count = results["signal"].str.contains("BUY", na=False).sum()
            sell_count = results["signal"].str.contains("SELL", na=False).sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Buy Signals", buy_count)
            col2.metric("Sell Signals", sell_count)
            col3.metric("Stocks Scanned", len(results))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Stock Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Stock Analysis":
    st.header("Stock Analysis")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    with col_b:
        tf_label = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=list(TIMEFRAMES.values()).index(DEFAULT_TIMEFRAME))
        interval = TIMEFRAMES[tf_label]

    if st.button("Analyse", type="primary") or ticker:
        with st.spinner(f"Analysing {ticker}…"):
            data = get_analysed_data(ticker, params=st.session_state.params, interval=interval)

        if data.empty:
            st.error(f"Could not fetch data for **{ticker}**. Check the symbol and try again.")
        else:
            # Fetch and display company name
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                company_name = info.get("shortName") or info.get("longName") or ticker
            except Exception:
                company_name = ticker
            st.subheader(f"{company_name} ({ticker})")

            latest = data.iloc[-1]

            # ── Summary metrics ──
            sig = "NEUTRAL"
            if latest["buy_signal"]:
                sig = "STRONG BUY" if latest["buy_count"] == 4 else "BUY"
            elif latest["sell_signal"]:
                sig = "STRONG SELL" if latest["sell_count"] == 4 else "SELL"

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Signal", sig)
            m2.metric("Price", f"${latest['Close']:.2f}")
            m3.metric("RSI", f"{latest['rsi']:.1f}" if pd.notna(latest['rsi']) else "N/A")
            m4.metric("EMA Trend", "Bullish" if latest["ema_bullish"] else "Bearish")
            m5.metric("MACD", "Bullish" if latest["macd"] > latest["macd_signal"] else "Bearish")

            # ── Risk levels ──
            if pd.notna(latest["atr"]):
                r1, r2, r3 = st.columns(3)
                r1.metric("ATR", f"${latest['atr']:.2f}")
                r2.metric("Stop Loss (buy)", f"${latest['sl_buy']:.2f}")
                r3.metric("Take Profit (buy)", f"${latest['tp_buy']:.2f}")

            # ── Chart ──
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=("Price + EMAs", "RSI", "MACD", "Volume"),
            )

            idx = data.index

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=idx, open=data["Open"], high=data["High"],
                low=data["Low"], close=data["Close"], name="Price",
            ), row=1, col=1)

            # EMAs
            fig.add_trace(go.Scatter(x=idx, y=data["ema_fast"], name=f"EMA {st.session_state.params['ema_fast']}", line=dict(color="dodgerblue", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=idx, y=data["ema_slow"], name=f"EMA {st.session_state.params['ema_slow']}", line=dict(color="orange", width=1.5)), row=1, col=1)

            # Buy / Sell markers
            buys = data[data["buy_signal"]]
            sells = data[data["sell_signal"]]
            fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"] * 0.995, mode="markers", marker=dict(symbol="triangle-up", size=12, color="lime"), name="BUY"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.005, mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"), name="SELL"), row=1, col=1)

            # ── Support & Resistance lines on price chart ──
            data_with_swings = find_swing_points(data, lookback=5)
            sr = find_support_resistance(data_with_swings, tolerance_atr_mult=0.5, max_levels=5)

            for lvl, cnt in sr["support"]:
                fig.add_hline(y=lvl, line_dash="dash", line_color="rgba(0,200,83,0.5)", line_width=2,
                              annotation_text=f"S {lvl} ({cnt})", annotation_position="bottom left",
                              annotation_font_color="lime", row=1, col=1)

            for lvl, cnt in sr["resistance"]:
                fig.add_hline(y=lvl, line_dash="dash", line_color="rgba(255,23,68,0.5)", line_width=2,
                              annotation_text=f"R {lvl} ({cnt})", annotation_position="top left",
                              annotation_font_color="red", row=1, col=1)

            # ── Swing point markers ──
            sh = data_with_swings[data_with_swings["swing_high"].notna()]
            sl_pts = data_with_swings[data_with_swings["swing_low"].notna()]
            fig.add_trace(go.Scatter(x=sh.index, y=sh["swing_high"], mode="markers", marker=dict(symbol="diamond", size=7, color="red"), name="Swing High", showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sl_pts.index, y=sl_pts["swing_low"], mode="markers", marker=dict(symbol="diamond", size=7, color="green"), name="Swing Low", showlegend=False), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=idx, y=data["rsi"], name="RSI", line=dict(color="purple", width=1.5)), row=2, col=1)
            fig.add_hline(y=st.session_state.params["rsi_overbought"], line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=st.session_state.params["rsi_oversold"], line_dash="dash", line_color="green", row=2, col=1)

            # MACD
            macd_colors = ["green" if v >= 0 else "red" for v in data["macd_hist"]]
            fig.add_trace(go.Bar(x=idx, y=data["macd_hist"], name="MACD Hist", marker_color=macd_colors), row=3, col=1)
            fig.add_trace(go.Scatter(x=idx, y=data["macd"], name="MACD", line=dict(color="blue", width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=idx, y=data["macd_signal"], name="Signal", line=dict(color="orange", width=1)), row=3, col=1)

            # Volume
            vol_colors = ["green" if data["Close"].iloc[i] >= data["Open"].iloc[i] else "red" for i in range(len(data))]
            fig.add_trace(go.Bar(x=idx, y=data["Volume"], name="Volume", marker_color=vol_colors), row=4, col=1)
            fig.add_trace(go.Scatter(x=idx, y=data["volume_ma"], name="Vol MA", line=dict(color="yellow", width=1)), row=4, col=1)

            fig.update_layout(
                height=900,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=50, r=20, t=60, b=30),
            )
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # ── Trend Classification ──
            trend_info = classify_trend(data_with_swings)
            st.subheader("Trend Analysis")
            t1, t2, t3, t4 = st.columns(4)
            trend_color = {"Uptrend": "green", "Downtrend": "red", "Sideways": "gray"}.get(trend_info["trend"], "gray")
            t1.markdown(f"**Trend:** :{trend_color}[{trend_info['trend']}]")
            t2.metric("Last Swing High", f"${trend_info['last_swing_high']}" if trend_info["last_swing_high"] else "–")
            t3.metric("Last Swing Low", f"${trend_info['last_swing_low']}" if trend_info["last_swing_low"] else "–")
            hh_hl = ("HH" if trend_info["higher_highs"] else "LH") + " / " + ("HL" if trend_info["higher_lows"] else "LL")
            t4.metric("Structure", hh_hl)

            # ── Support & Resistance table ──
            st.subheader("Support & Resistance Levels")
            sr_col1, sr_col2 = st.columns(2)
            with sr_col1:
                st.markdown("**Support (below price)**")
                if sr["support"]:
                    for lvl, cnt in sorted(sr["support"], key=lambda x: x[0], reverse=True):
                        st.markdown(f"- :green[${lvl}] — {cnt} touches")
                else:
                    st.write("No support levels found")
            with sr_col2:
                st.markdown("**Resistance (above price)**")
                if sr["resistance"]:
                    for lvl, cnt in sorted(sr["resistance"], key=lambda x: x[0]):
                        st.markdown(f"- :red[${lvl}] — {cnt} touches")
                else:
                    st.write("No resistance levels found")

            # ── Candle Patterns ──
            st.subheader("Candle Patterns")
            pattern_data = detect_patterns(data)
            recent_patterns = pattern_data[pattern_data["pattern"].notna()].tail(15)

            if recent_patterns.empty:
                st.info("No candle patterns detected in recent bars.")
            else:
                pat_display = recent_patterns[["Close", "pattern", "pattern_bias"]].copy()
                pat_display.columns = ["Price", "Pattern", "Bias"]
                pat_display["Price"] = pat_display["Price"].apply(lambda x: f"${x:.2f}")
                pat_display = pat_display.iloc[::-1]

                def color_bias(val):
                    if val == "bullish":
                        return "background-color: #1b5e20; color: white"
                    elif val == "bearish":
                        return "background-color: #b71c1c; color: white"
                    return ""

                styled_pat = pat_display.style.map(color_bias, subset=["Bias"])
                st.dataframe(styled_pat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    st.header("Settings")

    # ── Watchlist ──
    st.subheader("Watchlist")
    wl_text = st.text_area(
        "Enter ticker symbols (one per line or comma-separated)",
        value=", ".join(st.session_state.watchlist),
        height=100,
    )
    if st.button("Update Watchlist"):
        tickers = [t.strip().upper() for t in wl_text.replace("\n", ",").split(",") if t.strip()]
        st.session_state.watchlist = tickers
        st.success(f"Watchlist updated: {len(tickers)} stocks")

    st.divider()

    # ── Indicator parameters ──
    st.subheader("Indicator Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**EMA Settings**")
        st.session_state.params["ema_fast"] = st.slider("Fast EMA", 5, 100, st.session_state.params["ema_fast"])
        st.session_state.params["ema_slow"] = st.slider("Slow EMA", 10, 200, st.session_state.params["ema_slow"])

        st.markdown("**RSI Settings**")
        st.session_state.params["rsi_period"] = st.slider("RSI Period", 5, 50, st.session_state.params["rsi_period"])
        st.session_state.params["rsi_overbought"] = st.slider("RSI Overbought", 60, 90, st.session_state.params["rsi_overbought"])
        st.session_state.params["rsi_oversold"] = st.slider("RSI Oversold", 10, 40, st.session_state.params["rsi_oversold"])

    with col2:
        st.markdown("**MACD Settings**")
        st.session_state.params["macd_fast"] = st.slider("MACD Fast", 5, 50, st.session_state.params["macd_fast"])
        st.session_state.params["macd_slow"] = st.slider("MACD Slow", 10, 100, st.session_state.params["macd_slow"])
        st.session_state.params["macd_signal"] = st.slider("MACD Signal", 3, 30, st.session_state.params["macd_signal"])

        st.markdown("**Volume Settings**")
        st.session_state.params["volume_multiplier"] = st.slider("Volume Multiplier", 1.0, 3.0, st.session_state.params["volume_multiplier"], 0.1)
        st.session_state.params["volume_ma_period"] = st.slider("Volume MA Period", 5, 50, st.session_state.params["volume_ma_period"])

    st.markdown("**Risk Settings (ATR)**")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.session_state.params["atr_period"] = st.slider("ATR Period", 5, 50, st.session_state.params["atr_period"])
    with r2:
        st.session_state.params["sl_atr_multiplier"] = st.slider("Stop Loss Multiplier", 0.5, 5.0, st.session_state.params["sl_atr_multiplier"], 0.1)
    with r3:
        st.session_state.params["tp_atr_multiplier"] = st.slider("Take Profit Multiplier", 1.0, 10.0, st.session_state.params["tp_atr_multiplier"], 0.1)

    st.session_state.params["min_signals"] = st.slider("Min Signals to Trigger (2-4)", 2, 4, st.session_state.params["min_signals"])

    if st.button("Reset to Defaults"):
        st.session_state.params = INDICATOR_PARAMS.copy()
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
        st.rerun()
