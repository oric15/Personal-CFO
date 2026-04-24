import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import glob
import os
import subprocess
import sys

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_PATH       = "data/raw"
PROCESSED_PATH = "data/processed"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_latest_file(pattern: str, folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        return pd.DataFrame()
    latest = max(files, key=os.path.getmtime)
    return pd.read_csv(latest)


def load_signals() -> pd.DataFrame:
    path = os.path.join(PROCESSED_PATH, "signals.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_news() -> pd.DataFrame:
    path = os.path.join(PROCESSED_PATH, "news_with_sentiment.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def signal_color(signal: str) -> str:
    return {"BUY": "#00c48c", "SELL": "#ff4d4d", "WATCH": "#ffc107"}.get(signal, "#888")


def signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "WATCH": "🟡"}.get(signal, "⚪")


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🚦 Personal CFO")
st.caption("Sentiment-driven signals from Yahoo Finance news · Powered by FinBERT")
st.divider()

# ── Refresh Button ────────────────────────────────────────────────────────────

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])

with col_btn1:
    if st.button("🔄 Refresh Data", width="stretch"):
        with st.spinner("Collecting new data..."):
            subprocess.run([sys.executable, "yahoo_finance.py"])
        with st.spinner("Running sentiment analysis..."):
            subprocess.run([sys.executable, "sentiment.py"])
        with st.spinner("Generating signals..."):
            subprocess.run([sys.executable, "composite.py"])
        st.success("Done! Data refreshed.")
        st.rerun()

with col_btn2:
    st.write("")  # spacer

# ── Load Data ─────────────────────────────────────────────────────────────────

signals_df = load_signals()
news_df    = load_news()
prices_df  = load_latest_file("prices_*.csv", RAW_PATH)

if signals_df.empty:
    st.warning("No signal data found. Run the pipeline first or click 'Refresh Data'.")
    st.stop()

# ── Signal Cards ──────────────────────────────────────────────────────────────

st.subheader("📊 Current Signals")

cols = st.columns(len(signals_df))

for i, (_, row) in enumerate(signals_df.iterrows()):
    with cols[i]:
        color  = signal_color(row["signal"])
        emoji  = signal_emoji(row["signal"])

        st.markdown(f"""
        <div style="
            background: #1e1e2e;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        ">
            <div style="font-size: 2rem; font-weight: bold; color: white">{row['ticker']}</div>
            <div style="font-size: 2.5rem; margin: 8px 0">{emoji}</div>
            <div style="font-size: 1.4rem; font-weight: bold; color: {color}">{row['signal']}</div>
            <hr style="border-color: #333; margin: 12px 0">
            <div style="color: #aaa; font-size: 0.85rem">Sentiment: <b style="color:white">{row['sentiment_label'].upper()}</b></div>
            <div style="color: #aaa; font-size: 0.85rem">Trend: <b style="color:white">{row['trend_label'].upper()} ({row['trend_pct']:+.2f}%)</b></div>
            <div style="color: #aaa; font-size: 0.85rem">Price: <b style="color:white">${row['price_end']}</b></div>
            <div style="color: #666; font-size: 0.75rem; margin-top: 10px; font-style: italic">{row['reasoning']}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Ticker Detail ─────────────────────────────────────────────────────────────

st.subheader("🔍 Ticker Deep Dive")

selected = st.selectbox("Select a ticker", signals_df["ticker"].tolist())

col_left, col_right = st.columns([3, 2])

# Price chart
with col_left:
    st.markdown(f"#### 📈 Price History — {selected}")

    if not prices_df.empty:
        ticker_prices = prices_df[prices_df["ticker"] == selected].copy()
        ticker_prices["date"] = pd.to_datetime(ticker_prices["date"])

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=ticker_prices["date"],
            open=ticker_prices["open"],
            high=ticker_prices["high"],
            low=ticker_prices["low"],
            close=ticker_prices["close"],
            name=selected,
            increasing_line_color="#00c48c",
            decreasing_line_color="#ff4d4d",
        ))

        fig.update_layout(
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font_color="white",
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
        )

        fig.update_xaxes(gridcolor="#333")
        fig.update_yaxes(gridcolor="#333")

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No price data available.")

# News + Sentiment
with col_right:
    st.markdown(f"#### 📰 Latest News — {selected}")

    if not news_df.empty:
        ticker_news = news_df[news_df["ticker"] == selected].head(8)

        for _, article in ticker_news.iterrows():
            sentiment = article.get("sentiment", "neutral")
            score     = article.get("sentiment_score", 0)
            title     = article.get("title", "No title")
            publisher = article.get("publisher", "")
            link      = article.get("link", "#")

            badge_color = {"positive": "#00c48c", "negative": "#ff4d4d", "neutral": "#ffc107"}.get(sentiment, "#888")
            badge_label = sentiment.upper()

            st.markdown(f"""
            <div style="
                background: #1e1e2e;
                border-left: 3px solid {badge_color};
                padding: 10px 14px;
                margin-bottom: 10px;
                border-radius: 4px;
            ">
                <div style="font-size: 0.85rem; color: white; margin-bottom: 4px">
                    <a href="{link}" target="_blank" style="color: white; text-decoration: none">{title}</a>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center">
                    <span style="color: #666; font-size: 0.75rem">{publisher}</span>
                    <span style="
                        background: {badge_color}22;
                        color: {badge_color};
                        font-size: 0.7rem;
                        padding: 2px 8px;
                        border-radius: 10px;
                        font-weight: bold;
                    ">{badge_label} {score:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news data available.")

st.divider()

# ── Sentiment Distribution ────────────────────────────────────────────────────

st.subheader("🧠 Sentiment Distribution")

if not news_df.empty:
    cols_sent = st.columns(len(signals_df["ticker"].tolist()))

    for i, ticker in enumerate(signals_df["ticker"].tolist()):
        with cols_sent[i]:
            subset = news_df[news_df["ticker"] == ticker]
            counts = subset["sentiment"].value_counts()

            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.5,
                marker_colors=["#00c48c", "#ff4d4d", "#ffc107"],
            ))

            fig.update_layout(
                title=ticker,
                plot_bgcolor="#1e1e2e",
                paper_bgcolor="#1e1e2e",
                font_color="white",
                margin=dict(l=0, r=0, t=40, b=0),
                height=250,
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("⚠️ This is not financial advice. For educational and portfolio purposes only.")