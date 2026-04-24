import pandas as pd
import os
import glob

# ------------- Configuration -----------------------

RAW_PATH       = "data/raw"
PROCESSED_PATH = "data/processed"

PRICE_TREND_DAYS = 7   # how many days to look back for price trend

# ----------------------- Loaders ----------------------------

def load_latest_file(pattern: str, folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern}' found in {folder}")
    latest = max(files, key=os.path.getmtime)
    print(f"📂 Loaded: {latest}")
    return pd.read_csv(latest)


def load_sentiment() -> pd.DataFrame:
    path = os.path.join(PROCESSED_PATH, "news_with_sentiment.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run signals/sentiment.py first.")
    print(f"📂 Loaded: {path}")
    return pd.read_csv(path)


# ------------- Sentiment Score per Ticker -----------------------

def aggregate_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores per ticker into a single score and label.
    """
    agg = (
        sentiment_df
        .groupby("ticker")["sentiment_numeric"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_numeric": "sentiment_avg"})
    )

    def label(score):
        if score > 0.2:
            return "bullish"
        elif score < -0.2:
            return "bearish"
        else:
            return "neutral"

    agg["sentiment_label"] = agg["sentiment_avg"].apply(label)
    return agg


# --------- Price Trend per Ticker -----------------------

def compute_price_trend(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, compute the price trend over the last N days.
    Trend = % change from N days ago to today.
    """
    results = []

    for ticker in prices_df["ticker"].unique():
        subset = prices_df[prices_df["ticker"] == ticker].sort_values("date")

        if len(subset) < PRICE_TREND_DAYS:
            print(f"   ⚠️  Not enough data for {ticker} trend")
            continue

        recent       = subset.tail(PRICE_TREND_DAYS)
        price_start  = recent.iloc[0]["close"]
        price_end    = recent.iloc[-1]["close"]
        trend_pct    = ((price_end - price_start) / price_start) * 100

        if trend_pct > 1.0:
            trend_label = "uptrend"
        elif trend_pct < -1.0:
            trend_label = "downtrend"
        else:
            trend_label = "sideways"

        results.append({
            "ticker"      : ticker,
            "price_start" : round(price_start, 2),
            "price_end"   : round(price_end, 2),
            "trend_pct"   : round(trend_pct, 2),
            "trend_label" : trend_label,
        })

    return pd.DataFrame(results)


# ------------- Signal Logic -----------------------

def generate_signal(sentiment_label: str, trend_label: str) -> dict:
    """
    Combine sentiment + price trend into a final signal.

    Matrix:
    ┌─────────────┬──────────┬───────────┬──────────┐
    │             │ uptrend  │ sideways  │downtrend │
    ├─────────────┼──────────┼───────────┼──────────┤
    │ bullish     │ 🟢 BUY   │ 🟢 BUY   │ 🟡 WATCH │
    │ neutral     │ 🟡 WATCH │ 🟡 WATCH  │ 🟡 WATCH │
    │ bearish     │ 🟡 WATCH │ 🔴 SELL  │ 🔴 SELL  │
    └─────────────┴──────────┴───────────┴──────────┘
    """
    matrix = {
        ("bullish",  "uptrend")  : ("BUY",   "🟢", "Strong positive sentiment + rising price"),
        ("bullish",  "sideways") : ("BUY",   "🟢", "Positive sentiment, price consolidating"),
        ("bullish",  "downtrend"): ("WATCH", "🟡", "Positive news but price falling — divergence"),
        ("neutral",  "uptrend")  : ("WATCH", "🟡", "Price rising but sentiment unclear"),
        ("neutral",  "sideways") : ("WATCH", "🟡", "No strong signal in either direction"),
        ("neutral",  "downtrend"): ("WATCH", "🟡", "Price falling, sentiment unclear"),
        ("bearish",  "uptrend")  : ("WATCH", "🟡", "Price rising despite negative sentiment — divergence"),
        ("bearish",  "sideways") : ("SELL",  "🔴", "Negative sentiment, price stalling"),
        ("bearish",  "downtrend"): ("SELL",  "🔴", "Negative sentiment + falling price"),
    }

    key    = (sentiment_label, trend_label)
    result = matrix.get(key, ("WATCH", "🟡", "Unknown combination"))

    return {
        "signal"     : result[0],
        "emoji"      : result[1],
        "reasoning"  : result[2],
    }


# ------------- Main -----------------------

def run():
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("=" * 50)
    print("  🚦 Signal Generator")
    print("=" * 50 + "\n")

    # Load data
    sentiment_df = load_sentiment()
    prices_df    = load_latest_file("prices_*.csv", RAW_PATH)
    print()

    # Aggregate sentiment per ticker
    sentiment_agg = aggregate_sentiment(sentiment_df)

    # Compute price trend per ticker
    price_trend = compute_price_trend(prices_df)

    # Merge both
    combined = pd.merge(sentiment_agg, price_trend, on="ticker")

    # Generate signals
    signals = combined.apply(
        lambda row: generate_signal(row["sentiment_label"], row["trend_label"]),
        axis=1,
        result_type="expand"
    )

    final = pd.concat([combined, signals], axis=1)

    # Save
    output_path = os.path.join(PROCESSED_PATH, "signals.csv")
    final.to_csv(output_path, index=False)
    print(f"\n💾 Signals saved to {output_path}")

    # Print results
    print("\n" + "─" * 50)
    print("📊 Final Signals")
    print("─" * 50)

    for _, row in final.iterrows():
        print(f"""
  {row['ticker']}
    Sentiment : {row['sentiment_label'].upper()} (avg: {row['sentiment_avg']:.2f})
    Price     : {row['trend_label'].upper()} ({row['trend_pct']:+.2f}% over {PRICE_TREND_DAYS} days)
    Price Now : ${row['price_end']}
    Signal    : {row['emoji']}  {row['signal']}
    Reason    : {row['reasoning']}
        """)

    print("✅ Done!")
    return final


if __name__ == "__main__":
    run()