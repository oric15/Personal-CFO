import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# -------------------- Configuration --------------------

TICKERS = ["NVDA", "AIR.PA", "IREN"]

PRICE_PERIOD   = "3mo"   # how far back to fetch prices (3 months)
PRICE_INTERVAL = "1d"    # daily candles

SAVE_PATH = "data/raw"


# -------------------- Helpers --------------------

def ensure_dirs():
    os.makedirs(SAVE_PATH, exist_ok=True)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# ---------------- News Collector ----------------------

def collect_news(ticker: str) -> pd.DataFrame:
    """
    Fetch latest news headlines for a ticker from Yahoo Finance.
    Returns a DataFrame with title, publisher, link, and publish time.
    """
    stock = yf.Ticker(ticker)
    news  = stock.news

    if not news:
        print(f"   ⚠️  No news found for {ticker}")
        return pd.DataFrame()

    rows = []
    for item in news:
        # content is nested inside each news item
        content = item.get("content", {})
        rows.append({
            "ticker"       : ticker,
            "title"        : content.get("title", ""),
            "publisher"    : content.get("provider", {}).get("displayName", ""),
            "link"         : content.get("canonicalUrl", {}).get("url", ""),
            "published_at" : content.get("pubDate", ""),
            "summary"      : content.get("summary", ""),
        })

    df = pd.DataFrame(rows)
    return df


def collect_all_news() -> pd.DataFrame:
    all_news = []

    for ticker in TICKERS:
        print(f"📰 Fetching news for {ticker}...")
        df = collect_news(ticker)
        if not df.empty:
            all_news.append(df)
            print(f"   ✅ {len(df)} articles found")

    if not all_news:
        print("❌ No news collected.")
        return pd.DataFrame()

    combined = pd.concat(all_news, ignore_index=True)

    # Save to CSV
    path = os.path.join(SAVE_PATH, f"news_{timestamp()}.csv")
    combined.to_csv(path, index=False)
    print(f"\n💾 News saved to {path}")

    return combined


# ---------------- Price Collector ----------------------

def collect_prices() -> pd.DataFrame:
    """
    Fetch historical OHLCV price data for all tickers.
    Returns a combined DataFrame with a 'ticker' column.
    """
    all_prices = []

    for ticker in TICKERS:
        print(f"📈 Fetching prices for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df    = stock.history(period=PRICE_PERIOD, interval=PRICE_INTERVAL)

            if df.empty:
                print(f"   ⚠️  No price data for {ticker}")
                continue

            df = df.reset_index()
            df["ticker"] = ticker

            # Keep only the columns we need
            df = df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["ticker", "date", "open", "high", "low", "close", "volume"]

            # Calculate daily return (%)
            df["daily_return_pct"] = df["close"].pct_change() * 100

            all_prices.append(df)
            print(f"   ✅ {len(df)} days of data")

        except Exception as e:
            print(f"   ❌ Error fetching {ticker}: {e}")

    if not all_prices:
        print("❌ No price data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_prices, ignore_index=True)

    # Save to CSV
    path = os.path.join(SAVE_PATH, f"prices_{timestamp()}.csv")
    combined.to_csv(path, index=False)
    print(f"\n💾 Prices saved to {path}")

    return combined


# -------------- Summary ----------------------

def summarize(news_df: pd.DataFrame, prices_df: pd.DataFrame):
    print("\n" + "─" * 40)
    print("📊 Collection Summary")
    print("─" * 40)

    if not news_df.empty:
        print("\n📰 News:")
        for ticker in TICKERS:
            count = len(news_df[news_df["ticker"] == ticker])
            print(f"   {ticker}: {count} articles")

    if not prices_df.empty:
        print("\n📈 Prices:")
        for ticker in TICKERS:
            subset = prices_df[prices_df["ticker"] == ticker]
            if not subset.empty:
                latest_close = subset.iloc[-1]["close"]
                date_range   = f"{subset['date'].min()} → {subset['date'].max()}"
                print(f"   {ticker}: ${latest_close:.2f} | {date_range}")

    print("\n✅ All done!")


# ------- Entry point -------- 

if __name__ == "__main__":
    ensure_dirs()

    print("=" * 40)
    print("  🚀 Yahoo Finance Collector")
    print("=" * 40 + "\n")

    news_df   = collect_all_news()
    print()
    prices_df = collect_prices()

    summarize(news_df, prices_df)