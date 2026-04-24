from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
import glob

# ------------- Configuration -----------------------

RAW_PATH       = "data/raw"
PROCESSED_PATH = "data/processed"

# FinBERT labels
LABELS = ["positive", "negative", "neutral"]

# ------------- Load FinBERT -----------------------

def load_finbert():
    print("🤖 Loading FinBERT model (first time may take ~1 min)...")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    print("   ✅ Model ready\n")
    return tokenizer, model


# ------------- Sentiment Analysis -----------------------

def analyse_text(text: str, tokenizer, model) -> dict:
    """
    Run FinBERT on a single piece of text.
    Returns label (positive/negative/neutral) and confidence score.
    """
    if not text or str(text).strip() == "":
        return {"label": "neutral", "score": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence = probs.max().item()
    label_idx  = probs.argmax().item()
    label      = LABELS[label_idx]

    return {"label": label, "score": round(confidence, 4)}


def analyse_dataframe(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """
    Run sentiment analysis on every row in the news DataFrame.
    Uses title + summary as input text.
    """
    results = []

    for idx, row in df.iterrows():
        # Combine title and summary for richer signal
        text = f"{row.get('title', '')}. {row.get('summary', '')}"

        result = analyse_text(text, tokenizer, model)

        ticker = row.get("ticker", "")
        print(f"   [{ticker}] {str(row.get('title', ''))[:60]}...")
        print(f"           → {result['label'].upper()} ({result['score']:.2f})\n")

        results.append(result)

    df["sentiment"]       = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    # Map to numeric for later signal generation
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_numeric"] = df["sentiment"].map(sentiment_map)

    return df


# ---------------- Load latest news CSV -----------------------

def load_latest_news() -> pd.DataFrame:
    """
    Automatically picks the most recent news CSV from data/raw/
    """
    files = glob.glob(os.path.join(RAW_PATH, "news_*.csv"))

    if not files:
        raise FileNotFoundError(f"No news CSV found in {RAW_PATH}. Run yahoo_finance.py first.")

    latest = max(files, key=os.path.getmtime)
    print(f"📂 Loading: {latest}\n")
    return pd.read_csv(latest)


# ------------- Summary -----------------------

def summarize(df: pd.DataFrame):
    print("\n" + "─" * 45)
    print("📊 Sentiment Summary by Ticker")
    print("─" * 45)

    for ticker in df["ticker"].unique():
        subset = df[df["ticker"] == ticker]
        counts = subset["sentiment"].value_counts()

        positive = counts.get("positive", 0)
        negative = counts.get("negative", 0)
        neutral  = counts.get("neutral",  0)
        avg      = subset["sentiment_numeric"].mean()

        if avg > 0.2:
            signal = "🟢 BULLISH"
        elif avg < -0.2:
            signal = "🔴 BEARISH"
        else:
            signal = "🟡 NEUTRAL"

        print(f"\n  {ticker}")
        print(f"    Positive : {positive} | Negative : {negative} | Neutral : {neutral}")
        print(f"    Signal   : {signal} (avg score: {avg:.2f})")

    print("\n✅ Sentiment analysis complete!")


#  ------- Entry point -------

if __name__ == "__main__":
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("=" * 45)
    print("  🧠 FinBERT Sentiment Analyser")
    print("=" * 45 + "\n")

    # Load data
    df = load_latest_news()
    print(f"   {len(df)} articles loaded\n")

    # Load model
    tokenizer, model = load_finbert()

    # Run analysis
    print("🔍 Analysing sentiment...\n")
    df = analyse_dataframe(df, tokenizer, model)

    # Save results
    output_path = os.path.join(PROCESSED_PATH, "news_with_sentiment.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to {output_path}")

    # Print summary
    summarize(df)