
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import yfinance as yf

from sentiment import analyze_sentiment

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "nvda_sentiment_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "nvda_features.pkl"


def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(FEATURES_PATH)
    )


def fetch_prices():
    prices = yf.Ticker("NVDA").history(
        period="3mo",
        interval="1d",
        auto_adjust=True
    )

    if prices.empty:
        raise ValueError("Unable to fetch NVDA price history.")

    prices = prices.reset_index()
    prices.columns = [
        str(column).lower().replace(" ", "_")
        for column in prices.columns
    ]

    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices = prices.sort_values("date").reset_index(drop=True)

    now_et = datetime.now(ZoneInfo("America/New_York"))

    if now_et.hour < 16:
        prices = prices[
            prices["date"].dt.date < now_et.date()
        ].reset_index(drop=True)

    if len(prices) < 10:
        raise ValueError("Insufficient completed trading-day history.")

    return prices


def build_features(prices, sentiment):
    df = prices.copy()

    df["daily_return"] = df["close"].pct_change()
    df["return_lag_1"] = df["daily_return"].shift(1)
    df["return_lag_2"] = df["daily_return"].shift(2)
    df["return_lag_3"] = df["daily_return"].shift(3)
    df["return_lag_5"] = df["daily_return"].shift(5)
    df["price_change_5d"] = df["close"].pct_change(5)
    df["volatility_5d"] = df["daily_return"].rolling(5).std()

    df["avg_sentiment"] = float(sentiment["avg_sentiment"])
    df["post_count"] = int(sentiment["post_count"])
    df["has_posts"] = int(sentiment["post_count"] > 0)

    return df.dropna().reset_index(drop=True)


def predict_next_day():
    model, features = load_artifacts()
    prices = fetch_prices()
    sentiment = analyze_sentiment()

    feature_data = build_features(prices, sentiment)
    latest = feature_data.iloc[[-1]]

    predicted_return = float(model.predict(latest[features])[0])
    last_close = float(latest["close"].iloc[0])

    return {
        "ticker": "NVDA",
        "as_of": latest["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_close": round(last_close, 2),
        "predicted_return_pct": round(predicted_return * 100, 4),
        "predicted_close": round(
            last_close * (1 + predicted_return), 2
        ),
        "post_count": sentiment["post_count"],
        "average_sentiment": round(
            sentiment["avg_sentiment"], 4
        ),
        "sentiment_used": True
    }


if __name__ == "__main__":
    result = predict_next_day()

    for key, value in result.items():
        print(f"{key}: {value}")
