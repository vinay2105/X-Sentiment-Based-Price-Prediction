
import os
from datetime import datetime, timedelta, timezone

from apify_client import ApifyClient
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
ACTOR_ID = "smart_albatross/x-tweet-scraper"

sentiment_model = None


def load_sentiment_model():
    global sentiment_model

    if sentiment_model is None:
        sentiment_model = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1
        )

    return sentiment_model


def fetch_recent_posts(max_items=100):
    if not APIFY_TOKEN:
        raise ValueError("APIFY_API_TOKEN is missing from .env")

    client = ApifyClient(APIFY_TOKEN)

    since = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    until = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    run = client.actor(ACTOR_ID).call(run_input={
        "maxItems": max_items,
        "searchTerms": [
            f"NVDA since:{since} until:{until} lang:en",
            f"$NVDA since:{since} until:{until} lang:en",
            f"NVIDIA stock since:{since} until:{until} lang:en"
        ]
    })

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("Apify did not return a dataset ID.")

    posts = list(client.dataset(dataset_id).iterate_items())

    return [
        {
            "text": post.get("text", "").strip(),
            "created_at": post.get("createdAt"),
            "url": post.get("url") or post.get("tweetUrl") or post.get("twitterUrl")
        }
        for post in posts
        if post.get("text", "").strip()
    ]


def analyze_sentiment(max_items=100):
    posts = fetch_recent_posts(max_items)

    if not posts:
        return {
            "post_count": 0,
            "avg_sentiment": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "has_posts": 0,
            "posts": []
        }

    classifier = load_sentiment_model()
    predictions = []

    texts = [post["text"][:2000] for post in posts]

    for start in range(0, len(texts), 16):
        predictions.extend(
            classifier(texts[start:start + 16], truncation=True)
        )

    scores = {"positive": 1, "neutral": 0, "negative": -1}

    for post, result in zip(posts, predictions):
        post["sentiment"] = result["label"].lower()
        post["confidence"] = float(result["score"])

    return {
        "post_count": len(posts),
        "avg_sentiment": sum(
            scores[post["sentiment"]] for post in posts
        ) / len(posts),
        "positive_count": sum(
            post["sentiment"] == "positive" for post in posts
        ),
        "negative_count": sum(
            post["sentiment"] == "negative" for post in posts
        ),
        "has_posts": 1,
        "posts": posts
    }


if __name__ == "__main__":
    result = analyze_sentiment()

    print("Posts analyzed:", result["post_count"])
    print("Average sentiment:", round(result["avg_sentiment"], 4))
    print("Positive:", result["positive_count"])
    print("Negative:", result["negative_count"])

    for post in result["posts"][:3]:
        print("\n", post["sentiment"], "-", post["text"][:180])
