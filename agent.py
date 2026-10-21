import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from predictor import predict_next_day, fetch_prices
from sentiment import analyze_sentiment

load_dotenv()


@tool
def get_current_nvda_price() -> dict:
    """Get the latest available NVDA stock price from Yahoo Finance."""
    prices = fetch_prices()
    latest = prices.iloc[-1]

    return {
        "ticker": "NVDA",
        "latest_close": round(float(latest["close"]), 2),
        "date": latest["date"].strftime("%Y-%m-%d")
    }


@tool
def get_nvda_prediction() -> dict:
    """Generate an experimental next-trading-day NVDA price prediction."""
    return predict_next_day()


@tool
def get_latest_sentiment() -> dict:
    """Fetch recent NVDA-related posts and calculate their sentiment."""
    result = analyze_sentiment()
    result.pop("posts", None)
    return result


@tool
def search_nvidia_news(query: str) -> str:
    """Search recent NVIDIA and stock-market news."""
    if not os.getenv("TAVILY_API_KEY"):
        return "TAVILY_API_KEY is missing from .env."

    search = TavilySearch(max_results=5)
    return str(search.invoke({"query": query}))


tools = [
    get_current_nvda_price,
    get_nvda_prediction,
    get_latest_sentiment,
    search_nvidia_news
]

agent = create_react_agent(
    model="google_genai:gemini-2.5-flash",
    tools=tools,
    prompt=(
        "You are an NVDA market research assistant. "
        "For current or latest stock price questions, call "
        "get_current_nvda_price. Clearly state the price date and "
        "that it is the latest available closing price, not necessarily "
        "the live intraday price. For prediction questions, call "
        "get_nvda_prediction. Use other tools for sentiment and news. "
        "Never invent market data or claim predictions are guaranteed."
    )
)


def ask_agent(question: str) -> str:
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    return str(result["messages"][-1].content)


if __name__ == "__main__":
    question = input("Ask about NVDA: ")
    print(ask_agent(question))
