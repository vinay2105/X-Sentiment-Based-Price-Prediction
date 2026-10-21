# X Sentiment-Based Price Prediction

A quantitative research project exploring the relationship between X (Twitter) sentiment and NVIDIA (NVDA) stock-price movements using financial NLP, machine learning, and walk-forward evaluation.

The project combines social-media sentiment analysis, historical stock-price features, next-day return prediction, and an AI-powered research assistant in an interactive dashboard.

## Project Objectives

- Collect X posts related to a financial asset.
- Analyze financial sentiment using FinBERT.
- Combine sentiment and historical price features for return prediction.
- Compare price-only and price-plus-sentiment models.
- Evaluate predictions using chronological walk-forward validation.
- Explore financial news and market information through an AI research agent.

## Features

### 1. Stock Price Analysis
- Fetches historical NVDA prices using Yahoo Finance through `yfinance`.
- Calculates lagged returns, five-day price changes, and rolling volatility.
- Loads a trained Random Forest model.
- Estimates next-trading-day returns and an implied closing price.

### 2. X Sentiment Analysis
- Collects recent X posts using Apify.
- Analyzes financial sentiment using FinBERT.
- Calculates aggregate sentiment and post counts.
- Supplies sentiment features to the prediction pipeline.

### 3. AI Research Agent
- Uses LangGraph to orchestrate a tool-using research agent.
- Integrates Google Gemini for language-model reasoning.
- Provides tools for retrieving stock prices, predictions, sentiment, and NVIDIA news.
- Uses Tavily for news search.

### 4. Interactive Dashboard
- Built with Streamlit and Plotly.
- Presents stock predictions and research-agent interactions.
- Includes a model evaluation page comparing baseline and trained models.

### 5. Model Evaluation
- Uses recorded walk-forward results.
- Compares a zero-return baseline, a price-only model, and a price-plus-sentiment model.
- Reports mean absolute error, root mean squared error, and directional accuracy.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Data visualization | Plotly |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, Random Forest |
| Financial NLP | FinBERT, Hugging Face Transformers |
| Market data | yfinance |
| X data collection | Apify |
| Agent orchestration | LangGraph, LangChain |
| Language model | Google Gemini |
| News retrieval | Tavily |
| Model persistence | joblib |

## Architecture

```text
Yahoo Finance ─────────┐
                      │
Apify X Post Scraper ─┼──> Data Processing
                      │          │
FinBERT Sentiment ────┘          ▼
                           Feature Engineering
                                  │
                                  ▼
                         Random Forest Model
                                  │
                                  ▼
                         Next-Day Prediction

Streamlit Dashboard ──> LangGraph Research Agent
                                  │
                    ┌─────────────┼────────────┐
                    ▼             ▼            ▼
                 Price         Sentiment      News
                 Tools           Tools        Search

Evaluation Results ──> Evaluation Dashboard
```

## Repository Structure

```text
x-sentiment-price-prediction/
├── app.py
├── predictor.py
├── sentiment.py
├── agent.py
├── evaluation_results.csv
├── models/
│   ├── nvda_sentiment_model.pkl
│   └── nvda_features.pkl
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── EVALUATION_REPORT.md
```

This is the expected main structure. Retain any additional files required by your implementation.

## Installation and Setup

### Prerequisites

- Python version compatible with the project's dependencies.
- Google Gemini API key.
- Apify API token.
- Tavily API key.
- Internet access for external data services.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd x-sentiment-price-prediction
```

Replace the placeholder URL with your actual repository URL.

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as a template.

Set the required API credentials locally. Do not commit real credentials.

### 5. Verify model artifacts

Ensure the following files exist:

```text
models/nvda_sentiment_model.pkl
models/nvda_features.pkl
```

The current prediction pipeline loads these saved model artifacts.

### 6. Run the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit.

## Existing Evaluation Results

The recorded walk-forward evaluation contains 49 observations from August 12 through October 20, 2026.

| Model | MAE | RMSE | Directional Accuracy |
|---|---:|---:|---:|
| Zero-return baseline | 0.015837 | 0.021405 | 53.06% |
| Price-only | 0.015971 | 0.020660 | 57.14% |
| Price + sentiment | 0.015906 | 0.020524 | 53.06% |

### Interpretation

- The price-only model achieved the highest directional accuracy in the recorded evaluation.
- Both trained models had higher MAE than the zero-return baseline.
- The price-plus-sentiment model achieved a slightly lower RMSE than the price-only model, but did not improve directional accuracy.
- The test sample is small and does not establish a reliable trading edge.

These metrics represent the existing evaluation run, not a completed six-horizon evaluation.

## Scope and Limitations

The current implementation includes recent X sentiment collection, price-derived features, next-day prediction, an AI research agent, and an existing walk-forward evaluation.

The following requirements from the full quantitative research assessment remain outstanding or require further validation:

- Complete 180-day historical X ingestion, including all requested engagement fields.
- Forecasting all six horizons from n+0 through n+5.
- A dedicated one-page sentiment outlook report.
- A complete sentiment-versus-price visualization matrix.
- A defined backtesting strategy with Sharpe ratio, Sortino ratio, and maximum drawdown.
- Verified point-in-time alignment and reproducibility of the complete historical dataset.

The current prototype should not be interpreted as satisfying these incomplete requirements.

## Reproducibility and Data Quality

- External APIs may impose rate limits or return incomplete data.
- X posts may contain duplicates, promotional content, and irrelevant material.
- Forecast features must use only information available at the forecast cutoff.
- Model artifacts may depend on the scikit-learn version used during training.
- API credentials must be configured locally and kept private.
- Evaluation results should be regenerated when the model, data, or feature pipeline changes.

## Future Work

1. Complete historical data ingestion and timestamp alignment.
2. Add engagement-based sentiment features.
3. Implement all six forecast horizons.
4. Expand walk-forward evaluation.
5. Define a trading strategy and calculate risk-adjusted metrics.
6. Add the sentiment-price matrix and sentiment outlook report.

## Disclaimer

This project is intended for educational and quantitative research purposes only. It does not provide financial advice. Predictions and historical evaluation results do not guarantee future performance or profitability.
