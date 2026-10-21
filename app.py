import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import plotly.express as px

from predictor import predict_next_day, fetch_prices
from agent import ask_agent


def render_model_evaluation():
    results_path = Path(__file__).resolve().parent / "evaluation_results.csv"

    st.subheader("Model Evaluation")
    st.caption("Chronological walk-forward validation")

    if not results_path.exists():
        st.warning("evaluation_results.csv was not found.")
        return

    results = pd.read_csv(results_path)

    metric_columns = ["MAE", "RMSE"]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.bar(
                results,
                x="Model",
                y="MAE",
                title="Mean Absolute Error",
                color="Model",
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            px.bar(
                results,
                x="Model",
                y="RMSE",
                title="Root Mean Squared Error",
                color="Model",
            ),
            use_container_width=True,
        )

    accuracy = results.copy()
    accuracy["Directional accuracy"] = (
        accuracy["Directional accuracy"]
        .astype(str)
        .str.rstrip("%")
        .astype(float)
    )

    st.plotly_chart(
        px.bar(
            accuracy,
            x="Model",
            y="Directional accuracy",
            title="Directional Accuracy (%)",
            color="Model",
        ),
        use_container_width=True,
    )

    st.dataframe(results, use_container_width=True, hide_index=True)

    st.info(
        "The price-only model has the highest directional accuracy in this "
        "evaluation. The sample is small, and these results do not establish "
        "a reliable trading edge."
    )


st.set_page_config(
    page_title="X Sentiment-Based Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --app-bg: #080d18;
    --panel-bg: #111a2a;
    --input-bg: #0b1321;
    --border-color: #2b3a50;
    --text-primary: #edf3ff;
    --text-secondary: #a6b5ca;
    --accent: #57d6b2;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--app-bg);
    color: var(--text-primary);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

p, li, label, span, small {
    color: inherit;
}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: var(--text-primary);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: var(--text-secondary) !important;
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 24, 0.96);
}

[data-testid="stToolbar"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0d1422;
    border-right: 1px solid #202d42;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #141f31, #101827);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px 22px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: var(--text-secondary) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

div.stButton > button,
[data-testid="stFormSubmitButton"] button {
    min-height: 42px;
    border-radius: 10px;
    border: 1px solid #354761;
    background: #17243a;
    color: #f1f5ff !important;
    font-weight: 600;
}

div.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    border-color: var(--accent);
    color: var(--accent) !important;
}

div.stButton > button[kind="primary"] {
    background: var(--accent);
    color: #07131a !important;
    border: none;
}

div.stButton > button[kind="primary"]:hover {
    background: #7ce8ca;
    color: #07131a !important;
}

[data-testid="stChatMessage"] {
    background: #111b2c !important;
    border: 1px solid #2b3a50 !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
    color: #edf3ff !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
    color: #72dff0 !important;
    text-decoration: underline;
}

[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] code {
    background: #080f1c !important;
    color: #b8f3dc !important;
    border: 1px solid #28384e;
    border-radius: 6px;
}

[data-testid="stChatMessage"] blockquote {
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    color: var(--text-secondary) !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"] {
    background: var(--input-bg) !important;
    border-color: #354761 !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: var(--input-bg) !important;
    color: #f1f5ff !important;
    -webkit-text-fill-color: #f1f5ff !important;
    caret-color: var(--accent) !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9aabc2 !important;
    -webkit-text-fill-color: #9aabc2 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: #1c3044 !important;
    color: var(--accent) !important;
    border: 1px solid #354761 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div {
    background: var(--input-bg) !important;
    border-color: #354761 !important;
    color: var(--text-primary) !important;
}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background: var(--input-bg) !important;
    color: #f1f5ff !important;
    -webkit-text-fill-color: #f1f5ff !important;
    caret-color: var(--accent) !important;
}

[data-baseweb="input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder {
    color: #9aabc2 !important;
    -webkit-text-fill-color: #9aabc2 !important;
    opacity: 1 !important;
}

[data-baseweb="select"] *,
[data-testid="stSelectbox"] *,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: var(--text-primary) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background: #111b2c !important;
    color: var(--text-primary) !important;
}

[role="option"]:hover {
    background: #203149 !important;
}

[data-testid="stAlert"] {
    background: #111d2d !important;
    border: 1px solid #354761 !important;
    border-radius: 12px;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] {
    background: #101827;
    border: 1px solid var(--border-color);
    border-radius: 12px;
}

[data-testid="stProgressBar"] > div {
    background: #26364b;
}

hr {
    border-color: #26334a !important;
}

.eyebrow {
    color: var(--accent) !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero {
    background: linear-gradient(120deg, #14253a, #101827 65%, #102b30);
    border: 1px solid #2b4055;
    border-radius: 22px;
    padding: 28px 30px;
    margin-bottom: 25px;
}

.hero-subtitle {
    color: #b4c2d5 !important;
    font-size: 15px;
}

.panel {
    background: #101827;
    border: 1px solid var(--border-color);
    border-radius: 18px;
    padding: 20px;
}

.muted {
    color: var(--text-secondary) !important;
    font-size: 13px;
}

.status-dot {
    color: var(--accent) !important;
    font-weight: 700;
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 20px;
        border-radius: 16px;
    }

    [data-testid="stMetric"] {
        padding: 14px;
    }

    [data-testid="stChatMessage"] {
        padding: 10px !important;
    }
}
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="textarea"] > div {
    background-color: #080d18 !important;
    background: #080d18 !important;
}

[data-testid="stBottom"] {
    border-top: 1px solid #26334a !important;
}

[data-testid="stChatInput"] {
    border: 1px solid #354761 !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #edf3ff !important;
    -webkit-text-fill-color: #edf3ff !important;
    caret-color: #57d6b2 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9aabc2 !important;
    -webkit-text-fill-color: #9aabc2 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<p class="eyebrow">MARKET WORKSPACE</p>', unsafe_allow_html=True)
    st.title("NVDA Intelligence")
    st.caption("AI-powered market research")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "AI Research Agent",
            "Model Evaluation",
            "Model Information",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**Data sources**")
    st.markdown("- Yahoo Finance")
    st.markdown("- X posts via Apify")
    st.markdown("- FinBERT sentiment")
    st.markdown("- Gemini research agent")

    st.divider()

    st.markdown(
        '<p class="muted">Experimental research tool. '
        'Not financial advice.</p>',
        unsafe_allow_html=True
    )

st.markdown(
    '<p class="eyebrow">MARKET INTELLIGENCE / UNITED STATES</p>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:8px;">NVIDIA Market Intelligence</h1>
    <p class="hero-subtitle">
        Track market prices, estimate the next trading-day move,
        understand social sentiment, and investigate market news
        with an AI research agent.
    </p>
</div>
""", unsafe_allow_html=True)

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if page == "Overview":
    st.markdown("### Market overview")

    try:
        prices = fetch_prices()
        latest = prices.iloc[-1]
        latest_close = float(latest["close"])
        price_date = latest["date"].strftime("%b %d, %Y")

        previous_close = (
            float(prices.iloc[-2]["close"])
            if len(prices) > 1
            else latest_close
        )

        daily_change = (
            (latest_close / previous_close) - 1
        ) * 100

        prediction = st.session_state["prediction"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Latest closing price",
            f"${latest_close:,.2f}",
            f"{daily_change:+.2f}%"
        )

        if prediction:
            col2.metric(
                "Predicted next close",
                f"${prediction['predicted_close']:,.2f}",
                f"{prediction['predicted_return_pct']:+.4f}%"
            )

            col3.metric(
                "Posts analyzed",
                f"{prediction['post_count']:,}"
            )

            col4.metric(
                "Average sentiment",
                f"{prediction['average_sentiment']:+.3f}"
            )
        else:
            col2.metric("Predicted next close", "—")
            col3.metric("Posts analyzed", "—")
            col4.metric("Average sentiment", "—")

        st.caption(
            f"Latest available price date: {price_date} · "
            "Price data may be delayed."
        )

        st.markdown("### Price performance")

        chart_col, info_col = st.columns([2.2, 1])

        with chart_col:
            chart_prices = prices.tail(90)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=chart_prices["date"],
                y=chart_prices["close"],
                mode="lines",
                name="NVDA close",
                line=dict(color="#57d6b2", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(87, 214, 178, 0.07)",
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=390,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title=None,
                yaxis_title="Price (USD)",
                hovermode="x unified",
                showlegend=False,
                font=dict(color="#a8b7cc"),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    gridcolor="#243148",
                    zeroline=False
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        with info_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Forecast center")
            st.markdown(
                '<p class="muted">Run your trained model against '
                'the latest available price and social data.</p>',
                unsafe_allow_html=True
            )

            if st.button(
                "Generate new prediction",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Analyzing prices and sentiment..."):
                    try:
                        result = predict_next_day()
                        st.session_state["prediction"] = result
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

            prediction = st.session_state["prediction"]

            if prediction:
                st.divider()
                st.markdown("**Model output**")
                st.metric(
                    "Estimated next close",
                    f"${prediction['predicted_close']:.2f}"
                )
                st.write(
                    f"Predicted return: "
                    f"{prediction['predicted_return_pct']:+.4f}%"
                )
                st.write(
                    f"Sentiment score: "
                    f"{prediction['average_sentiment']:+.3f}"
                )
                st.caption(
                    f"Forecast based on price data through "
                    f"{prediction['as_of']}."
                )
            else:
                st.info("Generate a prediction to see the model output.")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Forecast interpretation")
        if prediction:
            predicted_return = prediction["predicted_return_pct"]

            if predicted_return > 0:
                st.success(
                    f"The model estimates a {predicted_return:.4f}% "
                    "increase for the next trading day."
                )
            elif predicted_return < 0:
                st.warning(
                    f"The model estimates a {predicted_return:.4f}% "
                    "move for the next trading day."
                )
            else:
                st.info("The model estimates almost no price movement.")

            st.caption(
                "This is a model estimate, not a probability of success. "
                "The model has not demonstrated that sentiment materially "
                "improves predictive accuracy."
            )
        else:
            st.info("Run a prediction to see the forecast interpretation.")

    except Exception as exc:
        st.error(f"Unable to load market data: {exc}")

elif page == "AI Research Agent":
    st.markdown("### AI research assistant")
    st.markdown(
        '<p class="muted">Ask about NVDA prices, experimental forecasts, '
        'social sentiment, or recent NVIDIA news.</p>',
        unsafe_allow_html=True
    )

    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about NVIDIA..."
    )

    if question:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Researching your question..."):
                try:
                    answer = ask_agent(question)
                    st.markdown(answer)

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as exc:
                    st.error(f"Agent error: {exc}")

elif page == "Model Evaluation":
    render_model_evaluation()

elif page == "Model Information":
    st.markdown("### About the forecasting system")

    st.markdown("""
    <div class="panel">
        <h4>Prediction pipeline</h4>
        <p class="muted">
            Historical prices → feature engineering → trained
            Random Forest model → estimated next-day return
        </p>
        <h4>Sentiment pipeline</h4>
        <p class="muted">
            X posts → text classification with FinBERT →
            aggregated sentiment features
        </p>
        <h4>Research agent</h4>
        <p class="muted">
            LangGraph coordinates tools for price lookup, forecasts,
            sentiment analysis, and news search.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Important limitations")
    st.warning(
        "This is an experimental project, not a validated trading system. "
        "Historical evaluation showed nearly identical error for the "
        "price-only and price-plus-sentiment models. The forecast can be "
        "wrong, sentiment data may contain noise, and prices can change "
        "rapidly. Do not use these predictions alone for financial decisions."
    )

st.divider()

st.markdown(
    '<p class="muted">NVDA Intelligence · Experimental ML research · '
    'Not investment advice</p>',
    unsafe_allow_html=True
)
