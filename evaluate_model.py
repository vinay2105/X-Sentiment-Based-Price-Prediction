
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "nvda_training_data.csv"

PRICE_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "price_change_5d",
    "volatility_5d",
]

SENTIMENT_FEATURES = PRICE_FEATURES + [
    "avg_sentiment",
    "post_count",
    "has_posts",
]

def evaluate_model(data, features, model_name):
    predictions = []
    actuals = []

    start = int(len(data) * 0.6)

    for index in range(start, len(data)):
        train = data.iloc[:index]
        test = data.iloc[[index]]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=4,
            random_state=42,
        )

        model.fit(train[features], train["target_next_return"])
        prediction = float(model.predict(test[features])[0])

        predictions.append(prediction)
        actuals.append(float(test["target_next_return"].iloc[0]))

    return {
        "Model": model_name,
        "MAE": mean_absolute_error(actuals, predictions),
        "RMSE": mean_squared_error(actuals, predictions) ** 0.5,
        "Directional accuracy": sum(
            (actual > 0) == (predicted > 0)
            for actual, predicted in zip(actuals, predictions)
        ) / len(actuals),
    }

def main():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").dropna(
        subset=SENTIMENT_FEATURES + ["target_next_return"]
    ).reset_index(drop=True)

    start = int(len(data) * 0.6)
    actuals = data.iloc[start:]["target_next_return"].tolist()

    baseline = {
        "Model": "Zero-return baseline",
        "MAE": mean_absolute_error(actuals, [0.0] * len(actuals)),
        "RMSE": mean_squared_error(actuals, [0.0] * len(actuals)) ** 0.5,
        "Directional accuracy": sum(value <= 0 for value in actuals) / len(actuals),
    }

    results = [
        baseline,
        evaluate_model(data, PRICE_FEATURES, "Price-only"),
        evaluate_model(data, SENTIMENT_FEATURES, "Price + sentiment"),
    ]

    report = pd.DataFrame(results)
    report["Directional accuracy"] = report["Directional accuracy"].map(
        lambda value: f"{value:.2%}"
    )

    print(report.to_string(index=False))

    output_path = BASE_DIR / "evaluation_results.csv"
    report.to_csv(output_path, index=False)
    print(f"\nSaved evaluation report to: {output_path}")

if __name__ == "__main__":
    main()
