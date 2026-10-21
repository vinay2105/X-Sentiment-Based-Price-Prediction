# Quantitative Evaluation Report: NVDA Sentiment and Price Prediction

## 1. Objective

Evaluate whether financial sentiment derived from X posts provides additional information for predicting NVIDIA stock returns compared with historical price features and a naive baseline.

## 2. Existing Evaluation

- Asset: NVIDIA Corporation (NVDA)
- Method: Walk-forward evaluation
- Recorded observations: 49
- Recorded evaluation period: August 12–October 20, 2026
- Approaches: zero-return baseline, price-only model, price-plus-sentiment model

This is the existing next-day evaluation, not a completed evaluation of all six horizons requested in the assessment.

## 3. Results

| Model | MAE | RMSE | Directional Accuracy |
|---|---:|---:|---:|
| Zero-return baseline | 0.015837 | 0.021405 | 53.06% |
| Price-only | 0.015971 | 0.020660 | 57.14% |
| Price + sentiment | 0.015906 | 0.020524 | 53.06% |

MAE and RMSE are reported in decimal-return units, following the recorded evaluation output.

## 4. Findings

The price-only model achieved 57.14% directional accuracy, compared with 53.06% for the price-plus-sentiment model.

Adding sentiment slightly reduced RMSE relative to the price-only model. However, both trained models had higher MAE than the zero-return baseline.

The results therefore do not establish that sentiment improves predictive performance consistently. The limited sample size also makes the findings preliminary.

## 5. Risk-Adjusted Performance

The Sharpe ratio, Sortino ratio, and maximum drawdown have not been established by the current prediction evaluation.

Calculating these metrics requires a defined trading strategy, a chronological series of strategy returns, portfolio-value calculations, and documented assumptions. Numerical risk metrics should not be reported until that backtest has been completed.

## 6. Remaining Assessment Requirements

- Complete 180-day historical data ingestion.
- Forecasts for n+0, n+1, n+2, n+3, n+4, and n+5.
- Sentiment-versus-price visual matrix.
- One-page sentiment outlook.
- Strategy backtesting and risk metrics.
- Verified point-in-time alignment and reproducibility.

## 7. Conclusion

The current prototype provides a foundation for investigating the relationship between financial sentiment and stock returns. Its recorded evaluation does not establish a reliable or profitable trading edge.

The full multi-horizon forecasting and risk-adjusted strategy evaluation remain incomplete.

## Disclaimer

This report is for educational and research purposes only and does not constitute financial advice.
