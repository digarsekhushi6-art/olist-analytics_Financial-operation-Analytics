"""
forecast.py  —  Revenue forecasting using Facebook Prophet.
Run:  python src/models/forecast.py
Output: data/processed/forecast.csv
"""
import pandas as pd
import numpy as np
import sqlite3, json
from pathlib import Path

DB        = str(Path(__file__).parent.parent.parent / "data" / "olist.db")
OUT       = Path(__file__).parent.parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def load_monthly():
    con = sqlite3.connect(DB)
    df  = pd.read_sql("SELECT order_month, gmv FROM fct_monthly_revenue ORDER BY order_month", con)
    con.close()
    df["ds"] = pd.to_datetime(df["order_month"])
    df["y"]  = df["gmv"]
    return df[["ds","y"]]


def run_prophet_forecast(df, periods=6):
    """Full Prophet pipeline with evaluation."""
    try:
        from prophet import Prophet
        from sklearn.metrics import mean_absolute_percentage_error

        n_test = 3
        train  = df.iloc[:-n_test]
        test   = df.iloc[-n_test:]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.15,
        )
        m.add_country_holidays(country_name="BR")
        m.fit(train)

        future   = m.make_future_dataframe(periods=periods, freq="MS")
        forecast = m.predict(future)

        # Evaluate on test set
        test_fc  = forecast[forecast["ds"].isin(test["ds"])][["ds","yhat"]]
        test_merged = test.merge(test_fc, on="ds")
        mape = mean_absolute_percentage_error(test_merged["y"], test_merged["yhat"])

        result = forecast[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        result["actual"] = df.set_index("ds").reindex(result["ds"])["y"].values
        result["model"]  = "Prophet"
        result["mape"]   = round(mape * 100, 2)
        return result

    except ImportError:
        return _fallback_forecast(df, periods)


def _fallback_forecast(df, periods=6):
    """Simple trend + seasonality fallback when Prophet not installed."""
    from sklearn.linear_model import LinearRegression

    df = df.copy()
    df["t"]     = np.arange(len(df))
    df["month"] = df["ds"].dt.month

    dummies = pd.get_dummies(df["month"], prefix="m")
    X = pd.concat([df[["t"]], dummies], axis=1).astype(float)
    y = df["y"].values

    model = LinearRegression().fit(X, y)

    future_dates = pd.date_range(df["ds"].max(), periods=periods+1, freq="MS")[1:]
    future_t     = np.arange(len(df), len(df) + periods)
    future_month = future_dates.month
    fd = pd.DataFrame({"t": future_t, "month": future_month})
    fd_dummies   = pd.get_dummies(fd["month"], prefix="m").reindex(columns=dummies.columns, fill_value=0)
    Xf = pd.concat([fd[["t"]], fd_dummies], axis=1).astype(float)

    preds = model.predict(Xf)

    hist = pd.DataFrame({
        "ds": df["ds"], "yhat": model.predict(X),
        "yhat_lower": model.predict(X)*0.85,
        "yhat_upper": model.predict(X)*1.15,
        "actual": y, "model":"LinearTrend", "mape": None
    })
    fut = pd.DataFrame({
        "ds": future_dates, "yhat": preds,
        "yhat_lower": preds*0.85, "yhat_upper": preds*1.15,
        "actual": None, "model":"LinearTrend", "mape": None
    })
    return pd.concat([hist, fut], ignore_index=True)


def run():
    print("Loading monthly GMV...")
    df = load_monthly()
    print(f"  {len(df)} monthly data points  ({df['ds'].min().date()} → {df['ds'].max().date()})")

    print("Running forecast model...")
    fc = run_prophet_forecast(df, periods=6)

    # Add is_forecast flag
    last_actual = df["ds"].max()
    fc["is_forecast"] = (fc["ds"] > last_actual).astype(int)

    out_path = OUT / "forecast.csv"
    fc.to_csv(out_path, index=False)
    print(f"  ✓ Forecast saved → {out_path}  ({len(fc)} rows)")

    mape_vals = fc["mape"].dropna()
    if len(mape_vals):
        print(f"  MAPE: {mape_vals.iloc[0]:.1f}%")

    return fc


if __name__ == "__main__":
    run()
