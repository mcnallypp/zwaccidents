import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet


def prepare_weekly_data(df):
    """Convert accident records into weekly accident counts."""

    data = df.dropna(subset=["Date"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    weekly = (
        data.set_index("Date")
        .resample("W-SUN")
        .size()
        .asfreq("W-SUN", fill_value=0)
    )

    if len(weekly) < 6:
        raise ValueError("At least 6 weekly observations are required.")

    return weekly.astype(float)


def forecast_holt_winters(weekly, horizon):
    model = ExponentialSmoothing(
        weekly,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    predictions = model.forecast(horizon).clip(lower=0)

    return predictions, None, None

### the parameters for the SARIMAX model were obtained from the notebook trials
def forecast_sarima(weekly, horizon):
    model = SARIMAX(
        weekly,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        trend="t",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False)
    result = fitted.get_forecast(steps=horizon)

    predictions = result.predicted_mean.clip(lower=0)
    intervals = result.conf_int(alpha=0.20)

    lower = intervals.iloc[:, 0].clip(lower=0)
    upper = intervals.iloc[:, 1].clip(lower=0)

    return predictions, lower, upper


def forecast_prophet(weekly, horizon):

    prophet_data = pd.DataFrame({
        "ds": weekly.index,
        "y": weekly.values
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.05
    )

    model.fit(prophet_data)

    future = model.make_future_dataframe(
        periods=horizon,
        freq="W-SUN"
    )

    forecast = model.predict(future).tail(horizon)

    predictions = pd.Series(
        forecast["yhat"].clip(lower=0).values,
        index=pd.DatetimeIndex(forecast["ds"]),
        name="Accidents"
    )

    lower = pd.Series(
        forecast["yhat_lower"].clip(lower=0).values,
        index=predictions.index
    )

    upper = pd.Series(
        forecast["yhat_upper"].clip(lower=0).values,
        index=predictions.index
    )

    return predictions, lower, upper


def forecast_accidents(df, horizon=8, model="holt_winters"):
    """
    Forecast weekly recorded accident counts.

    Supported models:
    - holt_winters
    - sarima
    - prophet
    """

    if horizon < 1:
        raise ValueError("Horizon must be at least 1.")

    weekly = prepare_weekly_data(df)

    if model == "holt_winters":
        predictions, lower, upper = forecast_holt_winters(
            weekly, horizon
        )

    elif model == "sarima":
        predictions, lower, upper = forecast_sarima(
            weekly, horizon
        )

    elif model == "prophet":
        predictions, lower, upper = forecast_prophet(
            weekly, horizon
        )

    else:
        raise ValueError(f"Unsupported forecasting model: {model}")

    historical_df = pd.DataFrame({
        "Week": weekly.index,
        "Accidents": weekly.values,
        "Type": "Historical",
        "Lower": float("nan"),
        "Upper": float("nan")
    })

    forecast_df = pd.DataFrame({
        "Week": predictions.index,
        "Accidents": predictions.values,
        "Type": "Forecast",
        "Lower": lower.values if lower is not None else float("nan"),
        "Upper": upper.values if upper is not None else float("nan")
    })

    return pd.concat(
        [historical_df, forecast_df],
        ignore_index=True
    )
