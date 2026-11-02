
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_accidents(df, horizon=8):
    """
    Forecast weekly accident record counts.

    Returns a DataFrame with:
    Week, Accidents, Type
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    data = df.dropna(subset=["Date"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    if data.empty:
        raise ValueError("No valid accident dates were found.")

    # Count records in each week ending Sunday.
    weekly = (
        data.set_index("Date")
        .resample("W-SUN")
        .size()
        .asfreq("W-SUN", fill_value=0)
    )

    if len(weekly) < 6:
        raise ValueError(
            "At least 6 weekly observations are required."
        )

    model = ExponentialSmoothing(
        weekly.astype(float),
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    predicted = model.forecast(horizon).clip(lower=0)

    historical_df = pd.DataFrame({
        "Week": weekly.index,
        "Accidents": weekly.values,
        "Type": "Historical"
    })

    forecast_df = pd.DataFrame({
        "Week": predicted.index,
        "Accidents": predicted.values,
        "Type": "Forecast"
    })

    return pd.concat(
        [historical_df, forecast_df],
        ignore_index=True
    )
