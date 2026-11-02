
from pathlib import Path

import pandas as pd


# Project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "road_traffic_accidents_zw.csv"
)

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mcnallypp/zwaccidents/refs/heads/main/data/road_traffic_accidents_zw.csv"


def load_data(source="local"):
    """Loading and validating the road traffic accident dataset."""

    if source == "github":
        if not GITHUB_DATA_URL:
            raise ValueError(
                "Set GITHUB_DATA_URL in src/data_loader.py first."
            )
        df = pd.read_csv(GITHUB_DATA_URL)
    else:
        df = pd.read_csv(LOCAL_DATA_PATH)

    required = [
        "ID", "Date", "Province", "Injured",
        "Fatalities", "Lat", "Long"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Convert fields to appropriate data types.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Lat", "Long", "Injured", "Fatalities"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove records without usable dates or coordinates.
    df = df.dropna(subset=["Date", "Lat", "Long"]).copy()

    # Keep plausible Zimbabwe coordinates.
    df = df[
        df["Lat"].between(-23.5, -15.5)
        & df["Long"].between(25.0, 34.5)
    ].copy()

    return df.reset_index(drop=True)
