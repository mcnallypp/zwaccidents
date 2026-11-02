
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data
from src.clustering import cluster_accidents
from src.forecasting import forecast_accidents


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zimbabwe Road Safety Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #FCE8E8;
    border: 1px solid #F5C6C6;
    padding: 18px 20px;
    border-radius: 12px;
    box-shadow: 0 2px 5px rgba(150, 50, 50, 0.06);
}

div[data-testid="stMetric"] label {
    color: #8B3030 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #B22222 !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data(ttl=300)
def get_data():
    return load_data(source="local")


st.title("🚦 Zimbabwe Road Traffic Accident Analytics")
st.caption(
    "Geographic hotspots, weekly forecasting and road safety indicators"
)

try:
    df = get_data()
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()


# Ensure required analytical fields exist
for col in ["Province", "Cause", "Location", "Road", "Time"]:
    if col not in df.columns:
        df[col] = "Unknown"

for col in ["Injured", "Fatalities", "Lat", "Long"]:
    if col not in df.columns:
        df[col] = float("nan")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Injured"] = pd.to_numeric(df["Injured"], errors="coerce")
df["Fatalities"] = pd.to_numeric(
    df["Fatalities"], errors="coerce"
)
df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
df["Long"] = pd.to_numeric(df["Long"], errors="coerce")

df = df.dropna(subset=["Date", "Lat", "Long"]).copy()

if df.empty:
    st.error("No valid dated accident records with coordinates were found.")
    st.stop()


# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

province_options = sorted(
    df["Province"].dropna().astype(str).unique().tolist()
)

selected_provinces = st.sidebar.multiselect(
    "Province",
    options=province_options,
    default=province_options
)

minimum_date = df["Date"].min().date()
maximum_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Accident date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date
)

if len(date_range) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("The start date must not be after the end date.")
    st.stop()

# Keep the selected province and date filters for the map and KPIs.
filtered = df[
    df["Province"].astype(str).isin(selected_provinces)
    & df["Date"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# --------------------------------------------------
# 4. KPI CARDS
# --------------------------------------------------

st.subheader("Key Indicators")

if "ID" in filtered.columns:
    accident_total = filtered["ID"].nunique()
else:
    accident_total = len(filtered)

# Missing casualty values remain unknown rather than being
# silently interpreted as zero.
fatality_total = filtered["Fatalities"].sum(min_count=1)
injury_total = filtered["Injured"].sum(min_count=1)

k1, k2, k3 = st.columns(3)

k1.metric("Accident reports", f"{accident_total:,}")

k2.metric(
    "Reported fatalities",
    "Unknown" if pd.isna(fatality_total)
    else f"{fatality_total:,.0f}"
)

k3.metric(
    "Reported injured",
    "Unknown" if pd.isna(injury_total)
    else f"{injury_total:,.0f}"
)

st.caption(
    f"Displaying {len(filtered):,} records "
    f"from {start_date} to {end_date}. "
    "Casualty totals exclude missing values."
)


# --------------------------------------------------
# 5. ACCIDENT HOTSPOT MAP
# --------------------------------------------------

st.divider()
st.subheader("Geographic Accident Hotspots")

map_col, settings_col = st.columns([4, 1])

with settings_col:
    radius_km = st.slider(
        "Cluster radius (km)",
        min_value=1,
        max_value=10,
        value=5
    )

    min_samples = st.slider(
        "Minimum cluster size",
        min_value=2,
        max_value=10,
        value=3
    )

    st.caption(
        "Smaller radii identify tighter groups. "
        "Adjust the settings to explore spatial patterns."
    )

map_data = filtered.dropna(subset=["Lat", "Long"]).copy()

if len(map_data) < min_samples:
    st.warning(
        "There are too few mapped records for the selected "
        "minimum cluster size."
    )
else:
    clustered = cluster_accidents(
        map_data,
        radius_km=radius_km,
        min_samples=min_samples
    )

    clustered["Cluster label"] = clustered["Cluster"].apply(
        lambda value:
        f"Cluster {value}" if value >= 0
        else "Unclustered"
    )

    cluster_count = clustered.loc[
        clustered["Cluster"] >= 0, "Cluster"
    ].nunique()

    noise_count = int((clustered["Cluster"] == -1).sum())

    with map_col:
        map_fig = px.scatter_map(
            clustered,
            lat="Lat",
            lon="Long",
            color="Cluster label",
            hover_name="Location",
            hover_data={
                "Date": True,
                "Road": True,
                "Province": True,
                "Fatalities": True,
                "Injured": True,
                "Lat": False,
                "Long": False
            },
            map_style="open-street-map",
            center={"lat": -19.0, "lon": 29.5},
            zoom=5,
            height=580,
            title="Reported Accident Locations"
        )

        map_fig.update_traces(
            marker={"size": 9, "opacity": 0.8}
        )

        map_fig.update_layout(
            margin={"l": 0, "r": 0, "t": 45, "b": 0}
        )

        st.plotly_chart(map_fig, use_container_width=True)

    st.write(
        f"**Clusters detected:** {cluster_count}  |  "
        f"**Unclustered records:** {noise_count}"
    )

    st.caption(
        "DBSCAN clusters represent concentrations of recorded "
        "locations, not proven road-risk rankings."
    )


# --------------------------------------------------
# 6. WEEKLY ACCIDENT FORECAST
# --------------------------------------------------

st.divider()
st.subheader("Weekly Accident Trend and Forecast")

forecast_horizon = st.selectbox(
    "Forecast horizon",
    options=[4, 6, 8],
    index=2,
    format_func=lambda value: f"Next {value} weeks"
)

st.caption(
    "The forecast uses all available history up to the selected "
    "end date for the selected provinces. The start date does "
    "not truncate model training."
)

forecast_input = df[
    df["Province"].astype(str).isin(selected_provinces)
    & (df["Date"].dt.date <= end_date)
].copy()

if forecast_input.empty:
    st.info("No historical data is available for this forecast.")
else:
    try:
        forecast_df = forecast_accidents(
            forecast_input,
            horizon=forecast_horizon
        )

        forecast_fig = px.line(
            forecast_df,
            x="Week",
            y="Accidents",
            color="Type",
            markers=True,
            title="Weekly Recorded Accidents",
            labels={
                "Week": "Week ending",
                "Accidents": "Accident reports",
                "Type": "Series"
            },
            color_discrete_map={
                "Historical": "#0D9488",
                "Forecast": "#D97706"
            }
        )

        forecast_fig.update_layout(
            hovermode="x unified",
            height=430,
            margin={"l": 10, "r": 10, "t": 55, "b": 10}
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        future_values = forecast_df.loc[
            forecast_df["Type"] == "Forecast", "Accidents"
        ]

        st.metric(
            "Average forecast per week",
            f"{future_values.mean():.1f}"
        )

        st.caption(
            "Forecasts are experimental estimates of recorded "
            "accident counts. Validate against held-out data "
            "before using them for operational decisions."
        )

    except Exception as exc:
        st.warning(
            f"Forecast unavailable. Check data coverage and "
            f"model requirements. Details: {exc}"
        )


# --------------------------------------------------
# 7. ACCIDENTS BY PROVINCE
# --------------------------------------------------

st.divider()
st.subheader("Accident Patterns")

left_col, right_col = st.columns(2)

with left_col:
    province_counts = (
        filtered.groupby("Province", dropna=False)
        .size()
        .reset_index(name="Accident reports")
        .sort_values("Accident reports", ascending=False)
    )

    province_fig = px.bar(
        province_counts,
        x="Province",
        y="Accident reports",
        title="Accident Reports by Province",
        labels={"Province": "Province"}
    )

    province_fig.update_layout(
        xaxis_tickangle=-35,
        height=400
    )

    st.plotly_chart(
        province_fig,
        use_container_width=True
    )

with right_col:
    cause_counts = (
        filtered["Cause"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(10)
        .rename_axis("Cause")
        .reset_index(name="Accident reports")
        .sort_values("Accident reports", ascending=True)
    )

    cause_fig = px.bar(
        cause_counts,
        x="Accident reports",
        y="Cause",
        orientation="h",
        title="Most Frequently Reported Causes"
    )

    cause_fig.update_layout(height=400)

    st.plotly_chart(
        cause_fig,
        use_container_width=True
    )


# --------------------------------------------------
# 8. FILTERED DATA TABLE AND DOWNLOAD
# --------------------------------------------------

st.divider()
st.subheader("Accident Records")

with st.expander("View and download filtered records"):
    display_columns = [
        col for col in [
            "ID", "Date", "Time", "Province", "City or Town",
            "Road", "Location", "Cause", "Vehicles Involved",
            "Injured", "Fatalities", "Lat", "Long", "Source"
        ]
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns].sort_values(
            "Date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    download_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered CSV",
        data=download_data,
        file_name="filtered_road_accidents.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Zimbabwe Road Traffic Accident Analytics | "
    "Source: Z.R.P X page "
)
