import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="Catastrophe Event Intelligence Dashboard",
    page_icon="🌪️",
    layout="wide",
)


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "final_catastrophe_event_dataset.csv"


# -----------------------------
# Load data
# -----------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            "Final dataset not found. Run `python src/run_pipeline.py` first."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    return df


df = load_data()


# -----------------------------
# Sidebar filters
# -----------------------------

st.sidebar.title("Filters")

source_options = sorted(df["source"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect(
    "Source",
    options=source_options,
    default=source_options,
)

risk_options = sorted(df["cat_risk_bucket"].dropna().unique().tolist())
selected_risk_buckets = st.sidebar.multiselect(
    "Risk bucket",
    options=risk_options,
    default=risk_options,
)

exposure_options = sorted(df["ils_exposure_zone"].dropna().unique().tolist())
selected_exposure_zones = st.sidebar.multiselect(
    "ILS exposure zone",
    options=exposure_options,
    default=exposure_options,
)

event_type_options = sorted(df["event_type"].dropna().unique().tolist())
selected_event_types = st.sidebar.multiselect(
    "Event type",
    options=event_type_options,
    default=event_type_options,
)


filtered_df = df[
    df["source"].isin(selected_sources)
    & df["cat_risk_bucket"].isin(selected_risk_buckets)
    & df["ils_exposure_zone"].isin(selected_exposure_zones)
    & df["event_type"].isin(selected_event_types)
].copy()


# -----------------------------
# Header
# -----------------------------

st.title("Catastrophe Event Intelligence Dashboard")

st.markdown(
    """
    This dashboard visualises an automated catastrophe event extraction pipeline
    using USGS, GDACS and NOAA data. Events are enriched with regex extraction,
    NLP features, heuristic risk scoring and ILS exposure-zone tagging.
    """
)


# -----------------------------
# KPI row
# -----------------------------

total_events = len(filtered_df)
unique_sources = filtered_df["source"].nunique()
high_or_severe = filtered_df[
    filtered_df["cat_risk_bucket"].isin(["High", "Severe"])
].shape[0]
unique_exposure_zones = filtered_df["ils_exposure_zone"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Filtered events", f"{total_events:,}")
col2.metric("Sources", unique_sources)
col3.metric("High / Severe events", f"{high_or_severe:,}")
col4.metric("Exposure zones", unique_exposure_zones)


# -----------------------------
# Charts
# -----------------------------

st.subheader("Dataset Overview")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    source_counts = (
        filtered_df["source"]
        .value_counts()
        .reset_index()
    )
    source_counts.columns = ["source", "count"]

    fig_source = px.bar(
        source_counts,
        x="source",
        y="count",
        title="Events by Source",
    )

    st.plotly_chart(fig_source, use_container_width=True)

with chart_col2:
    risk_counts = (
        filtered_df["cat_risk_bucket"]
        .value_counts()
        .reset_index()
    )
    risk_counts.columns = ["risk_bucket", "count"]

    fig_risk = px.bar(
        risk_counts,
        x="risk_bucket",
        y="count",
        title="Events by Risk Bucket",
    )

    st.plotly_chart(fig_risk, use_container_width=True)


st.subheader("ILS Exposure-Zone Breakdown")

top_exposure = (
    filtered_df["ils_exposure_zone"]
    .value_counts()
    .head(20)
    .reset_index()
)

top_exposure.columns = ["ils_exposure_zone", "count"]

fig_exposure = px.bar(
    top_exposure,
    x="count",
    y="ils_exposure_zone",
    orientation="h",
    title="Top ILS Exposure Zones",
)

fig_exposure.update_layout(yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_exposure, use_container_width=True)


# -----------------------------
# Event type breakdown
# -----------------------------

st.subheader("Event-Type Breakdown")

top_event_types = (
    filtered_df["event_type"]
    .value_counts()
    .head(20)
    .reset_index()
)

top_event_types.columns = ["event_type", "count"]

fig_event_type = px.bar(
    top_event_types,
    x="count",
    y="event_type",
    orientation="h",
    title="Top Event Types",
)

fig_event_type.update_layout(yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig_event_type, use_container_width=True)


# -----------------------------
# Map
# -----------------------------

st.subheader("Event Map")

map_df = filtered_df.dropna(subset=["latitude", "longitude"]).copy()

if not map_df.empty:
    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
    map_df = map_df.dropna(subset=["latitude", "longitude"])

    if not map_df.empty:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="cat_risk_bucket",
            size="cat_risk_score",
            hover_name="event_name",
            hover_data=[
                "source",
                "event_type",
                "region",
                "ils_exposure_zone",
                "cat_risk_score",
            ],
            zoom=1,
            height=600,
            title="Mapped Catastrophe Events",
        )

        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )

        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No valid latitude/longitude values available after filtering.")
else:
    st.info("No mappable events available for the selected filters.")


# -----------------------------
# Highest-risk events table
# -----------------------------

st.subheader("Highest-Risk Events")

table_cols = [
    "source",
    "event_type",
    "event_name",
    "region",
    "country",
    "cat_risk_score",
    "cat_risk_bucket",
    "ils_exposure_zone",
    "economic_damage_usd",
    "regex_wind_speed_mph",
    "regex_precipitation_inches",
    "fatalities",
    "injuries",
    "description",
]

existing_table_cols = [col for col in table_cols if col in filtered_df.columns]

top_events = (
    filtered_df
    .sort_values("cat_risk_score", ascending=False)
    [existing_table_cols]
    .head(50)
)

st.dataframe(
    top_events,
    use_container_width=True,
    height=500,
)

# -----------------------------
# Local LLM extraction table
# -----------------------------

st.subheader("Local LLM Extraction: High-Risk Event Narratives")

LLM_PATH = BASE_DIR / "data" / "processed" / "llm_extracted_high_risk_events_ollama.csv"

if LLM_PATH.exists():
    llm_df = pd.read_csv(LLM_PATH, low_memory=False)

    llm_display_cols = [
        "event_name",
        "region",
        "cat_risk_score",
        "ils_exposure_zone",
        "llm_summary",
        "llm_severity_indicators",
        "llm_human_impact",
        "llm_investment_relevance",
        "llm_confidence_score",
        "llm_parse_error",
    ]

    existing_llm_cols = [col for col in llm_display_cols if col in llm_df.columns]

    st.markdown(
        """
        Selected high-risk event narratives are processed with a local Ollama LLM
        to generate structured summaries, severity indicators, human-impact notes
        and ILS relevance flags.
        """
    )

    st.dataframe(
        llm_df[existing_llm_cols],
        use_container_width=True,
        height=400,
    )
else:
    st.info(
        "Local LLM extraction output not found. Run `python src/llm_extract_ollama.py` first."
    )


# -----------------------------
# SQLite note
# -----------------------------

st.subheader("SQLite Database Layer")

st.markdown(
    """
    The final enriched dataset is also exported to SQLite as `cat_events.db`.
    The database contains an `events` table and reusable SQL views for source summaries,
    exposure-zone summaries and high-risk event monitoring.
    """
)
# -----------------------------
# Raw data explorer
# -----------------------------

with st.expander("Show raw filtered data"):
    st.dataframe(filtered_df, use_container_width=True)