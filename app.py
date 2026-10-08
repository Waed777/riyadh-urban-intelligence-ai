import math

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from ml_engine import (
    segment_zones,
    train_explainability_model
)


# ============================================================
# RIYADH URBAN INTELLIGENCE AI
# ============================================================

st.set_page_config(
    page_title="Riyadh Urban Intelligence AI",
    page_icon="🇸🇦",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

RCRC_API = (
    "https://opendata.rcrc.gov.sa/"
    "api/explore/v2.1/catalog/datasets/"
    "metro-stations-in-riyadh-by-metro-line-and-station-type-2024/"
    "records"
)

RIYADH_LAT = 24.7136
RIYADH_LON = 46.6753


# ============================================================
# GEOSPATIAL FUNCTIONS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(
        math.sqrt(a)
    )


def calculate_accessibility_score(distance_km):

    score = 100 * math.exp(
        -distance_km / 2.0
    )

    return round(
        max(0, min(100, score)),
        2
    )


def classify_zone(score):

    if score >= 70:
        return "High Accessibility"

    elif score >= 40:
        return "Medium Accessibility"

    return "Low Accessibility"


# ============================================================
# RCRC DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_metro_data():

    response = requests.get(
        RCRC_API,
        params={"limit": 100},
        timeout=30
    )

    response.raise_for_status()

    records = response.json().get(
        "results",
        []
    )

    if not records:
        raise ValueError(
            "RCRC returned no metro station records."
        )

    rows = []

    for item in records:

        geo = item.get(
            "geo_point_2d"
        )

        lat = None
        lon = None

        if isinstance(geo, dict):

            lat = geo.get("lat")
            lon = geo.get("lon")

        if lat is None or lon is None:
            continue

        rows.append(
            {
                "station_code": item.get(
                    "metro_station_code"
                ),

                "station_name": item.get(
                    "metro_station_name"
                ),

                "metro_line": item.get(
                    "metro_line"
                ),

                "station_type": item.get(
                    "station_type"
                ),

                "sequence": item.get(
                    "sequence"
                ),

                "latitude": float(lat),

                "longitude": float(lon),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:

        raise ValueError(
            "No valid geographic coordinates were found."
        )

    return df


# ============================================================
# HEADER
# ============================================================

st.title(
    "🇸🇦 Riyadh Urban Intelligence AI"
)

st.markdown(
    """
### Turning Riyadh Open Data into Location Intelligence

**RCRC Open Data → Geospatial Analytics → Machine Learning → Interactive Intelligence**
"""
)

st.divider()


# ============================================================
# LOAD RCRC DATA
# ============================================================

try:

    with st.spinner(
        "Loading Riyadh Metro data from RCRC..."
    ):

        metro = load_metro_data()

except Exception as error:

    st.error(
        "Unable to load RCRC data."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

metro[
    "distance_from_riyadh_center_km"
] = metro.apply(
    lambda row: haversine_km(
        RIYADH_LAT,
        RIYADH_LON,
        row["latitude"],
        row["longitude"]
    ),
    axis=1
)


metro[
    "accessibility_score"
] = metro[
    "distance_from_riyadh_center_km"
].apply(
    calculate_accessibility_score
)


metro["zone"] = metro[
    "accessibility_score"
].apply(
    classify_zone
)


# ============================================================
# MACHINE LEARNING
# ============================================================

try:

    ml_data, clustering_model = segment_zones(
        metro,
        n_clusters=4
    )

    explainability_model, feature_importance = (
        train_explainability_model(
            metro
        )
    )

    metro = ml_data

except Exception as error:

    st.warning(
        "ML layer could not be initialized."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎛️ Intelligence Controls"
)

selected_line = st.sidebar.selectbox(
    "Metro Line",
    ["All"]
    + sorted(
        metro["metro_line"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
)

min_score = st.sidebar.slider(
    "Minimum AI Score",
    0,
    100,
    0
)


# ============================================================
# FILTERING
# ============================================================

filtered = metro.copy()

if selected_line != "All":

    filtered = filtered[
        filtered["metro_line"].astype(str)
        == selected_line
    ]

filtered = filtered[
    filtered["accessibility_score"]
    >= min_score
]


# ============================================================
# KPI DASHBOARD
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🚇 Metro Stations",
    len(metro)
)

col2.metric(
    "🛤️ Metro Lines",
    metro["metro_line"].nunique()
)

col3.metric(
    "📊 Average AI Score",
    f"{metro['accessibility_score'].mean():.1f}"
)

col4.metric(
    "🏆 Highest Score",
    f"{metro['accessibility_score'].max():.1f}"
)


st.divider()


# ============================================================
# TOP LOCATIONS
# ============================================================

st.header(
    "🏆 Highest Accessibility Locations"
)

top_locations = (
    filtered
    .sort_values(
        "accessibility_score",
        ascending=False
    )
    .head(10)
    [
        [
            "station_name",
            "metro_line",
            "station_type",
            "distance_from_riyadh_center_km",
            "accessibility_score",
            "ai_zone_segment"
        ]
    ]
    .reset_index(drop=True)
)

top_locations.index += 1

top_locations.columns = [
    "Station",
    "Metro Line",
    "Station Type",
    "Distance from Center (km)",
    "AI Score",
    "ML Zone"
]

st.dataframe(
    top_locations,
    use_container_width=True
)


# ============================================================
# MAP
# ============================================================

st.header(
    "🗺️ Riyadh Metro Intelligence Map"
)

fig = px.scatter_map(
    filtered,
    lat="latitude",
    lon="longitude",
    color="accessibility_score",
    size="accessibility_score",
    hover_name="station_name",
    hover_data={
        "metro_line": True,
        "station_type": True,
        "distance_from_riyadh_center_km": ":.2f",
        "accessibility_score": ":.2f",
        "latitude": False,
        "longitude": False
    },
    zoom=9,
    height=650,
    map_style="open-street-map",
    title="AI Accessibility Score by Metro Station"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ML SEGMENTS
# ============================================================

st.divider()

st.header(
    "🤖 Machine Learning Intelligence"
)

st.markdown(
    """
The system uses **K-Means clustering** to segment metro
locations according to their accessibility signals.
"""
)

segment_counts = (
    metro[
        "ai_zone_segment"
    ]
    .value_counts()
    .reset_index()
)

segment_counts.columns = [
    "AI Zone",
    "Stations"
]

col1, col2 = st.columns(2)

with col1:

    st.dataframe(
        segment_counts,
        use_container_width=True,
        hide_index=True
    )

with col2:

    segment_chart = px.bar(
        segment_counts,
        x="AI Zone",
        y="Stations",
        title="Stations by ML Segment"
    )

    st.plotly_chart(
        segment_chart,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header(
    "🔍 Model Feature Importance"
)

importance_chart = px.bar(
    feature_importance,
    x="importance",
    y="feature",
    orientation="h",
    title="Signals Used by the Explainability Model"
)

st.plotly_chart(
    importance_chart,
    use_container_width=True
)


# ============================================================
# AI ZONE EXPLORER
# ============================================================

st.header(
    "📍 AI Zone Explorer"
)

selected_zone = st.selectbox(
    "Select an AI Zone",
    sorted(
        metro[
            "ai_zone_segment"
        ]
        .dropna()
        .unique()
        .tolist()
    )
)

zone_data = metro[
    metro["ai_zone_segment"]
    == selected_zone
]

zone_col1, zone_col2, zone_col3 = st.columns(3)

zone_col1.metric(
    "Stations",
    len(zone_data)
)

zone_col2.metric(
    "Average AI Score",
    f"{zone_data['accessibility_score'].mean():.1f}"
)

zone_col3.metric(
    "Average Distance",
    f"{zone_data['distance_from_riyadh_center_km'].mean():.2f} km"
)


st.dataframe(
    zone_data[
        [
            "station_name",
            "metro_line",
            "accessibility_score",
            "ai_zone_segment"
        ]
    ]
    .sort_values(
        "accessibility_score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

st.header(
    "📥 Export Intelligence Data"
)

csv = metro.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download Riyadh AI Dataset",
    data=csv,
    file_name=(
        "riyadh_urban_intelligence_ai.csv"
    ),
    mime="text/csv"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.info(
    """
**Prototype note**

This system is an exploratory location-intelligence prototype.

The current accessibility score is based primarily on
geographic proximity to Riyadh Metro infrastructure.

It does **not** claim to predict business success.

Future versions will incorporate validated RCRC layers,
mobility signals, land use, commercial services, real-world
outcomes, and an LLM/RAG explanation layer.
"""
)


st.caption(
    """
Data source: Riyadh Royal Commission (RCRC) Open Data Portal

Riyadh Urban Intelligence AI
Built with Python • Pandas • Scikit-learn • Plotly • Streamlit
"""
)
