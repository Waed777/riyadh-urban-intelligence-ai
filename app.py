import requests
import math
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# RIYADH URBAN INTELLIGENCE AI
# RCRC Open Data + Geospatial Analytics + ML-ready Prototype
# ============================================================

st.set_page_config(
    page_title="Riyadh Urban Intelligence AI",
    page_icon="🇸🇦",
    layout="wide"
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RCRC_API = (
    "https://opendata.rcrc.gov.sa/"
    "api/explore/v2.1/catalog/datasets/"
    "metro-stations-in-riyadh-by-metro-line-and-station-type-2024/"
    "records"
)

# Riyadh approximate center
RIYADH_LAT = 24.7136
RIYADH_LON = 46.6753


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points."""

    r = 6371.0

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

    return 2 * r * math.asin(math.sqrt(a))


@st.cache_data(ttl=3600)
def load_metro_data():

    params = {
        "limit": 100
    }

    response = requests.get(
        RCRC_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    records = data.get("results", [])

    if not records:
        raise ValueError("RCRC returned no metro station records.")

    rows = []

    for item in records:

        geo = item.get("geo_point_2d")

        if isinstance(geo, dict):

            lat = geo.get("lat")
            lon = geo.get("lon")

        elif isinstance(geo, str):

            lat = None
            lon = None

        else:

            lat = None
            lon = None

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
            "Metro data loaded but no valid coordinates were found."
        )

    return df


def calculate_accessibility_score(distance_km):

    """
    Prototype accessibility score.

    Closer to a metro station = higher accessibility.

    This is a decision-support prototype,
    NOT a prediction of business success.
    """

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

    if score >= 40:
        return "Medium Accessibility"

    return "Low Accessibility"


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title(
    "🇸🇦 Riyadh Urban Intelligence AI"
)

st.markdown(
    """
### Turning Riyadh Open Data into Location Intelligence

**RCRC Open Data → Geospatial Analytics → AI Accessibility Score → Interactive Intelligence**
"""
)

st.divider()


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

try:

    with st.spinner(
        "Loading Riyadh Metro data from RCRC..."
    ):

        metro = load_metro_data()

except Exception as e:

    st.error(
        "Unable to load RCRC data."
    )

    st.code(
        str(e)
    )

    st.stop()


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

metro["distance_from_riyadh_center_km"] = metro.apply(
    lambda row: haversine_km(
        RIYADH_LAT,
        RIYADH_LON,
        row["latitude"],
        row["longitude"]
    ),
    axis=1
)

metro["accessibility_score"] = metro[
    "distance_from_riyadh_center_km"
].apply(
    calculate_accessibility_score
)

metro["zone"] = metro[
    "accessibility_score"
].apply(
    classify_zone
)


# ------------------------------------------------------------
# KPIs
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Metro Stations",
    len(metro)
)

col2.metric(
    "Metro Lines",
    metro["metro_line"].nunique()
)

col3.metric(
    "Average Accessibility",
    f"{metro['accessibility_score'].mean():.1f}"
)

col4.metric(
    "Highest Score",
    f"{metro['accessibility_score'].max():.1f}"
)


st.divider()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header(
    "🎛️ Intelligence Controls"
)

selected_line = st.sidebar.selectbox(
    "Metro Line",
    ["All"] +
    sorted(
        metro["metro_line"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
)

min_score = st.sidebar.slider(
    "Minimum AI Score",
    min_value=0,
    max_value=100,
    value=0
)


# ------------------------------------------------------------
# Filtering
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Top locations
# ------------------------------------------------------------

st.subheader(
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
            "zone"
        ]
    ]
    .reset_index(drop=True)
)

top_locations.index += 1

top_locations.columns = [
    "Station",
    "Metro Line",
    "Station Type",
    "Distance from Riyadh Center (km)",
    "AI Score",
    "AI Segment"
]

st.dataframe(
    top_locations,
    use_container_width=True,
    hide_index=False
)


# ------------------------------------------------------------
# Map
# ------------------------------------------------------------

st.subheader(
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
        "longitude": False,
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


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

st.subheader(
    "📊 Accessibility Distribution"
)

distribution = (
    metro["zone"]
    .value_counts()
    .reset_index()
)

distribution.columns = [
    "Zone",
    "Stations"
]

bar = px.bar(
    distribution,
    x="Zone",
    y="Stations",
    title="Metro Stations by AI Accessibility Segment"
)

st.plotly_chart(
    bar,
    use_container_width=True
)


# ------------------------------------------------------------
# AI Explanation
# ------------------------------------------------------------

st.subheader(
    "🧠 AI Interpretation"
)

best = metro.sort_values(
    "accessibility_score",
    ascending=False
).iloc[0]

st.info(
    f"""
**Top-ranked station:** {best['station_name']}

**AI Accessibility Score:** {best['accessibility_score']:.1f}/100

**Distance from Riyadh center:** \
{best['distance_from_riyadh_center_km']:.2f} km

**Metro line:** {best['metro_line']}

### Why this score?

The prototype currently uses geographic proximity to Riyadh's
metro infrastructure as an accessibility signal.

Closer proximity produces a higher accessibility score.

This is a **location-intelligence prototype**, not a prediction
of business success.

Future versions will incorporate additional urban signals such as
bus accessibility, traffic intersections, population, services,
and real-world outcome data.
"""
)


# ------------------------------------------------------------
# Download data
# ------------------------------------------------------------

st.subheader(
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
    file_name="riyadh_urban_intelligence_ai.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    """
Data source: Riyadh Royal Commission (RCRC) Open Data Portal.

Project: Riyadh Urban Intelligence AI

Built with Python • Pandas • Plotly • Streamlit • Geospatial Analytics
"""
)
