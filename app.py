import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

from ml_engine import segment_zones, train_explainability_model


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Riyadh Urban Intelligence AI",
    page_icon="🇸🇦",
    layout="wide"
)

st.title("🇸🇦 Riyadh Urban Intelligence AI")
st.caption(
    "AI-powered geospatial intelligence and location decision support for Riyadh"
)


# ============================================================
# DATA
# ============================================================

API_URL = (
    "https://opendata.rcrc.gov.sa/api/explore/v2.1/catalog/datasets/"
    "metro-stations-in-riyadh-by-metro-line-and-station-type-2024/"
    "records"
)

RIYADH_LAT = 24.7136
RIYADH_LON = 46.6753


@st.cache_data
def load_metro_data():

    response = requests.get(
        API_URL,
        params={"limit": 100},
        timeout=30
    )

    response.raise_for_status()

    records = response.json()["results"]

    rows = []

    for r in records:

        geo = r.get("geo_point_2d")

        if not geo:
            continue

        rows.append(
            {
                "station_name": r.get("station_name"),
                "metro_line": r.get("metro_line"),
                "station_type": r.get("station_type"),
                "latitude": geo.get("lat"),
                "longitude": geo.get("lon"),
            }
        )

    return pd.DataFrame(rows)


metro = load_metro_data()


# ============================================================
# GEOSPATIAL FUNCTIONS
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def calculate_accessibility_score(distance_km):

    score = 100 * np.exp(-distance_km / 2)

    return float(np.clip(score, 0, 100))


def classify_score(score):

    if score >= 80:
        return "Prime Accessibility Zone"

    elif score >= 60:
        return "Strong Accessibility Zone"

    elif score >= 40:
        return "Emerging Accessibility Zone"

    return "Lower Accessibility Zone"


# ============================================================
# CALCULATE METRO FEATURES
# ============================================================

metro["distance_from_riyadh_center_km"] = haversine_distance(
    RIYADH_LAT,
    RIYADH_LON,
    metro["latitude"],
    metro["longitude"]
)

metro["accessibility_score"] = (
    100
    * np.exp(
        -metro["distance_from_riyadh_center_km"] / 2
    )
).clip(0, 100)

metro["zone"] = metro["accessibility_score"].apply(
    classify_score
)


# ============================================================
# ML
# ============================================================

try:

    ml_data, cluster_model = segment_zones(
        metro,
        n_clusters=4
    )

    explain_model, importance = train_explainability_model(
        metro
    )

except Exception:

    ml_data = metro.copy()

    ml_data["ai_zone_segment"] = ml_data["zone"]

    importance = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Explore Riyadh")

lines = sorted(
    metro["metro_line"]
    .dropna()
    .unique()
    .tolist()
)

selected_lines = st.sidebar.multiselect(
    "Metro Line",
    lines,
    default=lines
)

min_score = st.sidebar.slider(
    "Minimum Accessibility Score",
    0,
    100,
    0
)


filtered = metro[
    metro["metro_line"].isin(selected_lines)
    &
    (metro["accessibility_score"] >= min_score)
]


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Application",
    [
        "🗺️ Explore Riyadh",
        "📍 Location Decision Engine"
    ],
    horizontal=True
)


# ============================================================
# PAGE 1 — EXPLORE
# ============================================================

if page == "🗺️ Explore Riyadh":

    st.subheader("Riyadh Metro Accessibility Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🚇 Stations",
        len(filtered)
    )

    col2.metric(
        "🛤️ Metro Lines",
        filtered["metro_line"].nunique()
    )

    col3.metric(
        "📊 Average Score",
        f"{filtered['accessibility_score'].mean():.1f}"
    )

    col4.metric(
        "🏆 Highest Score",
        f"{filtered['accessibility_score'].max():.1f}"
    )

    st.divider()

    st.subheader("📊 Top Accessibility Locations")

    top_locations = (
        filtered[
            [
                "station_name",
                "metro_line",
                "distance_from_riyadh_center_km",
                "accessibility_score",
                "zone"
            ]
        ]
        .sort_values(
            "accessibility_score",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_locations,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🗺️ Interactive Riyadh Map")

    fig = px.scatter_map(
        filtered,
        lat="latitude",
        lon="longitude",
        hover_name="station_name",
        hover_data=[
            "metro_line",
            "accessibility_score",
            "zone"
        ],
        size="accessibility_score",
        zoom=9,
        height=600,
        map_style="open-street-map"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("🤖 ML Zone Segmentation")

    segment_counts = (
        ml_data["ai_zone_segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Zone",
        "Locations"
    ]

    fig_segments = px.bar(
        segment_counts,
        x="Zone",
        y="Locations",
        title="AI Zone Distribution"
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True
    )

    if not importance.empty:

        st.subheader("🧠 Explainable AI")

        fig_importance = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )


# ============================================================
# PAGE 2 — LOCATION DECISION ENGINE
# ============================================================

else:

    st.header("📍 Location Decision Engine")

    st.write(
        "Evaluate a potential location using Riyadh Metro "
        "accessibility and machine-learning signals."
    )

    st.info(
        "💡 Enter the coordinates of a location you want to evaluate."
    )

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            value=RIYADH_LAT,
            format="%.6f"
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            value=RIYADH_LON,
            format="%.6f"
        )

    business_type = st.selectbox(
        "🏢 Location Type",
        [
            "Retail",
            "Café / Restaurant",
            "Healthcare",
            "Education",
            "Office",
            "Logistics",
            "General"
        ]
    )

    analyze = st.button(
        "🚀 Analyze Location",
        use_container_width=True
    )

    if analyze:

        # ----------------------------------------------------
        # DISTANCE TO EVERY METRO STATION
        # ----------------------------------------------------

        distances = haversine_distance(
            latitude,
            longitude,
            metro["latitude"],
            metro["longitude"]
        )

        nearest_index = distances.idxmin()

        nearest_station = metro.loc[
            nearest_index
        ]

        nearest_distance = float(
            distances.loc[nearest_index]
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        accessibility_score = calculate_accessibility_score(
            nearest_distance
        )

        zone = classify_score(
            accessibility_score
        )

        # ----------------------------------------------------
        # CENTRALITY
        # ----------------------------------------------------

        center_distance = haversine_distance(
            latitude,
            longitude,
            RIYADH_LAT,
            RIYADH_LON
        )

        centrality_score = calculate_accessibility_score(
            center_distance
        )

        # ----------------------------------------------------
        # OVERALL SCORE
        # ----------------------------------------------------

        overall_score = (
            accessibility_score * 0.75
            +
            centrality_score * 0.25
        )

        overall_score = round(
            float(
                np.clip(
                    overall_score,
                    0,
                    100
                )
            ),
            1
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Location Intelligence Report"
        )

        score_col1, score_col2, score_col3 = st.columns(3)

        score_col1.metric(
            "🤖 AI Location Score",
            f"{overall_score}/100"
        )

        score_col2.metric(
            "🚇 Metro Accessibility",
            f"{accessibility_score:.1f}/100"
        )

        score_col3.metric(
            "🏙️ Centrality",
            f"{centrality_score:.1f}/100"
        )

        st.divider()

        # ----------------------------------------------------
        # LOCATION DETAILS
        # ----------------------------------------------------

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "📏 Nearest Metro",
            f"{nearest_distance:.2f} km"
        )

        d2.metric(
            "🚇 Station",
            str(nearest_station["station_name"])
        )

        d3.metric(
            "🛤️ Metro Line",
            str(nearest_station["metro_line"])
        )

        d4.metric(
            "🏆 AI Zone",
            zone
        )

        st.divider()

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("💡 AI Recommendation")

        if overall_score >= 80:

            recommendation = (
                f"🟢 **Strong candidate location.** "
                f"The location has strong accessibility "
                f"and good proximity to Riyadh's urban center."
            )

        elif overall_score >= 60:

            recommendation = (
                f"🟡 **Potential candidate.** "
                f"The location shows moderate accessibility "
                f"but additional business and mobility data "
                f"should be evaluated."
            )

        else:

            recommendation = (
                f"🔴 **Lower accessibility signal.** "
                f"The location is relatively distant from "
                f"the current metro network or Riyadh center."
            )

        st.write(recommendation)

        st.divider()

        # ----------------------------------------------------
        # WHY THIS SCORE?
        # ----------------------------------------------------

        st.subheader("🧠 Why this score?")

        explanation = pd.DataFrame(
            {
                "Signal": [
                    "Metro Accessibility",
                    "Urban Centrality"
                ],
                "Score": [
                    round(accessibility_score, 1),
                    round(centrality_score, 1)
                ],
                "Weight": [
                    "75%",
                    "25%"
                ]
            }
        )

        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Business context selected: {business_type}. "
            "Business type does not yet change the model weights; "
            "this will be introduced in a future version."
        )

        st.divider()

        # ----------------------------------------------------
        # MAP
        # ----------------------------------------------------

        st.subheader("📍 Evaluated Location")

        location_df = pd.DataFrame(
            [
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "type": "Selected Location"
                }
            ]
        )

        station_df = pd.DataFrame(
            [
                {
                    "latitude": nearest_station["latitude"],
                    "longitude": nearest_station["longitude"],
                    "type": "Nearest Metro"
                }
            ]
        )

        map_df = pd.concat(
            [
                location_df,
                station_df
            ],
            ignore_index=True
        )

        fig_location = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            color="type",
            zoom=12,
            height=500,
            map_style="open-street-map"
        )

        st.plotly_chart(
            fig_location,
            use_container_width=True
        )

        st.divider()

        st.warning(
            "⚠️ This is an exploratory decision-support prototype. "
            "The score does not predict business success. "
            "Future versions will incorporate validated bus, "
            "traffic, land-use, commercial and mobility signals."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🇸🇦 Riyadh Urban Intelligence AI | "
    "Data Science • Machine Learning • Geospatial AI • GenAI"
)
