import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_engine import (
    load_rcrc_layers,
    build_candidate_grid,
    calculate_siteiq_score
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Riyadh SiteIQ",
    page_icon="🇸🇦",
    layout="wide"
)

st.title("🇸🇦 Riyadh SiteIQ")
st.caption(
    "AI-powered location intelligence and opportunity discovery"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_data():

    layers, discovery = load_rcrc_layers()

    candidates = build_candidate_grid(
        layers,
        lat_min=24.55,
        lat_max=25.00,
        lon_min=46.45,
        lon_max=47.05,
        step=0.02
    )

    candidates = calculate_siteiq_score(
        candidates
    )

    return layers, discovery, candidates


with st.spinner("Loading Riyadh intelligence data..."):

    try:

        layers, discovery, candidates = load_data()

        data_ready = True

    except Exception as e:

        data_ready = False

        st.error(
            f"Data engine error: {e}"
        )


# ============================================================
# DATA STATUS
# ============================================================

if data_ready:

    with st.expander(
        "🔌 RCRC Data Sources",
        expanded=False
    ):

        st.dataframe(
            discovery,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 SiteIQ Configuration")

business_type = st.sidebar.selectbox(
    "Business Type",
    [
        "Café / Restaurant",
        "Retail",
        "Healthcare",
        "Education",
        "Office",
        "Logistics",
        "General"
    ]
)

accessibility_weight = st.sidebar.slider(
    "🚇 Accessibility",
    0,
    100,
    30
)

mobility_weight = st.sidebar.slider(
    "🚦 Mobility",
    0,
    100,
    20
)

competition_weight = st.sidebar.slider(
    "🏪 Opportunity Gap",
    0,
    100,
    30
)

transit_weight = st.sidebar.slider(
    "🚌 Transit",
    0,
    100,
    20
)

total_weight = (
    accessibility_weight
    + mobility_weight
    + competition_weight
    + transit_weight
)

st.sidebar.metric(
    "Total Weight",
    f"{total_weight}%"
)

# ============================================================
# MAIN
# ============================================================

if not data_ready:

    st.stop()


# ============================================================
# HEADER METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "📍 Candidate Locations",
    f"{len(candidates):,}"
)

c2.metric(
    "🚇 Metro Records",
    f"{len(layers.get('metro', [])):,}"
)

c3.metric(
    "🚌 Bus Records",
    f"{len(layers.get('bus', [])):,}"
)

c4.metric(
    "🏪 Commercial Records",
    f"{len(layers.get('commercial', [])):,}"
)

st.divider()


# ============================================================
# FIND OPPORTUNITIES
# ============================================================

st.header("🔎 Find Business Opportunities")

st.write(
    f"Discover high-potential candidate locations for "
    f"**{business_type}** across Riyadh."
)

find = st.button(
    "🚀 FIND OPPORTUNITIES",
    type="primary",
    use_container_width=True
)


if find:

    if total_weight == 0:

        st.error(
            "Please assign at least one weight."
        )

        st.stop()

    # --------------------------------------------------------
    # CUSTOM SCORE
    # --------------------------------------------------------

    ranking = candidates.copy()

    ranking["custom_siteiq_score"] = (
        ranking["accessibility_score"]
        * accessibility_weight
        +
        ranking["mobility_score"]
        * mobility_weight
        +
        ranking["competition_gap_score"]
        * competition_weight
        +
        ranking["transit_score"]
        * transit_weight
    ) / total_weight

    ranking["custom_siteiq_score"] = (
        ranking["custom_siteiq_score"]
        .clip(0, 100)
        .round(1)
    )

    ranking = ranking.sort_values(
        "custom_siteiq_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    ranking["rank"] = (
        ranking.index + 1
    )

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------

    top10 = ranking.head(10)

    st.success(
        f"Found {len(top10)} leading opportunities "
        f"for {business_type}."
    )

    st.header("🏆 Top Riyadh Opportunities")

    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    top1, top2, top3 = st.columns(3)

    cards = [
        (top1, 0),
        (top2, 1),
        (top3, 2)
    ]

    for column, index in cards:

        if index >= len(top10):
            continue

        row = top10.iloc[index]

        with column:

            st.subheader(
                f"#{index + 1} Opportunity"
            )

            st.metric(
                "SiteIQ Score",
                f"{row['custom_siteiq_score']}/100"
            )

            st.write(
                f"📍 **{row['latitude']:.4f}, "
                f"{row['longitude']:.4f}**"
            )

            st.write(
                f"🚇 Metro: "
                f"{row['metro_distance_km']:.2f} km"
            )

            st.write(
                f"🚌 Bus: "
                f"{row['bus_distance_km']:.2f} km"
            )

            st.write(
                f"🚦 Intersections: "
                f"{int(row['traffic_intersections_2km'])}"
            )

            st.write(
                f"🏪 Services: "
                f"{int(row['commercial_services_2km'])}"
            )

    st.divider()

    # ========================================================
    # RANKING TABLE
    # ========================================================

    st.subheader("📊 Top 10 Ranking")

    display = top10[
        [
            "rank",
            "latitude",
            "longitude",
            "custom_siteiq_score",
            "accessibility_score",
            "transit_score",
            "mobility_score",
            "competition_gap_score",
            "metro_distance_km",
            "bus_distance_km",
            "traffic_intersections_2km",
            "commercial_services_2km"
        ]
    ].copy()

    display.columns = [
        "Rank",
        "Latitude",
        "Longitude",
        "SiteIQ",
        "Accessibility",
        "Transit",
        "Mobility",
        "Opportunity Gap",
        "Metro km",
        "Bus km",
        "Traffic",
        "Commercial"
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # OPPORTUNITY MAP
    # ========================================================

    st.subheader(
        "🗺️ Riyadh Opportunity Map"
    )

    map_data = ranking.head(1000).copy()

    fig = px.scatter_map(
        map_data,
        lat="latitude",
        lon="longitude",
        color="custom_siteiq_score",
        size="custom_siteiq_score",
        hover_name="custom_siteiq_score",
        hover_data={
            "latitude": ":.5f",
            "longitude": ":.5f",
            "custom_siteiq_score": ":.1f",
            "metro_distance_km": ":.2f",
            "bus_distance_km": ":.2f",
            "traffic_intersections_2km": True,
            "commercial_services_2km": True
        },
        zoom=9,
        height=650,
        map_style="open-street-map"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # SELECT LOCATION
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Analyze an Opportunity"
    )

    selected_rank = st.selectbox(
        "Select ranked location",
        top10["rank"].tolist()
    )

    selected = ranking[
        ranking["rank"] == selected_rank
    ].iloc[0]

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "SiteIQ Score",
        f"{selected['custom_siteiq_score']:.1f}"
    )

    s2.metric(
        "Metro Distance",
        f"{selected['metro_distance_km']:.2f} km"
    )

    s3.metric(
        "Opportunity Gap",
        f"{selected['competition_gap_score']:.1f}"
    )

    # ========================================================
    # WHY?
    # ========================================================

    st.subheader(
        "🧠 Why was this location ranked here?"
    )

    factors = pd.DataFrame(
        {
            "Signal": [
                "Accessibility",
                "Transit",
                "Mobility",
                "Opportunity Gap"
            ],
            "Score": [
                selected["accessibility_score"],
                selected["transit_score"],
                selected["mobility_score"],
                selected["competition_gap_score"]
            ],
            "Weight": [
                accessibility_weight,
                transit_weight,
                mobility_weight,
                competition_weight
            ]
        }
    )

    factors["Contribution"] = (
        factors["Score"]
        * factors["Weight"]
        / total_weight
    ).round(1)

    st.dataframe(
        factors,
        use_container_width=True,
        hide_index=True
    )

    fig_factors = px.bar(
        factors,
        x="Signal",
        y="Score",
        text="Score",
        title="Location Intelligence Signals"
    )

    st.plotly_chart(
        fig_factors,
        use_container_width=True
    )

    # ========================================================
    # AI-STYLE INSIGHT
    # ========================================================

    st.subheader(
        "🤖 SiteIQ Insight"
    )

    strongest = factors.sort_values(
        "Score",
        ascending=False
    ).iloc[0]

    weakest = factors.sort_values(
        "Score",
        ascending=True
    ).iloc[0]

    st.info(
        f"""
**Recommended candidate for {business_type}.**

The location achieved a **{selected['custom_siteiq_score']:.1f}/100**
SiteIQ score.

**Strongest signal:** {strongest['Signal']}
({strongest['Score']:.1f}/100)

**Weakest signal:** {weakest['Signal']}
({weakest['Score']:.1f}/100)

The location is approximately
**{selected['metro_distance_km']:.2f} km from the nearest Metro station**
and has **{int(selected['traffic_intersections_2km'])} traffic intersections**
within the analysis radius.

This is a data-driven location screening signal,
not a prediction of business success.
"""
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv = top10.to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Export Top Opportunities",
        csv,
        "riyadh_siteiq_opportunities.csv",
        "text/csv",
        use_container_width=True
    )

else:

    st.info(
        "Configure the business priorities on the left, "
        "then click **FIND OPPORTUNITIES**."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Riyadh SiteIQ • Geospatial AI • Machine Learning • "
    "Decision Intelligence"
)
