import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_engine import (
    load_rcrc_layers,
    build_candidate_grid,
    calculate_siteiq_score
)

st.set_page_config(
    page_title="Riyadh SiteIQ",
    page_icon="📍",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("📍 Riyadh SiteIQ")
st.subheader("AI-Powered Site Selection & Opportunity Intelligence")

st.markdown(
    """
    **Find high-potential locations in Riyadh using geospatial intelligence,
    accessibility, mobility, transit and commercial opportunity signals.**
    """
)

# =========================================================
# DATA
# =========================================================

@st.cache_data(ttl=3600)
def load_data():

    layers = load_rcrc_layers()

    candidates = build_candidate_grid(
        lat_min=24.55,
        lat_max=25.00,
        lon_min=46.45,
        lon_max=47.05,
        step=0.02
    )

    scored = calculate_siteiq_score(
        candidates,
        layers
    )

    return scored, layers


with st.spinner("Loading Riyadh intelligence data..."):
    df, layers = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🎯 SiteIQ Controls")

business_type = st.sidebar.selectbox(
    "Business Type",
    [
        "☕ Café / Restaurant",
        "🛍️ Retail",
        "🏥 Healthcare",
        "🎓 Education",
        "🏢 Office",
        "🚚 Logistics",
        "🌐 General"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚖️ Decision Weights")

accessibility_weight = st.sidebar.slider(
    "Accessibility",
    0.0,
    1.0,
    0.30,
    0.05
)

mobility_weight = st.sidebar.slider(
    "Mobility",
    0.0,
    1.0,
    0.25,
    0.05
)

competition_weight = st.sidebar.slider(
    "Opportunity Gap",
    0.0,
    1.0,
    0.25,
    0.05
)

transit_weight = st.sidebar.slider(
    "Transit",
    0.0,
    1.0,
    0.20,
    0.05
)

weights = np.array([
    accessibility_weight,
    mobility_weight,
    competition_weight,
    transit_weight
])

if weights.sum() == 0:
    weights = np.array([0.25, 0.25, 0.25, 0.25])

weights = weights / weights.sum()

# =========================================================
# CUSTOM SITEIQ SCORE
# =========================================================

df = df.copy()

df["siteiq_custom_score"] = (
    df["accessibility_score"] * weights[0]
    + df["mobility_score"] * weights[1]
    + df["opportunity_gap"] * weights[2]
    + df["transit_score"] * weights[3]
)

df = df.sort_values(
    "siteiq_custom_score",
    ascending=False
).reset_index(drop=True)

df["rank"] = np.arange(1, len(df) + 1)

# =========================================================
# FIND OPPORTUNITIES
# =========================================================

st.markdown("---")

if st.button(
    "🚀 FIND OPPORTUNITIES",
    use_container_width=True
):

    st.session_state["run_search"] = True

if "run_search" not in st.session_state:
    st.session_state["run_search"] = True


# =========================================================
# KPI
# =========================================================

top_location = df.iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎯 Top SiteIQ Score",
        f"{top_location['siteiq_custom_score']:.1f}"
    )

with col2:
    st.metric(
        "📍 Candidate Locations",
        f"{len(df):,}"
    )

with col3:
    st.metric(
        "🚇 Transit Stations",
        f"{len(layers.get('metro', [])):,}"
    )

with col4:
    st.metric(
        "🚌 Bus Stops",
        f"{len(layers.get('bus', [])):,}"
    )


# =========================================================
# TOP 3
# =========================================================

st.markdown("---")
st.header("🏆 Top Opportunities")

top3 = df.head(3)

cols = st.columns(3)

for i, (_, row) in enumerate(top3.iterrows()):

    with cols[i]:

        st.markdown(f"### #{int(row['rank'])}")

        st.metric(
            "SiteIQ Score",
            f"{row['siteiq_custom_score']:.1f}"
        )

        st.write(
            f"📍 **{row['latitude']:.4f}, {row['longitude']:.4f}**"
        )

        st.write(
            f"Accessibility: **{row['accessibility_score']:.1f}**"
        )

        st.write(
            f"Mobility: **{row['mobility_score']:.1f}**"
        )

        st.write(
            f"Opportunity Gap: **{row['opportunity_gap']:.1f}**"
        )

        st.write(
            f"Transit: **{row['transit_score']:.1f}**"
        )


# =========================================================
# OPPORTUNITY MAP
# =========================================================

st.markdown("---")
st.header("🗺️ Opportunity Intelligence Map")

map_df = df.head(100).copy()

fig = px.scatter_mapbox(
    map_df,
    lat="latitude",
    lon="longitude",
    size="siteiq_custom_score",
    color="siteiq_custom_score",
    hover_data={
        "rank": True,
        "siteiq_custom_score": ":.1f",
        "accessibility_score": ":.1f",
        "mobility_score": ":.1f",
        "opportunity_gap": ":.1f",
        "transit_score": ":.1f"
    },
    zoom=10,
    height=650
)

fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# TOP 10
# =========================================================

st.markdown("---")
st.header("📊 Top 10 Candidate Locations")

display_columns = [
    "rank",
    "latitude",
    "longitude",
    "siteiq_custom_score",
    "accessibility_score",
    "mobility_score",
    "opportunity_gap",
    "transit_score"
]

top10 = df.head(10)[display_columns].copy()

top10.columns = [
    "Rank",
    "Latitude",
    "Longitude",
    "SiteIQ Score",
    "Accessibility",
    "Mobility",
    "Opportunity Gap",
    "Transit"
]

st.dataframe(
    top10,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# LOCATION EXPLAINER
# =========================================================

st.markdown("---")
st.header("🔍 Why This Location?")

selected_rank = st.selectbox(
    "Select a location",
    df.head(20)["rank"].tolist(),
    format_func=lambda x: f"#{x}"
)

selected = df[df["rank"] == selected_rank].iloc[0]

st.markdown(
    f"""
    ### 📍 Location #{int(selected["rank"])}

    **Coordinates:** `{selected["latitude"]:.5f}, {selected["longitude"]:.5f}`

    **Business Type:** {business_type}

    **Overall SiteIQ Score:** **{selected["siteiq_custom_score"]:.1f}/100**
    """
)

# =========================================================
# FACTOR ANALYSIS
# =========================================================

factor_df = pd.DataFrame({
    "Factor": [
        "Accessibility",
        "Mobility",
        "Opportunity Gap",
        "Transit"
    ],
    "Score": [
        selected["accessibility_score"],
        selected["mobility_score"],
        selected["opportunity_gap"],
        selected["transit_score"]
    ],
    "Weight": [
        weights[0],
        weights[1],
        weights[2],
        weights[3]
    ]
})

factor_df["Contribution"] = (
    factor_df["Score"] *
    factor_df["Weight"]
)

col1, col2 = st.columns(2)

with col1:

    fig_factor = px.bar(
        factor_df,
        x="Factor",
        y="Score",
        text="Score",
        title="Location Factor Scores"
    )

    fig_factor.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    fig_factor.update_layout(
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig_factor,
        use_container_width=True
    )

with col2:

    st.dataframe(
        factor_df.round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# AI EXPLANATION
# =========================================================

scores = {
    "Accessibility": selected["accessibility_score"],
    "Mobility": selected["mobility_score"],
    "Opportunity Gap": selected["opportunity_gap"],
    "Transit": selected["transit_score"]
}

strongest_factor = max(
    scores,
    key=scores.get
)

weakest_factor = min(
    scores,
    key=scores.get
)

st.info(
    f"""
    🤖 **SiteIQ Insight**

    This location ranks **#{int(selected["rank"])}** for **{business_type}**
    under the current decision preferences.

    Its strongest signal is **{strongest_factor}**
    with a score of **{scores[strongest_factor]:.1f}/100**.

    The weakest signal is **{weakest_factor}**
    with a score of **{scores[weakest_factor]:.1f}/100**.

    **SiteIQ interpretation:** this location is an interesting candidate
    for further commercial validation, especially where accessibility,
    mobility and opportunity-gap signals align.
    """
)


# =========================================================
# COMPARE LOCATIONS
# =========================================================

st.markdown("---")
st.header("⚖️ Compare Locations")

compare_options = df.head(20)["rank"].tolist()

col1, col2 = st.columns(2)

with col1:

    location_a_rank = st.selectbox(
        "Location A",
        compare_options,
        index=0,
        format_func=lambda x: f"Location #{x}",
        key="location_a"
    )

with col2:

    default_b = 1 if len(compare_options) > 1 else 0

    location_b_rank = st.selectbox(
        "Location B",
        compare_options,
        index=default_b,
        format_func=lambda x: f"Location #{x}",
        key="location_b"
    )

location_a = df[
    df["rank"] == location_a_rank
].iloc[0]

location_b = df[
    df["rank"] == location_b_rank
].iloc[0]


comparison = pd.DataFrame({

    "Metric": [
        "SiteIQ Score",
        "Accessibility",
        "Mobility",
        "Opportunity Gap",
        "Transit"
    ],

    "Location A": [
        location_a["siteiq_custom_score"],
        location_a["accessibility_score"],
        location_a["mobility_score"],
        location_a["opportunity_gap"],
        location_a["transit_score"]
    ],

    "Location B": [
        location_b["siteiq_custom_score"],
        location_b["accessibility_score"],
        location_b["mobility_score"],
        location_b["opportunity_gap"],
        location_b["transit_score"]
    ]
})

st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# COMPARISON WINNER
# =========================================================

score_a = location_a["siteiq_custom_score"]
score_b = location_b["siteiq_custom_score"]

if score_a > score_b:

    winner = "Location A"
    difference = score_a - score_b

elif score_b > score_a:

    winner = "Location B"
    difference = score_b - score_a

else:

    winner = "Tie"
    difference = 0


if winner != "Tie":

    st.success(
        f"""
        🏆 **Recommended candidate: {winner}**

        SiteIQ score advantage: **+{difference:.1f} points**

        This recommendation is based on the current business type
        and selected decision weights.
        """
    )

else:

    st.warning(
        "Both locations currently have the same SiteIQ score."
    )


# =========================================================
# DOWNLOAD
# =========================================================

st.markdown("---")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download SiteIQ Opportunities CSV",
    csv,
    "riyadh_siteiq_opportunities.csv",
    "text/csv",
    use_container_width=True
)


# =========================================================
# DISCLAIMER
# =========================================================

st.caption(
    """
    SiteIQ is an exploratory location-intelligence decision-support prototype.
    Scores indicate relative opportunity signals and do not guarantee
    commercial success. Real-world decisions should incorporate verified
    commercial, demographic, rental, regulatory and market data.
    """
)
