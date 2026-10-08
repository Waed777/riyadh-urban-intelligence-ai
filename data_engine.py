import requests
import pandas as pd
import numpy as np


# ============================================================
# RCRC DATA ENGINE
# Riyadh SiteIQ
# ============================================================

BASE_URL = "https://opendata.rcrc.gov.sa/api/explore/v2.1"

CATALOG_URL = f"{BASE_URL}/catalog/datasets"


# ============================================================
# DATASET DISCOVERY
# ============================================================

DATASET_TARGETS = {
    "metro": [
        "metro stations",
        "metro station",
        "riyadh metro"
    ],
    "bus": [
        "bus stops",
        "bus stations",
        "bus line stations",
        "riyadh bus"
    ],
    "traffic": [
        "traffic intersections",
        "intersections",
        "traffic"
    ],
    "commercial": [
        "commercial services",
        "commercial service"
    ]
}


def get_catalog(limit=100):

    response = requests.get(
        CATALOG_URL,
        params={
            "limit": limit
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def discover_dataset(target_name):

    catalog = get_catalog()

    datasets = catalog.get("datasets", [])

    keywords = DATASET_TARGETS[target_name]

    candidates = []

    for item in datasets:

        dataset = item.get("dataset", {})

        dataset_id = dataset.get("dataset_id", "")
        title = dataset.get("metas", {}).get("default", {}).get(
            "title",
            ""
        )

        description = dataset.get("metas", {}).get(
            "default",
            {}
        ).get(
            "description",
            ""
        )

        text = (
            f"{dataset_id} "
            f"{title} "
            f"{description}"
        ).lower()

        score = 0

        for keyword in keywords:

            if keyword.lower() in text:

                score += 1

        if score > 0:

            candidates.append(
                {
                    "dataset_id": dataset_id,
                    "title": title,
                    "description": description,
                    "score": score
                }
            )

    if not candidates:

        return None

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    dataset_id,
    limit=10000
):

    url = (
        f"{CATALOG_URL}/"
        f"{dataset_id}/records"
    )

    response = requests.get(
        url,
        params={
            "limit": limit
        },
        timeout=60
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        []
    )

    return pd.json_normalize(results)


# ============================================================
# GEO EXTRACTION
# ============================================================

def extract_coordinates(df):

    data = df.copy()

    latitude = []
    longitude = []

    for _, row in data.iterrows():

        lat = None
        lon = None

        # ----------------------------------------------------
        # geo_point_2d
        # ----------------------------------------------------

        for column in data.columns:

            column_lower = column.lower()

            if "geo_point_2d" in column_lower:

                value = row[column]

                if isinstance(value, dict):

                    lat = value.get("lat")
                    lon = value.get("lon")

                elif isinstance(value, str):

                    try:

                        import json

                        parsed = json.loads(value)

                        lat = parsed.get("lat")
                        lon = parsed.get("lon")

                    except Exception:

                        pass

        # ----------------------------------------------------
        # Search latitude / longitude columns
        # ----------------------------------------------------

        if lat is None or lon is None:

            for column in data.columns:

                name = column.lower()

                if (
                    "latitude" in name
                    or name.endswith("_lat")
                    or name == "lat"
                ):

                    try:
                        lat = float(row[column])
                    except Exception:
                        pass

                if (
                    "longitude" in name
                    or name.endswith("_lon")
                    or name == "lon"
                ):

                    try:
                        lon = float(row[column])
                    except Exception:
                        pass

        # ----------------------------------------------------
        # GeoJSON
        # ----------------------------------------------------

        if lat is None or lon is None:

            for column in data.columns:

                if "geo_shape" not in column.lower():

                    continue

                value = row[column]

                if isinstance(value, dict):

                    geometry = value.get(
                        "geometry",
                        value
                    )

                    coordinates = geometry.get(
                        "coordinates"
                    )

                    if (
                        isinstance(coordinates, list)
                        and len(coordinates) >= 2
                    ):

                        try:

                            lon = float(
                                coordinates[0]
                            )

                            lat = float(
                                coordinates[1]
                            )

                        except Exception:

                            pass

        latitude.append(lat)
        longitude.append(lon)

    data["latitude"] = latitude
    data["longitude"] = longitude

    data["latitude"] = pd.to_numeric(
        data["latitude"],
        errors="coerce"
    )

    data["longitude"] = pd.to_numeric(
        data["longitude"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    # Riyadh geographic sanity check

    data = data[
        (data["latitude"] >= 24.0)
        &
        (data["latitude"] <= 25.5)
        &
        (data["longitude"] >= 46.0)
        &
        (data["longitude"] <= 47.5)
    ]

    return data.reset_index(
        drop=True
    )


# ============================================================
# LOAD RCRC LAYERS
# ============================================================

def load_rcrc_layers():

    layers = {}

    discovery_log = []

    for target in [
        "metro",
        "bus",
        "traffic",
        "commercial"
    ]:

        try:

            dataset = discover_dataset(
                target
            )

            if dataset is None:

                discovery_log.append(
                    {
                        "layer": target,
                        "status": "NOT FOUND",
                        "dataset_id": None
                    }
                )

                continue

            dataset_id = dataset[
                "dataset_id"
            ]

            raw = load_dataset(
                dataset_id
            )

            clean = extract_coordinates(
                raw
            )

            layers[target] = clean

            discovery_log.append(
                {
                    "layer": target,
                    "status": "LOADED",
                    "dataset_id": dataset_id,
                    "records": len(clean),
                    "title": dataset["title"]
                }
            )

        except Exception as e:

            discovery_log.append(
                {
                    "layer": target,
                    "status": f"ERROR: {str(e)}",
                    "dataset_id": None
                }
            )

    return layers, pd.DataFrame(
        discovery_log
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(
        lat2 - lat1
    )

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# NEAREST FEATURE
# ============================================================

def nearest_distance(
    latitude,
    longitude,
    layer
):

    if layer is None or layer.empty:

        return np.nan

    distances = haversine_distance(
        latitude,
        longitude,
        layer["latitude"].values,
        layer["longitude"].values
    )

    return float(
        np.min(distances)
    )


# ============================================================
# CANDIDATE FEATURES
# ============================================================

def calculate_location_features(
    latitude,
    longitude,
    layers,
    radius_km=2
):

    result = {
        "latitude": latitude,
        "longitude": longitude
    }

    # --------------------------------------------------------
    # Metro
    # --------------------------------------------------------

    metro = layers.get(
        "metro"
    )

    if metro is not None and not metro.empty:

        result["metro_distance_km"] = (
            nearest_distance(
                latitude,
                longitude,
                metro
            )
        )

    else:

        result["metro_distance_km"] = np.nan

    # --------------------------------------------------------
    # Bus
    # --------------------------------------------------------

    bus = layers.get(
        "bus"
    )

    if bus is not None and not bus.empty:

        result["bus_distance_km"] = (
            nearest_distance(
                latitude,
                longitude,
                bus
            )
        )

    else:

        result["bus_distance_km"] = np.nan

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    traffic = layers.get(
        "traffic"
    )

    if traffic is not None and not traffic.empty:

        distances = haversine_distance(
            latitude,
            longitude,
            traffic["latitude"].values,
            traffic["longitude"].values
        )

        result["traffic_intersections_2km"] = int(
            np.sum(
                distances <= radius_km
            )
        )

    else:

        result["traffic_intersections_2km"] = 0

    # --------------------------------------------------------
    # Commercial Services
    # --------------------------------------------------------

    commercial = layers.get(
        "commercial"
    )

    if (
        commercial is not None
        and not commercial.empty
    ):

        distances = haversine_distance(
            latitude,
            longitude,
            commercial["latitude"].values,
            commercial["longitude"].values
        )

        nearby = commercial[
            distances <= radius_km
        ]

        result[
            "commercial_services_2km"
        ] = len(nearby)

    else:

        result[
            "commercial_services_2km"
        ] = 0

    return result


# ============================================================
# BUILD CANDIDATE GRID
# ============================================================

def build_candidate_grid(
    layers,
    lat_min=24.55,
    lat_max=25.00,
    lon_min=46.45,
    lon_max=47.05,
    step=0.01
):

    latitudes = np.arange(
        lat_min,
        lat_max,
        step
    )

    longitudes = np.arange(
        lon_min,
        lon_max,
        step
    )

    candidates = []

    for lat in latitudes:

        for lon in longitudes:

            features = calculate_location_features(
                lat,
                lon,
                layers
            )

            candidates.append(
                features
            )

    return pd.DataFrame(
        candidates
    )


# ============================================================
# NORMALIZE FEATURE
# ============================================================

def normalize_inverse(
    series
):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum == minimum
    ):

        return pd.Series(
            50,
            index=series.index
        )

    return (
        100
        * (
            maximum - series
        )
        / (
            maximum - minimum
        )
    )


def normalize_positive(
    series
):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum == minimum
    ):

        return pd.Series(
            50,
            index=series.index
        )

    return (
        100
        * (
            series - minimum
        )
        / (
            maximum - minimum
        )
    )


# ============================================================
# SITEIQ OPPORTUNITY SCORE
# ============================================================

def calculate_siteiq_score(
    candidates
):

    data = candidates.copy()

    # Accessibility

    data["accessibility_score"] = (
        normalize_inverse(
            data["metro_distance_km"]
        )
    )

    # Public transport

    data["transit_score"] = (
        (
            data["accessibility_score"]
            +
            normalize_inverse(
                data["bus_distance_km"]
            )
        )
        / 2
    )

    # Road / traffic connectivity

    data["mobility_score"] = (
        normalize_positive(
            data[
                "traffic_intersections_2km"
            ]
        )
    )

    # Commercial activity

    data["commercial_activity_score"] = (
        normalize_positive(
            data[
                "commercial_services_2km"
            ]
        )
    )

    # Competition gap
    #
    # Fewer commercial services = larger gap.
    # This is NOT a business-success prediction.

    data["competition_gap_score"] = (
        normalize_inverse(
            data[
                "commercial_services_2km"
            ]
        )
    )

    # --------------------------------------------------------
    # FINAL SITEIQ SCORE
    # --------------------------------------------------------

    data["siteiq_score"] = (
        data["accessibility_score"] * 0.30
        +
        data["transit_score"] * 0.20
        +
        data["mobility_score"] * 0.20
        +
        data["competition_gap_score"] * 0.30
    )

    data["siteiq_score"] = (
        data["siteiq_score"]
        .clip(0, 100)
        .round(1)
    )

    return data


# ============================================================
# TOP OPPORTUNITIES
# ============================================================

def rank_opportunities(
    candidates,
    top_n=10
):

    data = calculate_siteiq_score(
        candidates
    )

    return (
        data
        .sort_values(
            "siteiq_score",
            ascending=False
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# ONE-CALL PIPELINE
# ============================================================

def run_siteiq_pipeline():

    layers, discovery = (
        load_rcrc_layers()
    )

    candidates = build_candidate_grid(
        layers
    )

    ranked = rank_opportunities(
        candidates,
        top_n=10
    )

    return {
        "layers": layers,
        "discovery": discovery,
        "candidates": candidates,
        "top_opportunities": ranked
    }
  if __name__ == "__main__":

    result = run_siteiq_pipeline()

    print("\n==============================")
    print("RCRC DATA DISCOVERY")
    print("==============================")

    print(
        result["discovery"].to_string(
            index=False
        )
    )

    print("\n==============================")
    print("TOP SITEIQ OPPORTUNITIES")
    print("==============================")

    print(
        result[
            "top_opportunities"
        ].to_string(
            index=False
        )
    )
