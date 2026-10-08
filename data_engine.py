import requests
import pandas as pd
import numpy as np


# =========================================================
# RCRC API
# =========================================================

BASE_URL = "https://opendata.rcrc.gov.sa/api/explore/v2.1/catalog/datasets"

DATASET_IDS = {
    "metro": "metro-stations-in-riyadh-by-metro-line-and-station-type-2024",
    "bus": "bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024",
    "traffic": "traffic-intersections-by-main-street-and-cross-street-2024",
    "commercial": "commercial-services-by-category-sub-municipality-and-district-2024"
}


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(dataset_id, limit=10000):

    url = f"{BASE_URL}/{dataset_id}/records"

    params = {
        "limit": limit
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        records = data.get("results", [])

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    except Exception as e:

        print(f"Dataset error: {dataset_id}")
        print(e)

        return pd.DataFrame()


# =========================================================
# EXTRACT COORDINATES
# =========================================================

def extract_coordinates(df):

    if df.empty:
        return df

    data = df.copy()

    # -----------------------------------------------------
    # geo_point_2d
    # -----------------------------------------------------

    if "geo_point_2d" in data.columns:

        def get_lat(value):

            try:

                if isinstance(value, dict):
                    return float(value.get("lat"))

                if isinstance(value, (list, tuple)):
                    return float(value[0])

                if isinstance(value, str):

                    value = value.replace("(", "")
                    value = value.replace(")", "")
                    value = value.replace("[", "")
                    value = value.replace("]", "")

                    parts = value.split(",")

                    if len(parts) >= 2:
                        return float(parts[0].strip())

            except Exception:
                pass

            return np.nan


        def get_lon(value):

            try:

                if isinstance(value, dict):
                    return float(value.get("lon"))

                if isinstance(value, (list, tuple)):
                    return float(value[1])

                if isinstance(value, str):

                    value = value.replace("(", "")
                    value = value.replace(")", "")
                    value = value.replace("[", "")
                    value = value.replace("]", "")

                    parts = value.split(",")

                    if len(parts) >= 2:
                        return float(parts[1].strip())

            except Exception:
                pass

            return np.nan


        data["latitude"] = data["geo_point_2d"].apply(
            get_lat
        )

        data["longitude"] = data["geo_point_2d"].apply(
            get_lon
        )

    # -----------------------------------------------------
    # latitude / longitude
    # -----------------------------------------------------

    if "latitude" not in data.columns:

        for col in [
            "lat",
            "Latitude",
            "LATITUDE"
        ]:

            if col in data.columns:

                data["latitude"] = pd.to_numeric(
                    data[col],
                    errors="coerce"
                )

                break


    if "longitude" not in data.columns:

        for col in [
            "lon",
            "lng",
            "Longitude",
            "LONGITUDE"
        ]:

            if col in data.columns:

                data["longitude"] = pd.to_numeric(
                    data[col],
                    errors="coerce"
                )

                break


    if "latitude" not in data.columns:
        data["latitude"] = np.nan

    if "longitude" not in data.columns:
        data["longitude"] = np.nan


    data["latitude"] = pd.to_numeric(
        data["latitude"],
        errors="coerce"
    )

    data["longitude"] = pd.to_numeric(
        data["longitude"],
        errors="coerce"
    )


    # Riyadh geographic filter

    data = data[
        data["latitude"].between(24.0, 25.5)
        &
        data["longitude"].between(46.0, 47.5)
    ].copy()


    return data


# =========================================================
# LOAD ALL RCRC LAYERS
# =========================================================

def load_rcrc_layers():

    layers = {}

    for layer_name, dataset_id in DATASET_IDS.items():

        df = load_dataset(dataset_id)

        df = extract_coordinates(df)

        layers[layer_name] = df

    return layers


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return 6371 * c


# =========================================================
# NEAREST DISTANCE
# =========================================================

def nearest_distance(
    lat,
    lon,
    points
):

    if points is None or points.empty:
        return 999.0

    if (
        "latitude" not in points.columns
        or
        "longitude" not in points.columns
    ):
        return 999.0

    points = points[
        points["latitude"].notna()
        &
        points["longitude"].notna()
    ]

    if points.empty:
        return 999.0

    distances = haversine_distance(
        lat,
        lon,
        points["latitude"].values,
        points["longitude"].values
    )

    return float(
        np.nanmin(distances)
    )


# =========================================================
# LOCATION FEATURES
# =========================================================

def calculate_location_features(
    candidates,
    layers
):

    data = candidates.copy()

    metro = layers.get(
        "metro",
        pd.DataFrame()
    )

    bus = layers.get(
        "bus",
        pd.DataFrame()
    )

    traffic = layers.get(
        "traffic",
        pd.DataFrame()
    )

    commercial = layers.get(
        "commercial",
        pd.DataFrame()
    )


    metro_distances = []
    bus_distances = []
    traffic_distances = []
    commercial_distances = []


    for _, row in data.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]


        metro_distances.append(
            nearest_distance(
                lat,
                lon,
                metro
            )
        )


        bus_distances.append(
            nearest_distance(
                lat,
                lon,
                bus
            )
        )


        traffic_distances.append(
            nearest_distance(
                lat,
                lon,
                traffic
            )
        )


        commercial_distances.append(
            nearest_distance(
                lat,
                lon,
                commercial
            )
        )


    data["distance_to_metro_km"] = (
        metro_distances
    )

    data["distance_to_bus_km"] = (
        bus_distances
    )

    data["distance_to_traffic_km"] = (
        traffic_distances
    )

    data["distance_to_commercial_km"] = (
        commercial_distances
    )


    return data


# =========================================================
# BUILD RIYADH CANDIDATE GRID
# =========================================================

def build_candidate_grid(
    lat_min=24.55,
    lat_max=25.00,
    lon_min=46.45,
    lon_max=47.05,
    step=0.02
):

    lats = np.arange(
        lat_min,
        lat_max + step,
        step
    )

    lons = np.arange(
        lon_min,
        lon_max + step,
        step
    )


    grid = []

    for lat in lats:

        for lon in lons:

            grid.append(
                {
                    "latitude": float(lat),
                    "longitude": float(lon)
                }
            )


    return pd.DataFrame(grid)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_positive(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        (series - min_value)
        /
        (max_value - min_value)
        *
        100
    )


def normalize_inverse(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(999)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        1
        -
        (
            (series - min_value)
            /
            (max_value - min_value)
        )
    ) * 100


# =========================================================
# SITEIQ SCORING
# =========================================================

def calculate_siteiq_score(
    candidates,
    layers
):

    data = calculate_location_features(
        candidates,
        layers
    )


    # -----------------------------------------------------
    # Accessibility
    # -----------------------------------------------------

    data["metro_accessibility"] = (
        normalize_inverse(
            data["distance_to_metro_km"]
        )
    )


    data["bus_accessibility"] = (
        normalize_inverse(
            data["distance_to_bus_km"]
        )
    )


    data["accessibility_score"] = (
        data["metro_accessibility"] * 0.6
        +
        data["bus_accessibility"] * 0.4
    )


    # -----------------------------------------------------
    # Mobility
    # -----------------------------------------------------

    data["mobility_score"] = (
        normalize_inverse(
            data["distance_to_traffic_km"]
        )
    )


    # -----------------------------------------------------
    # Transit
    # -----------------------------------------------------

    data["transit_score"] = (
        data["metro_accessibility"] * 0.7
        +
        data["bus_accessibility"] * 0.3
    )


    # -----------------------------------------------------
    # Commercial Opportunity Gap
    # -----------------------------------------------------

    data["commercial_access"] = (
        normalize_inverse(
            data["distance_to_commercial_km"]
        )
    )


    data["opportunity_gap"] = (
        100
        -
        data["commercial_access"]
    )


    # -----------------------------------------------------
    # Base SiteIQ Score
    # -----------------------------------------------------

    data["siteiq_score"] = (
        data["accessibility_score"] * 0.30
        +
        data["mobility_score"] * 0.25
        +
        data["opportunity_gap"] * 0.25
        +
        data["transit_score"] * 0.20
    )


    # -----------------------------------------------------
    # Safety / cleanup
    # -----------------------------------------------------

    score_columns = [
        "accessibility_score",
        "mobility_score",
        "transit_score",
        "opportunity_gap",
        "siteiq_score"
    ]


    for col in score_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        ).fillna(0)

        data[col] = data[col].clip(
            0,
            100
        )


    return data
