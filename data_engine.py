import requests
import pandas as pd
import numpy as np


BASE_URL = "https://opendata.rcrc.gov.sa/api/explore/v2.1"
CATALOG_URL = f"{BASE_URL}/catalog/datasets"


DATASET_IDS = {
    "metro": "metro-stations-in-riyadh-by-metro-line-and-station-type-2024",
    "bus": "bus-stops-in-riyadh-by-bus-route-direction-and-shelter-type-2024",
    "traffic": "traffic-intersections-by-main-street-and-cross-street-2024",
    "commercial": "commercial-services-by-category-sub-municipality-and-district-2024",
}


def load_dataset(dataset_id, limit=10000):

    url = f"{CATALOG_URL}/{dataset_id}/records"

    response = requests.get(
        url,
        params={"limit": limit},
        timeout=60
    )

    response.raise_for_status()

    return pd.json_normalize(
        response.json().get("results", [])
    )


def extract_coordinates(df):

    data = df.copy()

    data["latitude"] = np.nan
    data["longitude"] = np.nan

    for column in data.columns:

        name = column.lower()

        if "geo_point_2d" in name:

            for index, value in data[column].items():

                if isinstance(value, dict):

                    data.loc[index, "latitude"] = value.get("lat")
                    data.loc[index, "longitude"] = value.get("lon")

    for column in data.columns:

        name = column.lower()

        if "latitude" in name or name == "lat":

            values = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            data["latitude"] = data["latitude"].fillna(values)

        if "longitude" in name or name == "lon":

            values = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            data["longitude"] = data["longitude"].fillna(values)

    data = data.dropna(
        subset=["latitude", "longitude"]
    )

    data = data[
        (data["latitude"] >= 24.0)
        & (data["latitude"] <= 25.5)
        & (data["longitude"] >= 46.0)
        & (data["longitude"] <= 47.5)
    ]

    return data.reset_index(drop=True)


def load_rcrc_layers():

    layers = {}
    status = []

    for name, dataset_id in DATASET_IDS.items():

        try:

            raw = load_dataset(dataset_id)

            clean = extract_coordinates(raw)

            layers[name] = clean

            status.append(
                {
                    "layer": name,
                    "status": "LOADED",
                    "records": len(clean),
                    "dataset_id": dataset_id
                }
            )

        except Exception as error:

            layers[name] = pd.DataFrame()

            status.append(
                {
                    "layer": name,
                    "status": f"ERROR: {error}",
                    "records": 0,
                    "dataset_id": dataset_id
                }
            )

    return layers, pd.DataFrame(status)


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2
        * earth_radius
        * np.arcsin(np.sqrt(a))
    )


def nearest_distance(
    latitude,
    longitude,
    layer
):

    if layer.empty:
        return np.nan

    distances = haversine_distance(
        latitude,
        longitude,
        layer["latitude"].values,
        layer["longitude"].values
    )

    return float(np.min(distances))


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

    result["metro_distance_km"] = nearest_distance(
        latitude,
        longitude,
        metro
    )

    result["bus_distance_km"] = nearest_distance(
        latitude,
        longitude,
        bus
    )

    if not traffic.empty:

        distances = haversine_distance(
            latitude,
            longitude,
            traffic["latitude"].values,
            traffic["longitude"].values
        )

        result["traffic_intersections_2km"] = int(
            np.sum(distances <= radius_km)
        )

    else:

        result["traffic_intersections_2km"] = 0

    if not commercial.empty:

        distances = haversine_distance(
            latitude,
            longitude,
            commercial["latitude"].values,
            commercial["longitude"].values
        )

        result["commercial_services_2km"] = int(
            np.sum(distances <= radius_km)
        )

    else:

        result["commercial_services_2km"] = 0

    return result


def build_candidate_grid(
    layers,
    lat_min=24.55,
    lat_max=25.00,
    lon_min=46.45,
    lon_max=47.05,
    step=0.02
):

    candidates = []

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

    for latitude in latitudes:

        for longitude in longitudes:

            candidates.append(
                calculate_location_features(
                    latitude,
                    longitude,
                    layers
                )
            )

    return pd.DataFrame(candidates)


def normalize_inverse(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or maximum == minimum:

        return pd.Series(
            50,
            index=series.index
        )

    return (
        100
        * (maximum - series)
        / (maximum - minimum)
    )


def normalize_positive(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or maximum == minimum:

        return pd.Series(
            50,
            index=series.index
        )

    return (
        100
        * (series - minimum)
        / (maximum - minimum)
    )


def calculate_siteiq_score(candidates):

    data = candidates.copy()

    data["accessibility_score"] = normalize_inverse(
        data["metro_distance_km"]
    )

    bus_score = normalize_inverse(
        data["bus_distance_km"]
    )

    data["transit_score"] = (
        data["accessibility_score"]
        + bus_score
    ) / 2

    data["mobility_score"] = normalize_positive(
        data["traffic_intersections_2km"]
    )

    data["competition_gap_score"] = normalize_inverse(
        data["commercial_services_2km"]
    )

    data["siteiq_score"] = (
        data["accessibility_score"] * 0.30
        + data["transit_score"] * 0.20
        + data["mobility_score"] * 0.20
        + data["competition_gap_score"] * 0.30
    )

    data["siteiq_score"] = (
        data["siteiq_score"]
        .clip(0, 100)
        .round(1)
    )

    return data
