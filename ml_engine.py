import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor


def prepare_features(df):

    data = df.copy()

    # Geographic accessibility
    data["metro_proximity"] = 1 / (
        1 + data["distance_from_riyadh_center_km"]
    )

    # Normalize the existing accessibility signal
    scaler = StandardScaler()

    data["accessibility_normalized"] = scaler.fit_transform(
        data[["accessibility_score"]]
    )

    # Geographic density proxy
    data["urban_access_signal"] = (
        data["metro_proximity"] * 100
    )

    return data


def segment_zones(df, n_clusters=4):

    data = prepare_features(df)

    features = data[
        [
            "accessibility_score",
            "distance_from_riyadh_center_km",
            "urban_access_signal"
        ]
    ].fillna(0)

    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    data["ml_cluster"] = model.fit_predict(X)

    # Rank clusters by average accessibility
    cluster_scores = (
        data.groupby("ml_cluster")[
            "accessibility_score"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    labels = {}

    for rank, cluster in enumerate(
        cluster_scores.index
    ):

        if rank == 0:
            labels[cluster] = "Prime Accessibility Zone"

        elif rank == 1:
            labels[cluster] = "Strong Accessibility Zone"

        elif rank == 2:
            labels[cluster] = "Emerging Accessibility Zone"

        else:
            labels[cluster] = "Lower Accessibility Zone"

    data["ai_zone_segment"] = data[
        "ml_cluster"
    ].map(labels)

    return data, model


def train_explainability_model(df):

    data = prepare_features(df)

    features = [
        "distance_from_riyadh_center_km",
        "urban_access_signal",
        "accessibility_normalized"
    ]

    X = data[features].fillna(0)

    y = data["accessibility_score"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    return model, importance
