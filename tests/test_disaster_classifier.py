import numpy as np
import pytest
from disaster_classifier import (
    InverseFrequencyKNNClassifier,
    evaluate_macro_f1,
    extract_spatiotemporal_features,
    haversine_distance_km,
    load_eonet_dataset,
)


def test_haversine_distance_known_cities():
    # New York (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    dist = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5550.0 <= dist <= 5590.0
    assert haversine_distance_km(0.0, 180.0, 0.0, -180.0) == pytest.approx(0.0, abs=1e-6)


def test_spatiotemporal_features_continuity_across_dateline_and_months():
    recs = [
        {"latitude": 0.0, "longitude": 179.9, "month": 1, "geometry_count": 1, "type": "Point", "category": "Earthquakes"},
        {"latitude": 0.0, "longitude": -179.9, "month": 12, "geometry_count": 1, "type": "Polygon", "category": "Floods"},
    ]
    X, y = extract_spatiotemporal_features(recs)
    assert X.shape == (2, 7)
    assert y == ["Earthquakes", "Floods"]
    # 179.9E and 179.9W are <0.01 radians apart on the 3D unit sphere
    assert np.linalg.norm(X[0, :3] - X[1, :3]) < 0.01


def test_inverse_frequency_knn_handles_imbalance():
    # Majority class 'Wildfires' around (35, -120), minority class 'Volcanoes' around (19.4, -155.3)
    records = [
        {"latitude": 35.0 + 0.01 * i, "longitude": -120.0, "month": 8, "geometry_count": 1, "type": "Point", "category": "Wildfires"}
        for i in range(10)
    ] + [
        {"latitude": 19.4, "longitude": -155.3, "month": 5, "geometry_count": 3, "type": "Point", "category": "Volcanoes"},
        {"latitude": 19.5, "longitude": -155.2, "month": 5, "geometry_count": 4, "type": "Point", "category": "Volcanoes"},
    ]
    X, y = extract_spatiotemporal_features(records)
    clf = InverseFrequencyKNNClassifier(k=3).fit(X, y)
    preds = clf.predict(X)
    probas = clf.predict_proba(X)
    assert len(probas) == len(records)
    assert all(abs(sum(p.values()) - 1.0) < 1e-5 for p in probas)
    metrics = evaluate_macro_f1(y, preds)
    assert metrics["macro_f1"] >= 0.95


def test_load_eonet_dataset_and_classify_real_subset():
    records = load_eonet_dataset(limit=250)
    assert len(records) == 250
    X, y = extract_spatiotemporal_features(records)
    assert X.shape == (250, 7)
    clf = InverseFrequencyKNNClassifier(k=5).fit(X[:200], y[:200])
    preds = clf.predict(X[200:])
    assert len(preds) == 50
    metrics = evaluate_macro_f1(y[200:], preds)
    assert 0.0 <= metrics["accuracy"] <= 1.0

from disaster_classifier.pipeline import compute_class_balance_entropy


def test_compute_class_balance_entropy():
    stats = compute_class_balance_entropy(["Wildfires", "Wildfires", "Severe Storms", "Floods"])
    assert stats["total_samples"] == 4
    assert stats["num_classes"] == 3
    assert stats["imbalance_ratio"] == 2.0
    assert 0.8 < stats["normalized_entropy"] <= 1.0
