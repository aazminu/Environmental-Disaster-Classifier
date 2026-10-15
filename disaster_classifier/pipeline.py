from collections import Counter
import csv
from datetime import datetime
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two (lat, lon) coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def load_eonet_dataset(
    csv_path: str = "all_disasters_10_years.csv",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load and validate NASA EONET environmental disaster records from CSV."""
    records: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
                geom_count = int(float(row.get("geometry_count", 1) or 1))
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(lon) and math.isfinite(lat)):
                continue
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue

            raw_date = (row.get("date") or "").strip()
            month = 1
            if raw_date:
                try:
                    dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    month = dt.month
                except ValueError:
                    month = 1

            category = (row.get("category") or "").strip()
            if not category:
                continue

            records.append(
                {
                    "id": (row.get("id") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "category": category,
                    "month": month,
                    "type": (row.get("type") or "Point").strip(),
                    "longitude": lon,
                    "latitude": lat,
                    "geometry_count": max(1, geom_count),
                }
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def extract_spatiotemporal_features(records: Iterable[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert disaster records into an 7-dimensional feature matrix:
      [x_unit, y_unit, z_unit, sin_month, cos_month, log_geom_count, is_polygon]
    and a list of category labels.
    """
    features: List[List[float]] = []
    labels: List[str] = []
    for rec in records:
        lat_rad = math.radians(float(rec["latitude"]))
        lon_rad = math.radians(float(rec["longitude"]))
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)

        month = int(rec.get("month", 1))
        angle = 2.0 * math.pi * ((month - 1) % 12) / 12.0
        sin_m = math.sin(angle)
        cos_m = math.cos(angle)

        geom_count = max(1, int(rec.get("geometry_count", 1)))
        log_geom = math.log1p(geom_count)
        is_poly = 1.0 if str(rec.get("type", "")).lower() == "polygon" else 0.0

        features.append([x, y, z, sin_m, cos_m, log_geom, is_poly])
        labels.append(str(rec["category"]))

    if not features:
        return np.zeros((0, 7), dtype=np.float64), []
    return np.asarray(features, dtype=np.float64), labels


class InverseFrequencyKNNClassifier:
    """
    k-Nearest Neighbors classifier weighted by both inverse Euclidean distance
    and inverse class frequency to mitigate severe category imbalance.
    """

    def __init__(self, k: int = 5, eps: float = 1e-6):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.eps = eps
        self.X_train: Optional[np.ndarray] = None
        self.y_train: List[str] = []
        self.class_weights: Dict[str, float] = {}
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "InverseFrequencyKNNClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if len(y) != X_arr.shape[0]:
            raise ValueError("X and y must have the same length")

        self.mean = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        self.std = np.where(std > 1e-9, std, 1.0)
        self.X_train = (X_arr - self.mean) / self.std
        self.y_train = [str(label) for label in y]

        counts = Counter(self.y_train)
        total = float(len(self.y_train))
        num_classes = float(len(counts))
        self.class_weights = {
            cls: total / (num_classes * count) for cls, count in counts.items()
        }
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        if self.X_train is None or self.mean is None or self.std is None:
            raise RuntimeError("Classifier must be fitted before calling predict")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[0] == 0:
            return []

        X_norm = (X_arr - self.mean) / self.std
        k_eff = min(self.k, self.X_train.shape[0])
        preds: List[str] = []

        for row in X_norm:
            dists = np.linalg.norm(self.X_train - row, axis=1)
            nearest_idx = np.argsort(dists)[:k_eff]
            scores: Dict[str, float] = {}
            for idx in nearest_idx:
                cls = self.y_train[int(idx)]
                inv_dist = 1.0 / (float(dists[int(idx)]) + self.eps)
                weight = inv_dist * self.class_weights.get(cls, 1.0)
                scores[cls] = scores.get(cls, 0.0) + weight
            best_cls = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
            preds.append(best_cls)
        return preds


def evaluate_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    """Compute accuracy and macro-averaged precision, recall, and F1 score."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and of equal length")

    classes = sorted(set(y_true) | set(y_pred))
    precisions, recalls, f1s = [], [], []
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)

    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": round(correct / float(len(y_true)), 6),
        "macro_precision": round(float(np.mean(precisions)), 6),
        "macro_recall": round(float(np.mean(recalls)), 6),
        "macro_f1": round(float(np.mean(f1s)), 6),
    }
