# Environmental Disaster Classifier

Geospatial and temporal machine learning pipeline for classifying US natural disasters (`Flood`, `Severe Storm`, `Hurricane`, `Fire`, `Tornado`, `Snowstorm`, etc.) using 10 years of FEMA disaster declaration records (`all_disasters_10_years.csv`) merged with NOAA station temperature and precipitation histories.

## Overview
- **Dataset (`all_disasters_10_years.csv`)**: 10-year historical record of US county/state disaster declarations with latitude, longitude, elevation, temporal timestamps (`year`, `month`), and meteorological observations (`avg_temp`, `min_temp`, `max_temp`, `precipitation`).
- **Feature Engineering (`disaster_classifier/pipeline.py`)**:
  - Cyclical month encoding ($\sin(2\pi m/12), \cos(2\pi m/12)$) to preserve seasonal continuity across December–January boundaries.
  - Diurnal temperature range (`temp_spread = max_temp - min_temp`) and standardized spatial/meteorological features.
  - Great-circle Haversine distance calculations for spatial proximity queries.
- **Classification & Evaluation**:
  - Inverse-distance and inverse-class-frequency weighted $k$-Nearest Neighbors classifier designed to handle class imbalance across disaster categories.
  - Macro-F1, per-class precision/recall, and stratified cross-validation evaluation.

## Running Tests
```bash
pytest tests/
```
