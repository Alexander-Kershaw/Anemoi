# Zephyrus — Machine Learning Post-Processing Layer

Zephyrus provides probabilistic, machine-learning–based risk guidance on top of physically derived diagnostics directly from Notus and event definitions from Boreas.

The objective of Zephyrus is not to replace convensional physical reasoning, but to:
- learn smooth probabilistic mappings from physically meaningful predictors
- quantify uncertainty
- provide threshold-tunable guidance for operational decision-making



## Design Principles

- **Physics-aware features**: Inputs are derived from meteorological variables, not from arbitrary transformations.
- **Time-aware validation**: All models use chronological train/test (time partitioned) splits to avoid information leakage.
- **Baseline comparison**: All models are evaluated against climatology and persistence.
- **Probabilistic output**: Models predict event probabilities, not just binary outcomes.
- **Honest evaluation**: Rare-event limitations are explicitly documented (Rain model in particular).



## Data Source

All models use curated hourly observational data produced by the Notus ingestion and diagnostics pipeline:

- Hourly surface observations (temperature, humidity, pressure, wind, precipitation)
- Derived physical diagnostics (e.g. dewpoint depression, pressure tendency)
- Diagnostic flags (fog risk, high wind, heavy rain)



## Fog Risk Prediction

### Target
- `fog_risk_v2` (binary)
- Defined using physical diagnostics (near saturation, light winds, nighttime, persistence)

### Features
Physically motivated predictors including:
- temperature
- relative humidity
- pressure and pressure tendency
- dew point depression
- moisture variables
- wind speed/direction
- cloud cover and sunshine duration

### Model
- Logistic Regression (baseline, interpretable)
- Class-balanced loss
- Median imputation + standard scaling
- Time-aware 80/20 train/test split

### Results
- **ROC AUC ≈ 0.96**
- **Average Precision ≈ 0.78**
- Superior performance relative to:
  - climatology
  - persistence baseline

The model achieves high recall for fog events, with tunable thresholds allowing trade-offs between false alarms and missed events.

### Interpretation
The model learns precursors to fog formation, not just persistence, including:
- decreasing dewpoint depression
- weakening winds
- pressure tendencies


## High-Wind Risk Prediction

### Target
- `high_wind_v1`
- Defined using sustained wind speed and/or gust thresholds (via Boreas logic)

### Important Note on Leakage
An initial model using wind-speed predictors achieved unrealistically high skill due to label leakage.  
This was corrected by removing diagnostic wind variables from the feature set resulting in an exclusively wind indicator feature pool.

### Features (final model)
- surface pressure
- pressure tendency
- temperature
- humidity
- cloud cover
- precipitation indicators

### Results (leakage-free)
- **ROC AUC ≈ 0.94**
- **Average Precision ≈ 0.99**
- Skill comparable to persistence, with added value in identifying non-windy regimes
- Threshold-tunable to balance recall vs false alarms

### Interpretation
Persistence is a strong baseline for wind regimes.  
Zephyrus adds incremental synoptic context, improving probabilistic guidance beyond simple persistence.



## Heavy-Rain Risk

### Target
- `heavy_rain_v1` (≥ 2 mm/hr, persistent)

### Outcome
During the evaluation window, no heavy-rain events occurred in the test period.  
As a result:
- ROC AUC and Average Precision are undefined
- Persistence and climatology baselines are also non-informative

This is likely caused by the meteorological climate of London in the Autumn timeframe generally having less heavly precipitory episodes, there simply wasn't enough heavy rain events to generate any substancial predictions

### Interpretation
This reflects a realistic rare-event problem, common in:
- flood forecasting
- extreme precipitation modelling

### Design Decision
Heavy rainfall risk is currently handled via event-based diagnostics in Boreas, rather than hourly ML classification.

This avoids misleading metrics and preserves scientific integrity.



## Model Outputs

Saved artifacts include:
- trained model files (`.joblib`)
- ROC and Precision–Recall curves
- calibration plots
- probability time series
- threshold-performance tables

These outputs support:
- operational threshold selection
- risk-based decision-making
- transparent model evaluation

-

## Relationship to Other Modules

- **Notus**: physical diagnostics and curated datasets
- **Boreas**: event detection, severity scoring, hazard catalogues
- **Eurus**: visualization and exploratory analysis

Zephyrus sits between diagnostics and decision support, providing probabilistic interpretation.



## Future Work

- Temperature based model
- Lead-time prediction (e.g. t → t+3h)
- Gradient-boosted models for comparison
- Event-based cross-validation for rare precipitation events
- Multi-hazard joint risk modelling



## Summary

Zephyrus demonstrates how physically informed machine learning can:
- outperform naive baselines
- quantify uncertainty
- complement diagnostic meteorology
- support operational weather-risk decisions
