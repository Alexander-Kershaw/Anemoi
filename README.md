# Anemoi — End-to-End Weather Analytics & Decision Support

Anemoi is an end-to-end weather analytics project designed to mirror how meteorological data products are built in industry and government settings.

It combines:
- physical diagnostics,
- event-based hazard analysis,
- probabilistic machine learning,
- and decision-support visualization

into a single, coherent pipeline.

The project emphasizes scientific validity, interpretability, and operational realism over purely academic performance.


## Project Structure

```text
anemoi/
├── notus_lab/ # Data ingestion & physical diagnostics
├── boreas_lab/ # Event detection & severity analysis
├── zephyrus_lab/ # ML-based probabilistic post-processing
├── eurus_lab/ # Visualization & decision-support dashboards
└── src/ # Shared Python packages
```

Each module is named after one of the Anemoi (wind deities of greek mythology), each representing key layers of the pipeline.

---

## Notus — Data & Diagnostics

**Notus** ingests hourly surface observations and derives physically meaningful diagnostics, including:

- dewpoint and moisture variables
- pressure tendencies
- wind vector components
- precipitation indicators
- fog-relevant thermodynamic metrics

Outputs are written as partitioned Parquet datasets, forming the single source of truth for downstream modules.

---

## Boreas — Event & Hazard Analysis

**Boreas** detects and characterizes weather events using physically motivated rules:

- Fog / low-cloud events
- High-wind events
- Heavy-rain events

For each event, Boreas computes:
- duration
- intensity / accumulation
- severity scores
- summary statistics

This enables event-level reasoning, rather than relying only on hourly thresholds.

---

## Zephyrus — Probabilistic Machine Learning

**Zephyrus** applies machine learning to estimate **hourly hazard probabilities** using diagnostics from Notus and labels from Boreas.

Key characteristics:
- time-aware train/test splits
- explicit baselines (climatology, persistence)
- leakage-aware feature selection
- calibrated probabilistic outputs
- threshold-tunable decision support

Implemented hazards:
- Fog risk (strong improvement over persistence)
- High-wind risk (incremental synoptic-scale skill)
- Precipitation risk handled diagnostically due to rarity

---

## Eurus — Visualization & Decision Support

**Eurus** unifies diagnostics, events, and probabilities into clear dashboards designed for situational awareness and retrospective analysis.

Dashboards include:
1. **Hourly Risk Timeline** — probabilistic hazards with event overlays
2. **Event Severity Summary** — duration vs severity across hazards
3. **Model Performance Summary** — ROC, PR, calibration

The emphasis is on clarity and interpretability.

---

## Design Philosophy

- Physics-informed before ML
- Probabilities over binary flags
- Careful treatment of uncertainty
- Realistic handling of rare events
- Modular, extensible architecture

---

## Current Status (v1)

- End-to-end pipeline complete
- Fog, wind, and rain hazards implemented
- ML validated against strong baselines
- Visualization layer complete and coherent

---

## Future Work

Planned extensions include:
- Interactive dashboards (Plotly)
- Temperature-based hazards (heat/cold stress)
- Lead-time forecasting (t → t+3h)
- Multi-hazard joint risk analysis
- Expanded geographic coverage


***