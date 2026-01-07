# Eurus — Visualization & Decision-Support Layer

Eurus is the visualization and situational-awareness layer of the Anemoi project.  

It unifies diagnostics and ingestion from Notus, event severity catalogues from Boreas, and probabilistic outputs from machine learning models in Zephyrus into interpretable dashboards.

The goal of Eurus is not exploratory analysis, but the communication of the following questions:
- What is happening now? (diagnostics)
- What events mattered most? (severity)
- How trustworthy are the models? (evaluation)


## Design Principles

- **End-to-end coherence**: Diagnostics → Events → Probabilities → Visualization
- **Minimal but meaningful visuals**: No unnecessary interactivity or clutter
- **Operational realism**: Time-aware views, threshold-aware interpretation
- **Honest uncertainty**: Rare-event limitations are shown, not hidden (as seen with rain model performance)


## Dashboards

### Hourly Risk Timeline

**File**: `eurus_lab/dashboards/hourly_risk_timeline.py`  
**Output**: `eurus_lab/assets/hourly_risk_timeline_v2.png`

**Purpose**  
Situational awareness over recent days.

**What it shows**
- Fog probability (Zephyrus)
- High-wind probability (Zephyrus)
- Nighttime context
- Fog and wind events (Boreas), with shading intensity proportional to event severity

**Why it matters**
- Demonstrates that probabilistic risk rises before and during events
- Shows alignment between physical diagnostics, ML output, and observed impacts


### Event Severity Summary

**File**: `eurus_lab/dashboards/event_severity_summary.py`  
**Output**: `eurus_lab/assets/event_severity_summary.png`

**Purpose**  
Retrospective understanding of what mattered most.

**What it shows**
- Fog events: duration vs severity
- Wind events: duration vs severity
- Heavy-rain events: accumulation vs severity
- Labels for the most severe events in each category

**Why it matters**
- Enables comparison across hazards
- Highlights trade-offs between duration and intensity
- Provides context for post-event analysis and reporting


### Model Performance Summary

**File**: `eurus_lab/dashboards/model_performance_summary.py`  
**Output**: `eurus_lab/assets/model_performance_summary.png`

**Purpose**  
Model trust and validation.

**What it shows**
- ROC curves (fog, wind)
- Precision-Recall curves (fog, wind)
- Calibration curves (fog, wind)

**Why it matters**
- Demonstrates discrimination skill beyond climatology and persistence
- Shows probability calibration (are probabilities meaningful?)
- Makes limitations explicit

**Note on precipitation**
Heavy rainfall events were too rare in the evaluation window to support robust ML validation.  
As a result, precipitation risk is handled via event-based diagnostics in Boreas, rather than hourly ML classification in Zephyrus.


## Relationship to Other Modules

- **Notus**  
  Ingests observational data and derives physical diagnostics (dewpoint depression, pressure tendency, moisture variables).

- **Boreas**  
  Detects and scores weather events (fog, wind, rain) using physically motivated criteria.

- **Zephyrus**  
  Learns probabilistic mappings from diagnostics to hazard risk, benchmarked against persistence and climatology.

**Eurus** sits on top of all three, turning outputs into interpretable visuals.


## Summary

Eurus demonstrates how meteorological data products can be:
- physically grounded
- probabilistic rather than binary

Together with Notus, Boreas, and Zephyrus, it forms a complete, realistic end-to-end weather analytics pipeline.

# Additional Summaries 

## Fog / Low-Cloud Risk Animation (London)

This visualization combines hourly observations with a simple, interpretable fog/low-cloud risk diagnostic.

**Risk heuristic (diagnostic, not a forecast):**
- Dew point depression (T − Td) ≤ 2 °C
- Wind speed ≤ 3 m/s
- Nighttime hours (Europe/London local time)

Nighttime is shaded lightly; fog/low-cloud risk hours are highlighted more strongly.

