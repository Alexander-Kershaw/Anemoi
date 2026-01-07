from __future__ import annotations

import pandas as pd


def add_high_wind_flags(
    df: pd.DataFrame,
    speed_col: str = "wspd_ms",
    gust_col: str = "wpgt",
    speed_threshold_ms: float = 10.0,
    gust_threshold_ms: float = 15.0,
) -> pd.DataFrame:
    """
    Add per-hour high-wind diagnostic flags

    Default thresholds (tunable):
    - sustained wind >= 10 m/s  (~36 km/h)
    - gust >= 15 m/s           (~54 km/h)

    Important:
    - Many datasets provide gust in the same units as wind speed (often km/h in Meteostat),
      howver, curated dataset currently has 'wpgt' and 'wspd_ms'.
      conversions are handled as so:
        * If wpgt looks too large (e.g., > 60), chances are it might be in km/h.
        * convert km/h -> m/s if its warranted.

    - To manage missing values indicating not high wind fillna(-1) used as a logical constraint
    since wind can't be negative, this essentially translates to 'no high wind' to avoid issues with
    NA to int type conversions
    """
    out = df.copy()

    if speed_col not in out.columns:
        raise ValueError(f"Missing required wind speed column: {speed_col}")

    # Gust handling (gust being defined >= 15m/s)
    gust_ms = None
    if gust_col in out.columns:
        g = out[gust_col].copy()

        # unit check:
        # If gust values commonly exceed 60, it's almost certainly must be in km/h.
        # (60 m/s is extreme wind speeds; 60 km/h is sensible in comparison.)
        if pd.to_numeric(g, errors="coerce").dropna().quantile(0.90) > 60:
            gust_ms = g * (1000.0 / 3600.0)  # km/h -> m/s
        else:
            gust_ms = g.astype(float)

        out["wpgt_ms"] = gust_ms

    # Sustained wind flag
    speed = pd.to_numeric(out[speed_col], errors="coerce") # avoiding NA's since cant convert NA to int8 type
    out["high_wind_speed_v1"] = (speed.fillna(-1) >= speed_threshold_ms).astype("int8")

    # Gust flag (if gust has values)
    if gust_ms is not None:
        gust = pd.to_numeric(out["wpgt_ms"], errors="coerce") # also handling NA for integer conversion
        out["high_wind_gust_v1"] = (gust.fillna(-1) >= gust_threshold_ms).astype("int8")

        out["high_wind_v1"] = ((out["high_wind_speed_v1"] == 1) | (out["high_wind_gust_v1"] == 1)).astype("int8")
    else:
        out["high_wind_v1"] = out["high_wind_speed_v1"].astype("int8")

    return out


def score_wind_event_severity(events: pd.DataFrame) -> pd.DataFrame:
    """
    An interpretable wind-event severity score (0..1)

    Components:
    - intensity (mean sustained wind): higher is more severe
    - peak gust (if present): higher is more severe
    - duration: longer is more severe

    Normalization is dataset-relative to keep it simple and robust.
    """
    out = events.copy()
    components = []

    # Duration scaled by max duration
    if "duration_hours" in out.columns and out["duration_hours"].max() > 0:
        out["severity_duration"] = out["duration_hours"] / out["duration_hours"].max()
        components.append("severity_duration")

    # Mean wind scaled by 95th percentile
    if "mean_wspd_ms" in out.columns:
        denom = out["mean_wspd_ms"].quantile(0.95)
        denom = float(denom) if denom and denom > 0 else 1.0
        out["severity_intensity"] = (out["mean_wspd_ms"] / denom).clip(upper=1.5) / 1.5
        components.append("severity_intensity")

    # Peak gust scaled by 95th percentile (if present)
    if "max_wpgt_ms" in out.columns:
        denom = out["max_wpgt_ms"].quantile(0.95)
        denom = float(denom) if denom and denom > 0 else 1.0
        out["severity_gust"] = (out["max_wpgt_ms"] / denom).clip(upper=1.5) / 1.5
        components.append("severity_gust")

    out["severity_score"] = out[components].mean(axis=1) if components else 0.0
    return out
