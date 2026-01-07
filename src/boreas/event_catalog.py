from __future__ import annotations

import pandas as pd

# ensure timestamp_utc is datetime, sorted, witn no gaps
def _ensure_sorted_hourly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp_utc" not in out.columns:
        raise ValueError("Expected 'timestamp_utc' column.")
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    return out


def make_event_catalog(df: pd.DataFrame, flag_col: str, min_duration_hours: int = 2) -> pd.DataFrame:
    """
    Convert a per-hour binary flag into an event catalogue

    Event definition:
    - An event is a contiguous run of hours where flag_col == 1
    - Keep events with duration >= min_duration_hours

    Returns:
      DataFrame with one row per event and useful summary statistics
    """
    df = _ensure_sorted_hourly(df) # ensure proper datetime and sorting

    if flag_col not in df.columns:
        raise ValueError(f"Missing flag column: {flag_col}")

    flag = df[flag_col].fillna(0).astype(int) # ensure binary flag

    # Start of a new event when current is 1 and previous is 0 
    starts = (flag == 1) & (flag.shift(1, fill_value=0) == 0)
    event_id = starts.cumsum()

    # Keep only flagged rows
    flagged = df.loc[flag == 1].copy()
    if flagged.empty:
        return pd.DataFrame(columns=[
            "event_id", "start_utc", "end_utc", "duration_hours",
            "min_dewpoint_depression_c", "mean_dewpoint_depression_c",
            "mean_rhum", "mean_wspd_ms", "mean_temp_c",
            "any_precip", "mean_dpres_1h_hpa"
        ])

    flagged["event_id"] = event_id.loc[flag == 1].to_numpy()

    # Aggregation, only include columns if they exist
    agg = {
        "timestamp_utc": ["min", "max", "count"],
    }

    if "dewpoint_depression_c" in flagged.columns:
        agg["dewpoint_depression_c"] = ["min", "mean"]
    if "rhum" in flagged.columns:
        agg["rhum"] = ["mean"]
    if "wspd_ms" in flagged.columns:
        agg["wspd_ms"] = ["mean"]
    if "temp" in flagged.columns:
        agg["temp"] = ["mean"]
    if "prcp" in flagged.columns:
        agg["prcp"] = ["max"]  # any precip if max > 0
    if "dpres_1h_hpa" in flagged.columns:
        agg["dpres_1h_hpa"] = ["mean"]

    
    grouped = flagged.groupby("event_id").agg(agg) # group by event_id and aggregate
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()] # flatten MultiIndex
    grouped = grouped.reset_index() 

    # Build output DataFrame
    out = pd.DataFrame({
        "event_id": grouped["event_id"],
        "start_utc": grouped["timestamp_utc_min"],
        "end_utc": grouped["timestamp_utc_max"],
        "duration_hours": grouped["timestamp_utc_count"].astype(int),
    })

    if "dewpoint_depression_c_min" in grouped:
        out["min_dewpoint_depression_c"] = grouped["dewpoint_depression_c_min"]
    if "dewpoint_depression_c_mean" in grouped:
        out["mean_dewpoint_depression_c"] = grouped["dewpoint_depression_c_mean"]
    if "rhum_mean" in grouped:
        out["mean_rhum"] = grouped["rhum_mean"]
    if "wspd_ms_mean" in grouped:
        out["mean_wspd_ms"] = grouped["wspd_ms_mean"]
    if "temp_mean" in grouped:
        out["mean_temp_c"] = grouped["temp_mean"]
    if "prcp_max" in grouped:
        out["any_precip"] = (grouped["prcp_max"] > 0).astype("int8")
    if "dpres_1h_hpa_mean" in grouped:
        out["mean_dpres_1h_hpa"] = grouped["dpres_1h_hpa_mean"]

    # Filter short events out based on min_duration_hours
    out = out.loc[out["duration_hours"] >= int(min_duration_hours)].reset_index(drop=True)

    return out


def score_fog_event_severity(events: pd.DataFrame) -> pd.DataFrame:
    """
    Function interpreting fog event severity

    Components (normalized 0..1):
    - saturation: deeper saturation implies more severe
    - calmness: weaker winds implies more severe
    - duration: longer events implies more severe

    The final score is the mean of available components giving a comprehensive fog event
    severity score
    """
    out = events.copy() # avoid modifying original

    components = []

    # Saturation severity (lower dewpoint depression = worse saturation = higher severity)
    if "min_dewpoint_depression_c" in out.columns:
        # 0°C -> 1.0, 2°C or more -> 0.0
        sat = (2.0 - out["min_dewpoint_depression_c"]).clip(lower=0.0, upper=2.0) / 2.0
        out["severity_saturation"] = sat
        components.append("severity_saturation")

    # Wind calmness (0 m/s -> 1.0, 3 m/s or more -> 0.0)
    if "mean_wspd_ms" in out.columns:
        wnd = (3.0 - out["mean_wspd_ms"]).clip(lower=0.0, upper=3.0) / 3.0
        out["severity_calm"] = wnd
        components.append("severity_calm")

    # Duration severity (scale by max duration in dataset)
    if "duration_hours" in out.columns:
        max_dur = out["duration_hours"].max()
        if max_dur > 0:
            dur = out["duration_hours"] / max_dur
            out["severity_duration"] = dur
            components.append("severity_duration")

    # Final severity score (mean average of components)
    if components:
        out["severity_score"] = out[components].mean(axis=1)
    else:
        out["severity_score"] = 0.0

    return out
