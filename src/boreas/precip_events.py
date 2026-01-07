from __future__ import annotations

import pandas as pd
from termcolor import colored


def add_heavy_rain_flags(df: pd.DataFrame, prcp_col: str = "prcp", rate_threshold_mmph: float = 2.0) -> pd.DataFrame:
    """
    A per-hour heavy rain flag.

    Assumes prcp is hourly precipitation (mm) for that hour.

    Default precipitation threshold:
    - 2 mm/hr = noticeable steady rain
    """
    out = df.copy()
    if prcp_col not in out.columns:
        raise ValueError(f"Missing precipitation column: {prcp_col}")

    pr = pd.to_numeric(out[prcp_col], errors="coerce").fillna(0.0)
    out["heavy_rain_v1"] = (pr >= rate_threshold_mmph).astype("int8")
    out["prcp_mmph"] = pr 

    return out


def add_precip_event_metrics(events: pd.DataFrame, df_hourly: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """
    Serves accumulation and peak-rate metrics to events

    Assumptions:
    - precipitation events start when flag==1 and previous hour flag==0
    - event_id is cumulative sum of starts for event distinction
    """
    out = events.copy()

    full = df_hourly.copy()
    full["timestamp_utc"] = pd.to_datetime(full["timestamp_utc"], utc=True)
    full = full.sort_values("timestamp_utc").reset_index(drop=True)

    flag = full[flag_col].fillna(0).astype(int)
    starts = (flag == 1) & (flag.shift(1, fill_value=0) == 0)
    full["event_id"] = starts.cumsum()

    flagged = full.loc[flag == 1].copy()
    if flagged.empty:
        out["total_prcp_mm"] = pd.NA
        out["max_prcp_mmph"] = pd.NA
        out["rain_hours_fraction"] = pd.NA
        return out

    # Total accumulation and peak hourly rate
    acc = flagged.groupby("event_id")["prcp_mmph"].sum().rename("total_prcp_mm").reset_index()
    peak = flagged.groupby("event_id")["prcp_mmph"].max().rename("max_prcp_mmph").reset_index()

    # fraction of flagged hours where prcp>0 
    frac = (
        flagged.assign(_rain=(flagged["prcp_mmph"] > 0).astype(int))
        .groupby("event_id")["_rain"].mean()
        .rename("rain_hours_fraction")
        .reset_index()
    )

    out = out.merge(acc, on="event_id", how="left")
    out = out.merge(peak, on="event_id", how="left")
    out = out.merge(frac, on="event_id", how="left")
    return out


def score_rain_event_severity(events: pd.DataFrame) -> pd.DataFrame:
    """
    Severity score for rain events.

    Components:
    - accumulation (total_prcp_mm)
    - peak rate (max_prcp_mmph)
    - duration (duration_hours)

    Final score = mean of components
    """
    out = events.copy()
    comps = []

    if "total_prcp_mm" in out.columns:
        denom = float(out["total_prcp_mm"].quantile(0.95)) or 1.0
        out["severity_accum"] = (out["total_prcp_mm"] / denom).clip(upper=1.5) / 1.5
        comps.append("severity_accum")

    if "max_prcp_mmph" in out.columns:
        denom = float(out["max_prcp_mmph"].quantile(0.95)) or 1.0
        out["severity_peak"] = (out["max_prcp_mmph"] / denom).clip(upper=1.5) / 1.5
        comps.append("severity_peak")

    if "duration_hours" in out.columns and out["duration_hours"].max() > 0:
        out["severity_duration"] = out["duration_hours"] / out["duration_hours"].max()
        comps.append("severity_duration")

    out["severity_score"] = out[comps].mean(axis=1) if comps else 0.0
    return out
