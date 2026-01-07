from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import meteostat as ms
from termcolor import colored
from zoneinfo import ZoneInfo

from shared.meteo_math import (
    wind_spd_kmh_to_ms,
    wind_vec_components_ms,
    dewpoint_from_temp_rh_c,
    vapor_pressure_from_dewpoint_hpa,
    specific_humidity_g_per_kg,
)

"""
Notus: Ingest hourly surface observations (Meteostat)

The function of this script:
1) Finds the nearest weather station to a latitude and longitude (lat, lon)
2) Downloads HOURLY observations for a date range (using older start date to avoid data latency)
3) Applies some consercative quality control (QC)
4) Writes partitioned Parquet (location_name + date) to:
   notus_lab/data/curated/meteostat_hourly/

Why we do station-first (instead of point-first):
- It’s more reliable across Meteostat API variants because some don’t support point-based queries
- It avoids empty results for some point-based queries 
"""

# Configuration data classes
@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    altitude_m: float | None = None


def _to_naive_datetime(date_str: str) -> "pd.Timestamp":
    
    #Convert YYYY-MM-DD (or similar) to timezone-naive datetime for Meteostat
    
    return pd.to_datetime(date_str).to_pydatetime()


def nearest_station_id(lat: float, lon: float) -> str:
    if not hasattr(ms, "stations"):
        raise RuntimeError("Your 'meteostat' package does not expose ms.stations.")

    point = ms.Point(lat, lon)

    result = ms.stations.nearby(point)

    # Some variants return an object with .fetch(), others return a DataFrame directly
    stations_df = result.fetch(1) if hasattr(result, "fetch") else result

    if stations_df is None or len(stations_df) == 0:
        raise RuntimeError("No nearby stations found for this location.")

    return stations_df.index[0]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some Meteostat variants may return non-string column labels (e.g. enums)
    Conversion to strings for Parquet compatibility is safer
    """
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out


# Core pipeline functions

def fetch_hourly(location: Location, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Fetch hourly observations for the nearest station to the location

    Parameters : start_utc, end_utc:
        Date strings like "2024-01-01". (We store timestamps as UTC later.)
    """
    start = _to_naive_datetime(start_utc)
    end = _to_naive_datetime(end_utc)

    station_id = nearest_station_id(location.lat, location.lon)
    print(f"Using nearest station_id={station_id}")

    # Meteostat variants differ: sometimes ms.hourly returns an object with .fetch(),
    # sometimes it returns a DataFrame directly so we handle both cases
    result = ms.hourly(station_id, start, end)
    df = result.fetch() if hasattr(result, "fetch") else result

    if df is None or len(df) == 0:
        raise RuntimeError(
            "No hourly data returned (empty). "
            "This can be due to data coverage/latency. Try an earlier date range "
            "(e.g., 2024-01-01 to 2024-02-01)."
        )

    # Standardize: time index -> timestamp column in UTC
    df = df.reset_index().rename(columns={"time": "timestamp_utc"})
    df = normalize_column_names(df)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Attach metadata for downstream partitioning
    df["location_name"] = location.name
    df["station_id"] = station_id
    df["lat"] = location.lat
    df["lon"] = location.lon

    return df


def basic_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative first-pass QC:
    - Don't drop rows
    - Replace physically impossible values with NA
    """
    out = df.copy()

    # Temperature (°C): conservative global range
    if "temp" in out.columns:
        out.loc[(out["temp"] < -90) | (out["temp"] > 60), "temp"] = pd.NA

    # Dew point (°C) should generally not exceed air temp
    if "dwpt" in out.columns and "temp" in out.columns:
        out.loc[out["dwpt"] > out["temp"], "dwpt"] = pd.NA

    # RH (%) is 0..100
    if "rhum" in out.columns:
        # Some Meteostat variants call it rhum instead of rh
        out.loc[(out["rhum"] < 0) | (out["rhum"] > 100), "rhum"] = pd.NA
    if "rh" in out.columns:
        out.loc[(out["rh"] < 0) | (out["rh"] > 100), "rh"] = pd.NA

    # Wind direction 0..360
    if "wdir" in out.columns:
        out.loc[(out["wdir"] < 0) | (out["wdir"] > 360), "wdir"] = pd.NA

    # Wind speed non-negative
    if "wspd" in out.columns:
        out.loc[out["wspd"] < 0, "wspd"] = pd.NA

    # Precip non-negative
    if "prcp" in out.columns:
        out.loc[out["prcp"] < 0, "prcp"] = pd.NA

    # Pressure (hPa) conservative range
    if "pres" in out.columns:
        out.loc[(out["pres"] < 870) | (out["pres"] > 1085), "pres"] = pd.NA

    return out


def derive_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived meteorological variables with explicit units in column names

    after qiality control (QC): since we want to avoid propagating bad data
    """
    out = df.copy()

    # Wind speed in m/s (common in meteorology / models)
    if "wspd" in out.columns:
        out["wspd_ms"] = wind_spd_kmh_to_ms(out["wspd"])

    # Vapor pressure (hPa) from dew point
    if "dwpt" in out.columns:
        out["e_hpa"] = vapor_pressure_from_dewpoint_hpa(out["dwpt"])

    # Specific humidity (g/kg) from dew point and pressure
    if "dwpt" in out.columns and "pres" in out.columns:
        out["q_gkg"] = specific_humidity_g_per_kg(out["dwpt"], out["pres"])

    # Wind vector components (u, v) in m/s
    if "wspd_ms" in out.columns and "wdir" in out.columns:
        u, v = wind_vec_components_ms(out["wspd_ms"], out["wdir"])
        out["u_ms"] = u
        out["v_ms"] = v

    # Binary rain indicator
    if "prcp" in out.columns:
        out["is_rain_hour"] = (out["prcp"] > 0).astype("int8")

    # 1-hour pressure tendency (hPa)
    if "pres" in out.columns:
        out["dpres_1h_hpa"] = out["pres"].diff()

    # Derive dew point from temp + RH (since dwpt isn't provided)
    if "temp" in out.columns and "rhum" in out.columns:
        out["dwpt_c"] = dewpoint_from_temp_rh_c(out["temp"], out["rhum"])

    # Vapor pressure from dew point
    if "dwpt_c" in out.columns:
        out["e_hpa"] = vapor_pressure_from_dewpoint_hpa(out["dwpt_c"])

    # Specific humidity (g/kg) from dew point and pressure
    if "dwpt_c" in out.columns and "pres" in out.columns:
        out["q_gkg"] = specific_humidity_g_per_kg(out["dwpt_c"], out["pres"])

    # Dewpoint depression (°C) = T - Td
    if "temp" in out.columns and "dwpt_c" in out.columns:
        out["dewpoint_depression_c"] = out["temp"] - out["dwpt_c"]


    return out


def _night_mask_local(timestamp_utc: pd.Series, tz_name: str = "Europe/London") -> pd.Series:
    """
    Return boolean mask for nighttime hours in local time.
    Night definition (simple): 21:00–06:00 local.
    """
    ts = pd.to_datetime(timestamp_utc, utc=True)
    ts_local = ts.dt.tz_convert(ZoneInfo(tz_name))
    h = ts_local.dt.hour
    return (h >= 21) | (h <= 6)


def _apply_persistence(flag: pd.Series, min_consecutive_hours: int = 2) -> pd.Series:
    """
    Require events to persist for at least `min_consecutive_hours`.

    For min_consecutive_hours=2, we keep hours that are part of at least a 2-hour run.
    This removes isolated single-hour spikes.

    Note: assumes data is roughly hourly. We'll make it more strict later if needed.
    """
    f = flag.fillna(0).astype(int)

    if min_consecutive_hours <= 1:
        return f.astype("int8")

    # For 2-hour persistence: keep hours that have a neighbor also flagged
    if min_consecutive_hours == 2:
        keep = (f == 1) & ((f.shift(1, fill_value=0) == 1) | (f.shift(-1, fill_value=0) == 1))
        return keep.astype("int8")

    # For >2 hours: rolling window count (centered)
    # Keeps hours that lie within any window of length min_consecutive_hours with all ones.
    roll = f.rolling(window=min_consecutive_hours, min_periods=min_consecutive_hours).sum()
    # Mark window endings, then expand back to all hours inside valid windows
    window_ok = (roll == min_consecutive_hours)
    expanded = pd.Series(False, index=f.index)
    for idx in window_ok[window_ok].index:
        expanded.loc[idx - (min_consecutive_hours - 1): idx] = True
    keep = (f == 1) & expanded
    return keep.astype("int8")


def detect_fog_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fog/low-cloud diagnostics based on simple physical rules

    Outputs:
    - fog_risk_v1: baseline rule (UTC-night; no persistence)
    - fog_risk_v2: improved rule (local-night; persistence >=2 hours)
    - fog_risk: alias to fog_risk_v1 (so downstream code won't break)
    - fog_risk_score: continuous 0..1 score (useful for tuning/ML))

    Baseline physical logic:
    - near saturation: dewpoint_depression_c <= 2°C
    - light wind: wspd_ms <= 3 m/s
    - nighttime: defined by hour range
    """
    out = df.copy()

    # Ensure required features exist in the DataFrame
    if "dewpoint_depression_c" not in out.columns and "temp" in out.columns and "dwpt_c" in out.columns:
        out["dewpoint_depression_c"] = out["temp"] - out["dwpt_c"]

    required = ["timestamp_utc", "dewpoint_depression_c", "wspd_ms"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Fog diagnostic missing required columns: {missing}")

    # Core physical conditions
    near_saturation = out["dewpoint_depression_c"] <= 2.0 
    light_wind = out["wspd_ms"] <= 3.0

    # v1: UTC night (21:00–06:00) + no persistence
    hour_utc = pd.to_datetime(out["timestamp_utc"], utc=True).dt.hour
    night_utc = (hour_utc >= 21) | (hour_utc <= 6) # UTC nighttime mask

    fog_v1 = (near_saturation & light_wind & night_utc).astype("int8")
    out["fog_risk_v1"] = fog_v1

    # v2: Local night + persistence >=2 hours
    night_local = _night_mask_local(out["timestamp_utc"], tz_name="Europe/London")
    fog_raw_v2 = (near_saturation & light_wind & night_local).astype("int8")
    out["fog_risk_v2"] = _apply_persistence(fog_raw_v2, min_consecutive_hours=2)

    # Backwards-compatible alias (I dont want to break Eurus/Boreas)
    out["fog_risk"] = out["fog_risk_v1"]

    # Continuous score (0..1): how strongly conditions support fog risk at this hour
    # Saturation factor: 1 at 0°C depression, 0 at >=2°C 
    sat = (2.0 - out["dewpoint_depression_c"]).clip(lower=0.0, upper=2.0) / 2.0
    # Wind factor: 1 at 0 m/s, 0 at >=3 m/s
    wnd = (3.0 - out["wspd_ms"]).clip(lower=0.0, upper=3.0) / 3.0
    nmask = night_local.astype(float)

    out["fog_risk_score"] = (sat * wnd * nmask).astype(float)

    return out


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Write Parquet partitioned by location_name and date in following structure:

    notus_lab/data/curated/meteostat_hourly/location_name=London/date=2026-01-01/hourly.parquet
    """
    
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

    for (loc, date), g in df.groupby(["location_name", "date"], sort=True):
        part_dir = out_dir / f"location_name={loc}" / f"date={date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        (
            g.drop(columns=["date"])
            .sort_values("timestamp_utc")
            .to_parquet(part_dir / "hourly.parquet", index=False)
        )


def main() -> None:
    # Portfolio home base (demo set to London), change location as needed in main()
    location = Location(name="London", lat=51.5074, lon=-0.1278, altitude_m=25)

    # Lag-safe window: end 14 days ago, look back ~74 days
    # (This avoids empty data due to provider update delays, too recent data may be missing)
    end_utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=14)).date().isoformat()
    start_utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=88)).date().isoformat()

    print(colored(f"Fetching hourly observations for {location.name} [{start_utc} → {end_utc}] ...", "cyan"))

    raw = fetch_hourly(location, start_utc, end_utc)

    print("Columns returned:", list(raw.columns))
    print("Rows:", len(raw))

    curated = basic_qc(raw)
    curated = derive_variables(curated) # Add derived variables
    curated = detect_fog_risk(curated)  # Add fog risk diagnostic flag

    # Fog risk summary report
    def _pct(x: int, n: int) -> float:
        return 100.0 * x / n if n else 0.0 # avoid division by zero

    # Fog risk summary
    n = len(curated)
    v1 = int(curated["fog_risk_v1"].sum()) if "fog_risk_v1" in curated.columns else 0
    v2 = int(curated["fog_risk_v2"].sum()) if "fog_risk_v2" in curated.columns else 0
    print(colored(f"\nFog/low-cloud risk v1 hours: {v1} ({_pct(v1, n):.1f}%)", "yellow")) # print summary of fog risk v1 percentage of hours wuth risk of fog/low-cloud
    print(colored(f"Fog/low-cloud risk v2 hours: {v2} ({_pct(v2, n):.1f}%)", "yellow")) # print summary of fog risk v2 percentage of hours wuth risk of fog/low-cloud


    # Missingness report, important for understanding data quality
    def _missing_val_report(df: pd.DataFrame, cols: list[str]) -> None:
        present = [c for c in cols if c in df.columns]
        if not present:
            print("No columns found for missingness report.")
            return
        report = (
            df[present]
            .isna()
            .mean()
            .sort_values(ascending=False)
            .mul(100)
            .round(2)
        )
        print("\nMissingness (% of rows):")
        for k, v in report.items():
            print(f"  {k}: {v}%")

    out_dir = Path("notus_lab/data/curated/meteostat_hourly")
    write_partitioned_parquet(curated, out_dir)

    print(colored(f"Wrote curated Parquet partitions to: {out_dir.resolve()}", "green"))

    # Report missingness for key variables
    _missing_val_report(curated, [
    "dewpoint_depression_c", "fog_risk", "fog_risk_v1", "fog_risk_v2", "fog_risk_score"])

    # Fog risk summary
    fog_hours = curated["fog_risk"].sum()
    total_hours = len(curated)


if __name__ == "__main__":
    main()
