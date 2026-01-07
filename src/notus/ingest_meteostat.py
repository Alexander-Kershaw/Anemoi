from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import meteostat as ms
from termcolor import colored

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


def detect_fog_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify hours with risk of fog formation based on meteorological conditions

    Simple heuristic based on:
    - Dew point depression ≤ 2 °C  -> near saturation
    - Wind speed ≤ 3 m/s           -> weak mixing
    - Nighttime hours              -> radiative cooling

    This is a diahgnostic flag NOT a forcast
    """
    out = df.copy()

    # Nighttime definition (UTC-based, simple)
    hour = out["timestamp_utc"].dt.hour
    is_night = (hour >= 21) | (hour <= 6) # 9 PM to 6 AM UTC

    # Conditions for fog risk
    near_saturation = out["dewpoint_depression_c"] <= 2.0
    light_wind = out["wspd_ms"] <= 3.0

    out["fog_risk"] = (
        near_saturation
        & light_wind
        & is_night
    ).astype("int8") 

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
    "temp", "rhum", "pres", "wspd", "wdir", "prcp",
    "wspd_ms", "u_ms", "v_ms", "dpres_1h_hpa", "is_rain_hour",
    "dwpt_c", "e_hpa", "q_gkg"])


    # Fog risk summary
    fog_hours = curated["fog_risk"].sum()
    total_hours = len(curated)

    print(colored(
        f"\nFog / low-cloud risk hours: {fog_hours} "
        f"({100 * fog_hours / total_hours:.1f}% of all hours)", "yellow")) # Percentage of hours with risk of fog / low clouds


if __name__ == "__main__":
    main()
