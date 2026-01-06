from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import meteostat as ms


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
    
    #Convert YYYY-MM-DD (or similar) to timezone-naive datetime for Meteostat.
    
    return pd.to_datetime(date_str).to_pydatetime()


def nearest_station_id(lat: float, lon: float) -> str:
    if not hasattr(ms, "stations"):
        raise RuntimeError("Your 'meteostat' package does not expose ms.stations.")

    point = ms.Point(lat, lon)

    result = ms.stations.nearby(point)

    # Some variants return an object with .fetch(), others return a DataFrame directly.
    stations_df = result.fetch(1) if hasattr(result, "fetch") else result

    if stations_df is None or len(stations_df) == 0:
        raise RuntimeError("No nearby stations found for this location.")

    return stations_df.index[0]



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


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Write Parquet partitioned by location_name and date:

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
    # Portfolio home base, change location as needed
    location = Location(name="London", lat=51.5074, lon=-0.1278, altitude_m=25)

    # Lag-safe window: end 14 days ago, look back ~74 days
    # (This avoids empty data due to provider update delays.)
    end_utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=14)).date().isoformat()
    start_utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=88)).date().isoformat()

    print(f"Fetching hourly observations for {location.name} [{start_utc} → {end_utc}] ...")

    raw = fetch_hourly(location, start_utc, end_utc)

    print("Columns returned:", list(raw.columns))
    print("Rows:", len(raw))

    curated = basic_qc(raw)

    out_dir = Path("notus_lab/data/curated/meteostat_hourly")
    write_partitioned_parquet(curated, out_dir)

    print(f"Wrote curated Parquet partitions to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
