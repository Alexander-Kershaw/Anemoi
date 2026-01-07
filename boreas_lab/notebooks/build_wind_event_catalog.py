from pathlib import Path
import pandas as pd
from termcolor import colored

from boreas.event_catalog import make_event_catalog
from boreas.wind_events import add_high_wind_flags, score_wind_event_severity

# Load curated Notus dataset (London for demonstration)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# high-wind flags (tunable thresholds)
df = add_high_wind_flags(
    df,
    speed_col="wspd_ms",
    gust_col="wpgt",
    speed_threshold_ms=10.0,  # sustained
    gust_threshold_ms=15.0,   # gust
)

# Build events from the hourly high-wind flag
events = make_event_catalog(df, flag_col="high_wind_v1", min_duration_hours=2) 

# Add gust summary if present
if "wpgt_ms" in df.columns and not events.empty:
    # For each event, compute max gust during event
    flagged = df[df["high_wind_v1"] == 1].copy()
    flagged["timestamp_utc"] = pd.to_datetime(flagged["timestamp_utc"], utc=True)
    flagged = flagged.sort_values("timestamp_utc").reset_index(drop=True)

    # Recreate event_id the same way make_event_catalog does
    f = flagged["high_wind_v1"].fillna(0).astype(int)
    # event_id in the events table is based on starts computed on the full series,
    # is computed again here so on the full df the merge is clean
    full = df.copy()
    full["timestamp_utc"] = pd.to_datetime(full["timestamp_utc"], utc=True)
    full = full.sort_values("timestamp_utc").reset_index(drop=True)
    flag = full["high_wind_v1"].fillna(0).astype(int)
    starts = (flag == 1) & (flag.shift(1, fill_value=0) == 0)
    full["event_id"] = starts.cumsum()

    flagged = full.loc[flag == 1].copy()
    gust_max = flagged.groupby("event_id")["wpgt_ms"].max().rename("max_wpgt_ms").reset_index()

    events = events.merge(gust_max, on="event_id", how="left")

# Add severity score
events = score_wind_event_severity(events)

print("High-wind events detected:", len(events))

if len(events) > 0:
    print(colored("\nTop 10 most severe high-wind events:", "light_green"))
    cols = [c for c in [
        "start_utc", "end_utc", "duration_hours",
        "mean_wspd_ms", "max_wpgt_ms",
        "severity_score", "any_precip"
    ] if c in events.columns]
    print(events.sort_values("severity_score", ascending=False).head(10)[cols].to_string(index=False))

out_dir = Path("boreas_lab/data/curated/events")
out_dir.mkdir(parents=True, exist_ok=True)

events.to_parquet(out_dir / "wind_events_london_v1.parquet", index=False)
events.to_csv(out_dir / "wind_events_london_v1.csv", index=False)

print(colored(f"\nWrote:\n  {out_dir / 'wind_events_london_v1.parquet'}\n  {out_dir / 'wind_events_london_v1.csv'}", "green"))
