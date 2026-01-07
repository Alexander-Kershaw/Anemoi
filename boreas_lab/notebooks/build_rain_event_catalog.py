from pathlib import Path
import pandas as pd
from termcolor import colored

from boreas.event_catalog import make_event_catalog
from boreas.precip_events import (
    add_heavy_rain_flags,
    add_precip_event_metrics,
    score_rain_event_severity,
)

# Load curated Notus dataset (London for demonstration)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Add hourly heavy-rain flag
df = add_heavy_rain_flags(df, prcp_col="prcp", rate_threshold_mmph=2.0)

# Build events from heavy_rain_v1 flag
events = make_event_catalog(df, flag_col="heavy_rain_v1", min_duration_hours=2)

# Add precip metrics & severity
events = add_precip_event_metrics(events, df_hourly=df, flag_col="heavy_rain_v1")
events = score_rain_event_severity(events)

print("Heavy-rain events detected:", len(events))

if len(events) > 0:
    print("\nTop 10 most severe heavy-rain events:")
    cols = [c for c in [
        "start_utc", "end_utc", "duration_hours",
        "total_prcp_mm", "max_prcp_mmph", "severity_score", "mean_wspd_ms"
    ] if c in events.columns]
    print(events.sort_values("severity_score", ascending=False).head(10)[cols].to_string(index=False))

out_dir = Path("boreas_lab/data/curated/events")
out_dir.mkdir(parents=True, exist_ok=True)

events.to_parquet(out_dir / "rain_events_london_v1.parquet", index=False)
events.to_csv(out_dir / "rain_events_london_v1.csv", index=False)

print(colored(f"\nWrote:\n  {out_dir / 'rain_events_london_v1.parquet'}\n  {out_dir / 'rain_events_london_v1.csv'}", "light_green"))
