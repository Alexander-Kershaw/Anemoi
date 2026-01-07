from pathlib import Path
import pandas as pd
from termcolor import colored

from boreas.event_catalog import make_event_catalog, score_fog_event_severity

# Load curated Notus dataset (London for demonstration)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Use hardened Notus fog diagnostic (v2)
events = make_event_catalog(df, flag_col="fog_risk_v2", min_duration_hours=2) # using v2 fog risk flag (improvement on v1)
events = score_fog_event_severity(events) # add severity scoring

print("Fog events (v2) detected:", len(events))

# Display top 10 most severe fog/ lowx-cloud events
if len(events) > 0:
    print("\nTop 10 most severe events:")
    print(events.sort_values("severity_score", ascending=False).head(10).to_string(index=False) # display severity score in decending order
)
    
# Save event catalog
out_dir = Path("boreas_lab/data/curated/events")
out_dir.mkdir(parents=True, exist_ok=True)

events.to_parquet(out_dir / "fog_events_london_v2.parquet", index=False)
events.to_csv(out_dir / "fog_events_london_v2.csv", index=False)

print(colored(f"\nWrote:\n  {out_dir / 'fog_events_london_v2.parquet'}\n  {out_dir / 'fog_events_london_v2.csv'}", "light_green"))
