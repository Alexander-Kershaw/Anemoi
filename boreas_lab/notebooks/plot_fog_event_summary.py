from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

# Load events for London with v2 fog risk flag
events_path = Path("boreas_lab/data/curated/events/fog_events_london_v2.parquet")
events = pd.read_parquet(events_path)

if events.empty:
    raise SystemExit("No events found. Run build_fog_event_catalog.py first.")

# Ensure datetime types are valid
events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True)
events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True)

out_dir = Path("boreas_lab/notebooks/outputs")
out_dir.mkdir(parents=True, exist_ok=True)


# Plot 1: Histogram of durations
plt.hist(events["duration_hours"], bins=range(2, int(events["duration_hours"].max()) + 2))
plt.xlabel("Event duration (hours)")
plt.ylabel("Count")
plt.title("Fog/Low-Cloud Events (London): Duration Distribution")
plt.tight_layout()
plt.savefig(out_dir / "fog_event_duration_hist.png", dpi=150)
plt.show()


# Plot 2: Severity vs duration
if "severity_score" not in events.columns:
    raise SystemExit("Missing severity_score. Ensure you ran the severity scoring step.")

plt.figure()
plt.scatter(events["duration_hours"], events["severity_score"], s=30)
plt.xlabel("Duration (hours)")
plt.ylabel("Severity score (0–1)")
plt.title("Fog/Low-Cloud Events (London): Severity vs Duration")
plt.tight_layout()
plt.savefig(out_dir / "fog_event_severity_vs_duration.png", dpi=150)
plt.show()

print(colored("Saved plots to:", out_dir.resolve(), "light_green"))
print("\nTop 5 most severe events:")
print(
    events.sort_values("severity_score", ascending=False)
    .head(5)[[
        "start_utc", "end_utc", "duration_hours", "severity_score",
        "min_dewpoint_depression_c", "mean_wspd_ms", "mean_rhum", "any_precip"
    ]]
    .to_string(index=False)
)
