from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

# Read parquet for wind events for location (London for demonstration)
events = pd.read_parquet(
    "boreas_lab/data/curated/events/wind_events_london_v1.parquet"
)

out_dir = Path("boreas_lab/notebooks/outputs")
out_dir.mkdir(parents=True, exist_ok=True)


# Plot 1: Duration histogram
plt.figure()
plt.hist(events["duration_hours"], bins=20)
plt.xlabel("Event duration (hours)")
plt.ylabel("Count")
plt.title("High-Wind Events (London): Duration Distribution")
plt.tight_layout()
plt.savefig(out_dir / "wind_event_duration_hist.png", dpi=150)
plt.show()


# Plot 2: Severity vs duration
plt.figure()
plt.scatter(events["duration_hours"], events["severity_score"], s=30)
plt.xlabel("Duration (hours)")
plt.ylabel("Severity score")
plt.title("High-Wind Events: Severity vs Duration")
plt.tight_layout()
plt.savefig(out_dir / "wind_event_severity_vs_duration.png", dpi=150)
plt.show()

print("Saved wind plots to:", out_dir.resolve())
