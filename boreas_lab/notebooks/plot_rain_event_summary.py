from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

events = pd.read_parquet(
    "boreas_lab/data/curated/events/rain_events_london_v1.parquet"
)

out_dir = Path("boreas_lab/notebooks/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

if events.empty:
    raise SystemExit("No rain events found.")


# Plot 1: Duration histogram
plt.figure()
plt.hist(events["duration_hours"], bins=range(1, int(events["duration_hours"].max()) + 2))
plt.xlabel("Event duration (hours)")
plt.ylabel("Count")
plt.title("Heavy-Rain Events (London): Duration Distribution")
plt.tight_layout()
plt.savefig(out_dir / "rain_event_duration_hist.png", dpi=150)
plt.show()


# Plot 2: Severity vs accumulation
plt.figure()
plt.scatter(events["total_prcp_mm"], events["severity_score"], s=40)
plt.xlabel("Total precipitation (mm)")
plt.ylabel("Severity score")
plt.title("Heavy-Rain Events: Severity vs Accumulation")
plt.tight_layout()
plt.savefig(out_dir / "rain_event_severity_vs_accum.png", dpi=150)
plt.show()

print("Saved rain plots to:", out_dir.resolve())
