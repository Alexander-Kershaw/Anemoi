from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out_dir = Path("eurus_lab/assets")
out_dir.mkdir(parents=True, exist_ok=True)

# Load events
fog = pd.read_parquet("boreas_lab/data/curated/events/fog_events_london_v2.parquet")
wind = pd.read_parquet("boreas_lab/data/curated/events/wind_events_london_v1.parquet")
rain = pd.read_parquet("boreas_lab/data/curated/events/rain_events_london_v1.parquet")

# Ensure datetimes exist for labeling
for df in (fog, wind, rain):
    if "start_utc" in df.columns:
        df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)

def _label_top(ax, df, xcol, ycol, n=3):
    if df.empty or ycol not in df.columns:
        return
    top = df.sort_values(ycol, ascending=False).head(n)
    for _, r in top.iterrows():
        txt = str(r["start_utc"].date()) if "start_utc" in df.columns else str(int(r.get("event_id", -1)))
        ax.annotate(txt, (r[xcol], r[ycol]), xytext=(5, 5), textcoords="offset points", fontsize=8)

# Create figure (3 panels: fog, wind, rain)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Fog panel
ax = axes[0]
if not fog.empty:
    ax.scatter(fog["duration_hours"], fog["severity_score"], s=35)
    _label_top(ax, fog, "duration_hours", "severity_score", n=3)
ax.set_title("Fog events (Boreas)")
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Severity score")

# Wind panel
ax = axes[1]
if not wind.empty:
    ax.scatter(wind["duration_hours"], wind["severity_score"], s=35)
    _label_top(ax, wind, "duration_hours", "severity_score", n=3)
ax.set_title("High-wind events (Boreas)")
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Severity score")

# Rain panel
ax = axes[2]
if not rain.empty:
    # rain uses accumulation as the x-axis if available
    xcol = "total_prcp_mm" if "total_prcp_mm" in rain.columns else "duration_hours"
    ax.scatter(rain[xcol], rain["severity_score"], s=45)
    _label_top(ax, rain, xcol, "severity_score", n=2)  # Only 2 scores exist (limited severe precipitation events)
    ax.set_xlabel("Total precipitation (mm)" if xcol == "total_prcp_mm" else "Duration (hours)")
ax.set_title("Heavy-rain events (Boreas)")
ax.set_ylabel("Severity score")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.suptitle("Event Severity Summary — London (Fog, Wind, Rain)", y=1.02)
plt.tight_layout()

out = out_dir / "event_severity_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved: {out.resolve()}")
