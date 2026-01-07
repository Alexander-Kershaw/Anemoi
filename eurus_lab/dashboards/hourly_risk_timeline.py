from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from termcolor import colored


# Load hourly Notus data for location (London for demonstration)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
df = pd.concat([pd.read_parquet(f) for f in base.rglob("hourly.parquet")], ignore_index=True)

# Ensuring appropriate timestamps
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# Manageable temporal window (last 14 days)
df = df.iloc[-14 * 24 :]


# Load Zephyrus probabilities and features
from joblib import load

fog_model = load("zephyrus_lab/models/fog_classifier_logreg_v1.joblib")["model"]
wind_model = load("zephyrus_lab/models/wind_classifier_logreg_v1.joblib")["model"]

fog_features = load("zephyrus_lab/models/fog_classifier_logreg_v1.joblib")["features"]
wind_features = load("zephyrus_lab/models/wind_classifier_logreg_v1.joblib")["features"]

df["p_fog"] = fog_model.predict_proba(df[fog_features])[:, 1]
df["p_wind"] = wind_model.predict_proba(df[wind_features])[:, 1]


# Load Boreas events
fog_events = pd.read_parquet("boreas_lab/data/curated/events/fog_events_london_v2.parquet")
wind_events = pd.read_parquet("boreas_lab/data/curated/events/wind_events_london_v1.parquet")


# Isolate events within the display window (cleaner plot)
t_start = df["timestamp_utc"].min()
t_end = df["timestamp_utc"].max()

fog_events = fog_events[
    (fog_events["end_utc"] >= t_start) &
    (fog_events["start_utc"] <= t_end)
]

wind_events = wind_events[
    (wind_events["end_utc"] >= t_start) &
    (wind_events["start_utc"] <= t_end)
]


# Missing severity score -> resort to constant shading
if "severity_score" not in fog_events.columns:
    fog_events["severity_score"] = 1.0
if "severity_score" not in wind_events.columns:
    wind_events["severity_score"] = 1.0


# Normalize severity to 0..1 within the plotted window
def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return pd.Series(1.0, index=s.index)
    return (s - lo) / (hi - lo)

fog_events["sev01"] = _norm01(fog_events["severity_score"])
wind_events["sev01"] = _norm01(wind_events["severity_score"])

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df["timestamp_utc"], df["p_fog"], label="Fog probability", color="tab:blue")
ax.plot(df["timestamp_utc"], df["p_wind"], label="High-wind probability", color="tab:red")

# Nighttime shading
hours = df["timestamp_utc"].dt.hour
night = (hours < 6) | (hours >= 21)
ax.fill_between(
    df["timestamp_utc"], 0, 1,
    where=night,
    color="k", alpha=0.05, label="Nighttime"
)

# Event shading with severity-weighted alpha (shading degree)
# (low severity = faint; high severity = stronger)
for _, ev in fog_events.iterrows():
    alpha = 0.04 + 0.18 * float(ev["sev01"])   # ~0.04..0.22
    ax.axvspan(ev["start_utc"], ev["end_utc"], color="C0", alpha=alpha)

for _, ev in wind_events.iterrows():
    alpha = 0.03 + 0.15 * float(ev["sev01"])   # ~0.03..0.18
    ax.axvspan(ev["start_utc"], ev["end_utc"], color="C3", alpha=alpha)

# Legend patches for events
event_patches = [
    Patch(facecolor="C0", alpha=0.15, label="Fog events (Boreas; alpha=severity)"),
    Patch(facecolor="C3", alpha=0.12, label="Wind events (Boreas; alpha=severity)"),
]

handles, labels = ax.get_legend_handles_labels()

# Add the patches and their labels
handles = handles + event_patches
labels = labels + [p.get_label() for p in event_patches]

ax.legend(handles=handles, labels=labels, loc="upper left")


ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Hourly Weather Risk Timeline — London")
ax.grid(True, alpha=0.3)

plt.tight_layout()

out = Path("eurus_lab/assets/hourly_risk_timeline.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
plt.show()

print(colored(f"Saved dashboard to: {out.resolve()}", "light_green"))
