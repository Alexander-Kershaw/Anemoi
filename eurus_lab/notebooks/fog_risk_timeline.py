from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load curated parquet for London (demo location), specify different location as needed in ingest_meteostat.py main()
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("timestamp_utc")

# Basic series
t = df["timestamp_utc"]
temp = df["temp"]
td = df["dwpt_c"]
fog = df["fog_risk"].fillna(0).astype(int)

# Plot: temperature & dewpoint, with fog-risk hours shaded
plt.figure()

plt.plot(t, temp, label="Temp (°C)")
plt.plot(t, td, label="Dew point (°C)")

# Shade fog-risk hours (vertical spans)
# This is accomplished by plotting a semi-transparent fill where fog==1 (fog binary flag is 1 for risk, 0 for no risk)
ymin = min(temp.min(), td.min())
ymax = max(temp.max(), td.max())
plt.fill_between(t, ymin, ymax, where=(fog == 1), alpha=0.2, label="Fog/low-cloud risk")

plt.xlabel("Time (UTC)")
plt.ylabel("°C")
plt.title("London: Temperature & Dewpoint with Fog/Low-Cloud Risk (shaded)")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
out_dir = Path("eurus_lab/notebooks/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "fog_risk_timeline.png", dpi=150)
plt.show()
