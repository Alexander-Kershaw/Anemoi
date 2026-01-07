from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from termcolor import colored
from zoneinfo import ZoneInfo

"""
# Fog Risk Animation with Nighttime Shading

This script creates an animated timeline showing temperature and dewpoint 
with fog/low-cloud risk highlighted, along with nighttime shading based on local time in London.
"""

# Load data fro m curated parquet files for London (demo location), change as needed in ingest_meteostat.py main()
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# Ensure expected columns exist
required = ["timestamp_utc", "temp", "dwpt_c", "fog_risk"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in curated data: {missing}")

t = df["timestamp_utc"]
temp = df["temp"].astype(float)
td = df["dwpt_c"].astype(float)
fog = df["fog_risk"].fillna(0).astype(int)


# Animation settings and parameters
out_dir = Path("eurus_lab/notebooks/outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# Window size: show last N hours in each frame
window_hours = 7 * 24   # 7 days
step_hours = 3          # advance 3 hours per frame (should be smoother animation)

# Convert to indices (data is hourly-ish)
window = window_hours
step = step_hours

# Frame start indices
starts = list(range(0, max(1, len(df) - window), step))
if not starts:
    starts = [0]

# Y bounds
ymin = float(np.nanmin([temp.min(), td.min()]))
ymax = float(np.nanmax([temp.max(), td.max()]))


# Build figure and animation
fig, ax = plt.subplots()
ax.set_ylabel("°C")
ax.set_xlabel("Time (UTC)")

line_temp, = ax.plot([], [], label="Temp (°C)")
line_td, = ax.plot([], [], label="Dew point (°C)")

# Simply shading handle for fog-risk areas and nighttime shading
night_shade = None
fog_shade = None

# Set static y-limits
ax.set_ylim(ymin - 1, ymax + 1)
ax.legend(loc="lower left")

title = ax.set_title("London: Temp & Dewpoint with Fog/Low-Cloud Risk (moving window)")

# Animation functions
def init():
    line_temp.set_data([], [])
    line_td.set_data([], [])
    return line_temp, line_td, title

# Update function for each frame
def update(frame_idx: int):
    global night_shade, fog_shade
    start = starts[frame_idx]
    end = min(start + window, len(df))

    x = t.iloc[start:end]
    y1 = temp.iloc[start:end]
    y2 = td.iloc[start:end]
    f = fog.iloc[start:end].to_numpy()

    line_temp.set_data(x, y1)
    line_td.set_data(x, y2)

    ax.set_xlim(x.iloc[0], x.iloc[-1])

    # Remove previous shadings
    if night_shade is not None:
        night_shade.remove()
        night_shade = None
    if fog_shade is not None:
        fog_shade.remove()
        fog_shade = None

    # Nighttime shading (local time)
    # Convert UTC timestamps to Europe/London local time (handles DST)
    x_local = x.dt.tz_convert(ZoneInfo("Europe/London"))
    hour_local = x_local.dt.hour
    is_night = (hour_local >= 21) | (hour_local <= 6)

    # Draw nighttime shading initially (lighter shade than fog)
    night_shade = ax.fill_between(
        x,
        ymin - 1,
        ymax + 1,
        where=is_night.to_numpy(),
        alpha=0.12,
        label=None,
    )

    # Draw fog-risk shading on top (stronger shade than night)
    fog_shade = ax.fill_between(
        x,
        ymin - 1,
        ymax + 1,
        where=(f == 1),
        alpha=0.30,
        label=None,
    )

    # Update title with window range
    title.set_text(f"London: {x.iloc[0].date()} → {x.iloc[-1].date()} (night shaded, fog-risk highlighted)")

    return line_temp, line_td, night_shade, fog_shade, title



anim = FuncAnimation(
    fig,
    update,
    frames=len(starts),
    init_func=init,
    interval=120,   # ms between frames (playback speed)
    blit=False
)


# Save outputs
mp4_path = out_dir / "fog_risk_timeline.mp4"
gif_path = out_dir / "fog_risk_timeline.gif"

print(colored(f"Saving MP4 to {mp4_path} ...", "light_green"))
anim.save(mp4_path, fps=15, dpi=150)

print(colored(f"Saving GIF to {gif_path} ...", "light_green"))
anim.save(gif_path, fps=15, dpi=120)

print(colored("Done.", "green"))
