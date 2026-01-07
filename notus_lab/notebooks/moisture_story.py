from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load all parquet partitions for London (or whatever location specified in ingest_meteostat.py main())
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))

# Dataframe with all data sorted by timestamp
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("timestamp_utc")


# Dewpoint depression = temp - dew point (measure of humidity; lower = more humid)
df["dewpoint_depression_c"] = df["temp"] - df["dwpt_c"]

print("Basic stats:")
print(df[["temp", "dwpt_c", "rhum", "q_gkg", "dewpoint_depression_c"]].describe().round(2))

# Plot 1: Temperature and dew point time series
plt.figure()
plt.plot(df["timestamp_utc"], df["temp"], label="Temp (°C)")
plt.plot(df["timestamp_utc"], df["dwpt_c"], label="Dew point (°C)")
plt.xlabel("Time (UTC)")
plt.ylabel("°C")
plt.title("London: Temperature vs Dew Point (hourly)")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("notus_lab/notebooks/outputs/temp_vs_dewpoint.png", dpi=150)
plt.show()

# Plot 2: Dew point depression time series
plt.figure()
plt.plot(df["timestamp_utc"], df["dewpoint_depression_c"])
plt.xlabel("Time (UTC)")
plt.ylabel("Temp - Dew point (°C)")
plt.title("London: Dew Point Depression (lower = more humid)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("notus_lab/notebooks/outputs/dewpoint_depression.png", dpi=150)
plt.show()

# Plot 3: Scatter (Temp vs Dew point)
plt.figure()
plt.scatter(df["temp"], df["dwpt_c"], s=8)
plt.xlabel("Temperature (°C)")
plt.ylabel("Dew point (°C)")
plt.title("London: Temp vs Dew Point (hourly scatter)")
plt.tight_layout()
plt.savefig("notus_lab/notebooks/outputs/temp_vs_dewpoint_scatter.png", dpi=150)
plt.show()


