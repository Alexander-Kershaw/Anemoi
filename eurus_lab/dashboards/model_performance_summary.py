from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import calibration_curve


# Load curated Notus data (London)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
df = pd.concat([pd.read_parquet(f) for f in base.rglob("hourly.parquet")], ignore_index=True)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# Same time-aware split strategy as in the Zephyrus scripts
split_idx = int(len(df) * 0.8)


# Load Zephyrus models
fog_bundle = load("zephyrus_lab/models/fog_classifier_logreg_v1.joblib")
wind_bundle = load("zephyrus_lab/models/wind_classifier_logreg_v1.joblib")

fog_model = fog_bundle["model"]
wind_model = wind_bundle["model"]

fog_features = fog_bundle["features"]
wind_features = wind_bundle["features"]


# Build labels
# Fog uses notus label directly; dropping rows where missing
fog_df = df.dropna(subset=["fog_risk_v2"]).copy()
fog_df = fog_df.reset_index(drop=True)
fog_split = int(len(fog_df) * 0.8)

y_fog_test = fog_df["fog_risk_v2"].astype(int).iloc[fog_split:]
p_fog_test = fog_model.predict_proba(fog_df[fog_features].iloc[fog_split:])[:, 1]

# Wind label derived from Boreas logic (to match training)
from boreas.wind_events import add_high_wind_flags
wind_df = add_high_wind_flags(df.copy(), speed_col="wspd_ms", gust_col="wpgt", speed_threshold_ms=10.0, gust_threshold_ms=15.0)
wind_df = wind_df.dropna(subset=["high_wind_v1"]).reset_index(drop=True)
wind_split = int(len(wind_df) * 0.8)

y_wind_test = wind_df["high_wind_v1"].astype(int).iloc[wind_split:]
p_wind_test = wind_model.predict_proba(wind_df[wind_features].iloc[wind_split:])[:, 1]


# Plot panel (2x3)
fig, axes = plt.subplots(2, 3, figsize=(18, 9))

# Fog ROC
ax = axes[0, 0]
RocCurveDisplay.from_predictions(y_fog_test, p_fog_test, ax=ax)
ax.set_title("Fog (v2) — ROC")

# Fog PR
ax = axes[0, 1]
PrecisionRecallDisplay.from_predictions(y_fog_test, p_fog_test, ax=ax)
ax.set_title("Fog (v2) — Precision-Recall")

# Fog calibration
ax = axes[0, 2]
pt, pp = calibration_curve(y_fog_test, p_fog_test, n_bins=10, strategy="uniform")
ax.plot(pp, pt, marker="o", label="Fog model")
ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Fog (v2) — Calibration")
ax.legend()

# Wind ROC
ax = axes[1, 0]
RocCurveDisplay.from_predictions(y_wind_test, p_wind_test, ax=ax)
ax.set_title("Wind (v1) — ROC")

# Wind PR
ax = axes[1, 1]
PrecisionRecallDisplay.from_predictions(y_wind_test, p_wind_test, ax=ax)
ax.set_title("Wind (v1) — Precision-Recall")

# Wind calibration
ax = axes[1, 2]
pt, pp = calibration_curve(y_wind_test, p_wind_test, n_bins=10, strategy="uniform")
ax.plot(pp, pt, marker="o", label="Wind model")
ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Wind (v1) — Calibration")
ax.legend()

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.suptitle("Model Performance Summary — Zephyrus (Fog + Wind)", y=1.02)
plt.tight_layout()

out_dir = Path("eurus_lab/assets")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "model_performance_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved: {out.resolve()}")
