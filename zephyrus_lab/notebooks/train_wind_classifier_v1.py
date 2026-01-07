from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from termcolor import colored

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve

from boreas.wind_events import add_high_wind_flags

"""
See train_fog_classifier_v1 for operational details since all models
follow same architecture and logic
"""


# Load Notus curated data for desired location (London)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)


# Create label (Like from Boreas)
df = add_high_wind_flags(
    df,
    speed_col="wspd_ms",
    gust_col="wpgt",
    speed_threshold_ms=10.0,  # sustained
    gust_threshold_ms=15.0,   # gust
)

target = "high_wind_v1"
df = df.dropna(subset=[target]).copy()
y = df[target].astype(int)


# Features (These features are influencers of wind BUT not wind metrics in itself, make it more meaningful)
# Features directly referencing wind would cause leakage
features = features = ["pres", "dpres_1h_hpa", "temp", "rhum", "cldc", "prcp", "is_rain_hour",]

features = [c for c in features if c in df.columns]

X = df[features]


# Time-aware split (last 20% of time used for testing)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
t_train = df["timestamp_utc"].iloc[:split_idx]
t_test = df["timestamp_utc"].iloc[split_idx:]

print(colored(f"Train period: {t_train.iloc[0]} → {t_train.iloc[-1]}", "yellow"))
print(colored(f"Test  period: {t_test.iloc[0]} → {t_test.iloc[-1]}", "yellow"))
print(colored(f"Train size: {len(X_train)}, Test size: {len(X_test)}", "yellow"))
print(colored(f"Test positive rate ({target}): {y_test.mean():.3f}", "yellow"))


# Logistic Regression baseline
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), features)
    ],
    remainder="drop"
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

pipe.fit(X_train, y_train)

p_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (p_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, p_test)
ap = average_precision_score(y_test, p_test)

print(colored("\n=== Metrics (test) ===", "blue"))
print(f"ROC AUC: {auc:.3f}")
print(f"Avg Precision (PR AUC): {ap:.3f}")
print(colored("\nClassification report @0.5 threshold:", "blue"))
print(classification_report(y_test, y_pred, digits=3))
print(colored("Confusion matrix:", "blue"))
print(confusion_matrix(y_test, y_pred))


# Baselines: climatology + persistence
p_clim = y_train.mean()
p_test_clim = [p_clim] * len(y_test)

print(colored("\n=== Baseline: Climatology ===", "blue"))
print(f"ROC AUC: {roc_auc_score(y_test, p_test_clim):.3f}")
print(f"Avg Precision: {average_precision_score(y_test, p_test_clim):.3f}")

y_full = y.reset_index(drop=True)
y_test_persist = y_full.shift(1).iloc[split_idx:].fillna(0).astype(int)

print(colored("\n=== Baseline: Persistence ===", "blue"))
print(f"ROC AUC: {roc_auc_score(y_test, y_test_persist):.3f}")
print(f"Avg Precision: {average_precision_score(y_test, y_test_persist):.3f}")


# Reports/plots
out_dir = Path("zephyrus_lab/reports")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure()
RocCurveDisplay.from_predictions(y_test, p_test)
plt.title("High Wind v1 — ROC Curve (LogReg baseline)")
plt.tight_layout()
plt.savefig(out_dir / "wind_v1_roc.png", dpi=150)
plt.show()

plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, p_test)
plt.title("High Wind v1 — Precision-Recall Curve (LogReg baseline)")
plt.tight_layout()
plt.savefig(out_dir / "wind_v1_pr.png", dpi=150)
plt.show()

# Calibration
prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="uniform")
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("High Wind v1 — Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "wind_v1_calibration.png", dpi=150)
plt.show()

# Threshold sweep
print(colored("\n=== Threshold sweep ===", "blue"))
print("thr | precision | recall | false_alarm_rate")
for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
    y_hat = (p_test >= thr).astype(int)
    tp = ((y_hat == 1) & (y_test == 1)).sum()
    fp = ((y_hat == 1) & (y_test == 0)).sum()
    fn = ((y_hat == 0) & (y_test == 1)).sum()
    tn = ((y_hat == 0) & (y_test == 0)).sum()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    far = fp / (fp + tn) if fp + tn > 0 else 0.0
    print(f"{thr:0.2f} | {precision:0.3f}     | {recall:0.3f} | {far:0.3f}")

# Save model
model_dir = Path("zephyrus_lab/models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": pipe, "features": features, "target": target},
    model_dir / "wind_classifier_logreg_v1.joblib"
)
print(colored("\nSaved model to: {(model_dir / 'wind_classifier_logreg_v1.joblib').resolve()}", "green"))
