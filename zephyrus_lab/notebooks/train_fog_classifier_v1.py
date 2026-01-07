from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

import joblib


"""
Note that Zephyrus does not rely on Boreas outputs for labels, instead derives labels
directly from the Notus curated data using the same Boreas logic
This keeps everything singular source and resilient
"""

# Load Notus curated data (for location: London)
base = Path("notus_lab/data/curated/meteostat_hourly/location_name=London")
files = sorted(base.rglob("hourly.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Ensure temporal data is timestamp_utc and sort
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# Target is fog_risk_v2 (what our actual metrics suggest as actual fog risk)
target = "fog_risk_v2"
if target not in df.columns:
    raise RuntimeError(f"Missing target column: {target}")


# Feature set 
features = [
    "temp",
    "rhum",
    "pres",
    "dpres_1h_hpa",
    "wspd_ms",
    "wdir",
    "dewpoint_depression_c",
    "q_gkg",
    "e_hpa",
    "is_rain_hour",
    "cldc",   
    "tsun",
]

# Keep only available columns
features = [c for c in features if c in df.columns]

# Keep only rows where the label exists
df = df.dropna(subset=[target]).copy()
y = df[target].astype(int)
X = df[features]

# Time-aware split
# Concerns for leakage so time aware approach is used
# Use last 20% of time as test
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

t_train = df["timestamp_utc"].iloc[:split_idx]
t_test = df["timestamp_utc"].iloc[split_idx:]

print(colored(f"Train period: {t_train.iloc[0]} → {t_train.iloc[-1]}", "yellow"))
print(colored(f"Test  period: {t_test.iloc[0]} → {t_test.iloc[-1]}", "yellow"))
print(colored(f"Train size: {len(X_train)}, Test size: {len(X_test)}", "yellow"))
print(colored(f"Test positive rate (fog_risk_v2): {y_test.mean():.3f}", "yellow"))


# Model: Logistic Regression baseline (chose for interpretibility reasons)
numeric_features = features

# apply different preprocessing to numeric columns, drop others (non numeric)
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")), # median imputation robust to outliers
            ("scaler", StandardScaler()), # standardize numeric values
        ]), numeric_features)
    ],
    remainder="drop"
)

# Establish Logistic regression (class balanced to improve feature weights)
clf = LogisticRegression(max_iter=2000, class_weight="balanced") 

# apply prepocessing then logistic regression in order
pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", clf),
    ])

pipe.fit(X_train, y_train)

# Probabilities
p_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (p_test >= 0.5).astype(int)


# Metrics
auc = roc_auc_score(y_test, p_test)
ap = average_precision_score(y_test, p_test)

print(colored("\n=== Metrics (test) ===", "blue"))
print(f"ROC AUC: {auc:.3f}")
print(f"Avg Precision (PR AUC): {ap:.3f}")
print(colored("\nClassification report @0.5 threshold:", "blue"))
print(classification_report(y_test, y_pred, digits=3))
print(colored("Confusion matrix:", "blue"))
print(confusion_matrix(y_test, y_pred))

# Threshold sweep (show tradeoff between recall and precision)
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
    false_alarm_rate = fp / (fp + tn) if fp + tn > 0 else 0.0

    print(f"{thr:0.2f} | {precision:0.3f}     | {recall:0.3f} | {false_alarm_rate:0.3f}")


# Baselines
# Climatology baseline
p_clim = y_train.mean()
p_test_clim = [p_clim] * len(y_test)

auc_clim = roc_auc_score(y_test, p_test_clim)
ap_clim = average_precision_score(y_test, p_test_clim)

print(colored("\n=== Baseline: Climatology ===", "blue"))
print(f"ROC AUC: {auc_clim:.3f}")
print(f"Avg Precision: {ap_clim:.3f}")

# Persistence baseline
y_test_persist = y.shift(1).iloc[split_idx:].fillna(0).astype(int)

auc_persist = roc_auc_score(y_test, y_test_persist)
ap_persist = average_precision_score(y_test, y_test_persist)

print(colored("\n=== Baseline: Persistence ===", "blue"))
print(f"ROC AUC: {auc_persist:.3f}")
print(f"Avg Precision: {ap_persist:.3f}")



# Plots 
out_dir = Path("zephyrus_lab/reports")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure()
RocCurveDisplay.from_predictions(y_test, p_test)
plt.title("Fog Risk v2 — ROC Curve (LogReg baseline)")
plt.tight_layout()
plt.savefig(out_dir / "fog_v2_roc.png", dpi=150)
plt.show()

plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, p_test)
plt.title("Fog Risk v2 — Precision-Recall Curve (LogReg baseline)")
plt.tight_layout()
plt.savefig(out_dir / "fog_v2_pr.png", dpi=150)
plt.show()


# Calibration plot (veryfy if probabilities are authentic)
prob_true, prob_pred = calibration_curve(
    y_test, p_test, n_bins=10, strategy="uniform"
)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Fog Risk v2 — Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "fog_v2_calibration.png", dpi=150)
plt.show()


# Probability timeline
plt.figure()
plt.plot(t_test, p_test)
plt.xlabel("Time (UTC)")
plt.ylabel("Predicted fog probability")
plt.title("Fog Risk v2 — Predicted Probability (test period)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(out_dir / "fog_v2_prob_timeline.png", dpi=150)
plt.show()

print(colored(f"\nSaved plots to: {out_dir.resolve()}", "green"))


# Save model artifact
model_dir = Path("zephyrus_lab/models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": pipe, "features": features, "target": target},
    model_dir / "fog_classifier_logreg_v1.joblib"
)

print(colored(f"Saved model to: {(model_dir / 'fog_classifier_logreg_v1.joblib').resolve()}", "light_cyan"))
