"""
train_lgbm.py — Sprint 7 full LightGBM training pipeline
=========================================================

Replaces train.py + train_regressor.py.

Pipeline:
  Raw CSV (86,478 rows, 219 cols)
  -> Drop post-flight features -> 203 pre-flight features remain
  -> 70/10/20 split: train / calibration-val / test  (stratified, seed=42)
  -> FeatureEngineer.fit(train) -> 218 features
  -> LGBMClassifier  (early stop on val)
  -> CalibratedClassifierCV(cv='prefit').fit(val)
  -> LGBMRegressor  main + p10/p90 + 4 per-type  (log1p target, delayed only)
  -> Evaluate on strictly held-out test set
  -> Save all artifacts to model/

Run: python train_lgbm.py   (takes ~5-10 min)
"""

import warnings
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "final_preprocessed_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from api.services.feature_engineering import FeatureEngineer

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                              mean_absolute_error, median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
POST_FLIGHT  = [
    "Unnamed: 0", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
TARGET     = "FLIGHT_STATUS"
DELAY_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
              "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
SEP = "=" * 65

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print(SEP); print("STEP 1 — Loading data"); print(SEP)

data = pd.read_csv(DATA_PATH)
data["total_delay"] = data[DELAY_COLS].sum(axis=1)
print(f"  Loaded {len(data):,} rows × {data.shape[1]} columns")

drop_cols = [c for c in POST_FLIGHT if c in data.columns] + [TARGET, "total_delay"]
X_raw     = data.drop(columns=drop_cols)
constant  = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
if constant:
    X_raw = X_raw.drop(columns=constant)
    print(f"  Dropped {len(constant)} constant columns")

feature_names = list(X_raw.columns)
print(f"  Pre-flight features: {len(feature_names)}")

y_status  = data[TARGET].values
y_total   = data["total_delay"].values
y_carrier = data["CARRIER_DELAY"].values
y_weather = data["WEATHER_DELAY"].values
y_nas     = data["NAS_DELAY"].values
y_late_ac = data["LATE_AIRCRAFT_DELAY"].values

joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

# ── 2. THREE-WAY SPLIT: 70% train / 10% val / 20% test ───────────────────────
print(f"\nSTEP 2 — Three-way split (70/10/20, stratified, seed={RANDOM_STATE})")

idx = np.arange(len(data))
idx_trainval, idx_test = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_status)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125,   # 0.125 × 0.80 ≈ 0.10 of total
    random_state=RANDOM_STATE, stratify=y_status[idx_trainval])

print(f"  Train: {len(idx_train):,}  |  Val: {len(idx_val):,}  |  Test: {len(idx_test):,}")

X_tr_df  = pd.DataFrame(X_raw.values[idx_train],  columns=feature_names)
X_val_df = pd.DataFrame(X_raw.values[idx_val],    columns=feature_names)
X_te_df  = pd.DataFrame(X_raw.values[idx_test],   columns=feature_names)

y_tr_status  = y_status[idx_train];   y_val_status = y_status[idx_val];   y_te_status = y_status[idx_test]
y_tr_total   = y_total[idx_train];    y_val_total  = y_total[idx_val];    y_te_total  = y_total[idx_test]
y_tr_carrier = y_carrier[idx_train];  y_val_carrier = y_carrier[idx_val];  y_te_carrier = y_carrier[idx_test]
y_tr_weather = y_weather[idx_train];  y_val_weather = y_weather[idx_val];  y_te_weather = y_weather[idx_test]
y_tr_nas     = y_nas[idx_train];      y_val_nas     = y_nas[idx_val];      y_te_nas     = y_nas[idx_test]
y_tr_late    = y_late_ac[idx_train];  y_val_late    = y_late_ac[idx_val];  y_te_late    = y_late_ac[idx_test]

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print(f"\nSTEP 3 — Feature engineering (fit on train fold only)")

fe = FeatureEngineer()
fe.fit(X_tr_df, y_tr_total, y_tr_status)

X_tr  = fe.transform(X_tr_df)
X_val = fe.transform(X_val_df)
X_te  = fe.transform(X_te_df)

print(f"  Feature shape: {X_tr.shape}  ({len(feature_names)} pre-flight + 15 engineered)")
joblib.dump(fe, MODEL_DIR / "feature_engineering.pkl")
print("  Saved -> model/feature_engineering.pkl")

# ── 4. CLASSIFICATION ─────────────────────────────────────────────────────────
print(f"\nSTEP 4 — LGBMClassifier (early stopping on val, then isotonic calibration)")

clf = LGBMClassifier(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    max_depth         = 8,
    num_leaves        = 63,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    class_weight      = "balanced",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)
clf.fit(
    X_tr, y_tr_status,
    eval_set=[(X_val, y_val_status)],
    callbacks=[early_stopping(50, verbose=False), log_evaluation(100)],
)
print(f"  Best iteration: {clf.best_iteration_}")
joblib.dump(clf, MODEL_DIR / "lgbm_clf.pkl")
print("  Saved raw classifier -> model/lgbm_clf.pkl")

# Calibrate on val set (cv='prefit': no refitting, just calibration layer)
cal_clf = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
cal_clf.fit(X_val, y_val_status)
joblib.dump(cal_clf, MODEL_DIR / "lgbm_clf_calibrated.pkl")
print("  Saved calibrated classifier -> model/lgbm_clf_calibrated.pkl")

# ── 5. REGRESSION ─────────────────────────────────────────────────────────────
print(f"\nSTEP 5 — Regression suite (delayed flights only, log1p target)")

tr_delayed  = y_tr_status == 1
val_delayed = y_val_status == 1
te_delayed  = y_te_status == 1

X_tr_d  = X_tr[tr_delayed];    X_val_d = X_val[val_delayed];   X_te_d = X_te[te_delayed]
y_tr_d  = y_tr_total[tr_delayed];  y_val_d = y_val_total[val_delayed]; y_te_d = y_te_total[te_delayed]
y_tr_log  = np.log1p(y_tr_d)
y_val_log = np.log1p(y_val_d)

print(f"  Delayed rows — train: {tr_delayed.sum():,}  val: {val_delayed.sum():,}  test: {te_delayed.sum():,}")

REG_BASE = dict(
    n_estimators=1000, learning_rate=0.04, max_depth=8, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)

# Main point-estimate regressor
reg_main = LGBMRegressor(objective="regression", **REG_BASE)
reg_main.fit(X_tr_d, y_tr_log,
             eval_set=[(X_val_d, y_val_log)],
             callbacks=[early_stopping(50, verbose=False), log_evaluation(100)])
preds = np.clip(np.expm1(reg_main.predict(X_te_d)), 1, 800)
mae   = mean_absolute_error(y_te_d, preds)
medae = median_absolute_error(y_te_d, preds)
r2    = r2_score(y_te_d, preds)
print(f"  Main regressor — MAE: {mae:.1f} min | Median AE: {medae:.1f} | R²: {r2:.3f}")
joblib.dump(reg_main, MODEL_DIR / "lgbm_reg.pkl")

# Quantile regressors (80% prediction interval)
Q_BASE = {**REG_BASE, "n_estimators": 600, "max_depth": 6}
reg_p10 = LGBMRegressor(objective="quantile", alpha=0.10, **Q_BASE)
reg_p90 = LGBMRegressor(objective="quantile", alpha=0.90, **Q_BASE)
reg_p10.fit(X_tr_d, y_tr_log, eval_set=[(X_val_d, y_val_log)],
            callbacks=[early_stopping(30, verbose=False)])
reg_p90.fit(X_tr_d, y_tr_log, eval_set=[(X_val_d, y_val_log)],
            callbacks=[early_stopping(30, verbose=False)])
p10 = np.clip(np.expm1(reg_p10.predict(X_te_d)), 0, 800)
p90 = np.clip(np.expm1(reg_p90.predict(X_te_d)), 0, 800)
coverage = float(np.mean((y_te_d >= p10) & (y_te_d <= p90)))
print(f"  80% PI coverage: {coverage*100:.1f}%  (target ~80%)")
joblib.dump(reg_p10, MODEL_DIR / "lgbm_reg_p10.pkl")
joblib.dump(reg_p90, MODEL_DIR / "lgbm_reg_p90.pkl")

# Per-type regressors
TYPE_BASE = {**REG_BASE, "n_estimators": 500, "max_depth": 6}
type_targets = {
    "carrier":       (y_tr_carrier[tr_delayed],   y_val_carrier[val_delayed],   y_te_carrier[te_delayed]),
    "weather":       (y_tr_weather[tr_delayed],   y_val_weather[val_delayed],   y_te_weather[te_delayed]),
    "nas":           (y_tr_nas[tr_delayed],        y_val_nas[val_delayed],       y_te_nas[te_delayed]),
    "late_aircraft": (y_tr_late[tr_delayed],       y_val_late[val_delayed],      y_te_late[te_delayed]),
}
type_regressors = {}
for name, (y_tr_t, y_val_t, y_te_t) in type_targets.items():
    rg = LGBMRegressor(objective="regression", **TYPE_BASE)
    rg.fit(X_tr_d, np.log1p(y_tr_t),
           eval_set=[(X_val_d, np.log1p(y_val_t))],
           callbacks=[early_stopping(30, verbose=False)])
    tp  = np.clip(np.expm1(rg.predict(X_te_d)), 0, 800)
    t_mae   = mean_absolute_error(y_te_t, tp)
    t_medae = median_absolute_error(y_te_t, tp)
    print(f"  {name:20s}: MAE={t_mae:.1f} min | Median AE={t_medae:.1f}")
    type_regressors[name] = rg
joblib.dump(type_regressors, MODEL_DIR / "lgbm_type_regressors.pkl")

# ── 6. EVALUATE CLASSIFIER ────────────────────────────────────────────────────
print(f"\nSTEP 6 — Final classifier evaluation on held-out test set")

probs_cal = cal_clf.predict_proba(X_te)[:, 1]
preds_bin = (probs_cal >= 0.5).astype(int)
accuracy  = accuracy_score(y_te_status, preds_bin)
roc_auc   = roc_auc_score(y_te_status, probs_cal)

print(f"\n  Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(classification_report(y_te_status, preds_bin, target_names=["On-Time", "Delayed"]))

# ── 7. SAVE METADATA ──────────────────────────────────────────────────────────
metadata = {
    # New fields
    "model_version"        : 3,
    "trained_at"           : datetime.now(timezone.utc).isoformat(),
    "model_type"           : "LGBMClassifier + LGBMRegressor",
    "overall_accuracy"     : accuracy,
    "roc_auc"              : roc_auc,
    "regression_mae"       : mae,
    "regression_median_ae" : medae,
    "regression_r2"        : r2,
    "pi_80_coverage"       : coverage,
    "n_features"           : X_tr.shape[1],
    "feature_names"        : feature_names,
    "incremental_updates"  : 0,
    "feedback_rows_used"   : 0,
    # Legacy keys kept for backward compat with /model/info
    "n_clusters"           : 0,
    "n_cluster_models"     : 0,
    "selected_features"    : feature_names[:30],
    "top_k"                : len(feature_names),
    "dbscan_eps"           : 0,
    "dbscan_min_samples"   : 0,
}
joblib.dump(metadata, MODEL_DIR / "metadata.pkl")
print(f"\n  Saved metadata -> model/metadata.pkl")
print(f"\n{SEP}")
print("  TRAINING COMPLETE — artifacts saved to model/")
print(SEP)
