"""
train_incremental.py — Warm-start LightGBM update on feedback rows
==================================================================

Reads data/feedback.csv (actual flight outcomes from POST /feedback).
If fewer than RETRAIN_THRESHOLD rows exist, exits cleanly (no-op).

When enough rows exist:
  1. Rebuilds the validation set from original data (same seed=42 split)
  2. Warm-starts LGBMClassifier with init_model (adds N_NEW_TREES new trees)
  3. Re-calibrates classifier on val set
  4. Safety gate: if accuracy drops > ACCURACY_FLOOR, rolls back + exits 1
  5. Saves versioned backup (lgbm_clf_v{N}.pkl), keeps last MAX_VERSIONS
  6. Overwrites lgbm_clf.pkl + lgbm_clf_calibrated.pkl
  7. Updates metadata.pkl
  8. Archives processed rows to data/feedback_archive.csv
  9. Resets data/feedback.csv to header-only

Run: python train_incremental.py
Exit code 0 = success or no-op.  Exit code 1 = rollback (accuracy degraded).
"""

import warnings
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "model"
DATA_DIR     = BASE_DIR / "data"
FEEDBACK_CSV = DATA_DIR / "feedback.csv"
ARCHIVE_CSV  = DATA_DIR / "feedback_archive.csv"
ORIG_DATA    = DATA_DIR / "final_preprocessed_data.csv"

RETRAIN_THRESHOLD = 500    # minimum feedback rows before retraining
N_NEW_TREES       = 100    # trees added on top of existing model
MAX_VERSIONS      = 3      # keep this many versioned backups
ACCURACY_FLOOR    = 0.02   # rollback if accuracy drops by more than this
RANDOM_STATE      = 42

sys.path.insert(0, str(BASE_DIR))
from api.services.feature_engineering import FeatureEngineer

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEP = "=" * 65
print(SEP); print("Incremental Model Update"); print(SEP)

# ── 1. CHECK FEEDBACK COUNT ───────────────────────────────────────────────────
if not FEEDBACK_CSV.exists():
    print("  No feedback.csv found — nothing to do.")
    sys.exit(0)

feedback_df = pd.read_csv(FEEDBACK_CSV)
n_new = len(feedback_df)
print(f"  Feedback rows available: {n_new}  (threshold: {RETRAIN_THRESHOLD})")

if n_new < RETRAIN_THRESHOLD:
    print(f"  Fewer than {RETRAIN_THRESHOLD} rows — skipping retrain (no-op).")
    sys.exit(0)

# ── 2. LOAD EXISTING ARTIFACTS ────────────────────────────────────────────────
print("\n  Loading existing artifacts...")
fe            = joblib.load(MODEL_DIR / "feature_engineering.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
metadata      = joblib.load(MODEL_DIR / "metadata.pkl")
clf_raw       = joblib.load(MODEL_DIR / "lgbm_clf.pkl")
old_accuracy  = metadata["overall_accuracy"]
print(f"  Current model accuracy: {old_accuracy:.4f}")

# ── 3. REBUILD VAL + TEST SETS FROM ORIGINAL DATA ────────────────────────────
print("\n  Rebuilding val/test sets from original data (same seed=42 split)...")

POST_FLIGHT = [
    "Unnamed: 0", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
TARGET     = "FLIGHT_STATUS"
DELAY_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
              "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]

orig       = pd.read_csv(ORIG_DATA)
y_status   = orig[TARGET].values
drop_cols  = [c for c in POST_FLIGHT if c in orig.columns] + [TARGET]
X_raw      = orig.drop(columns=drop_cols).reindex(columns=feature_names, fill_value=0)

idx = np.arange(len(orig))
idx_trainval, idx_test = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_status)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125, random_state=RANDOM_STATE,
    stratify=y_status[idx_trainval])

X_val_df = pd.DataFrame(X_raw.values[idx_val],  columns=feature_names)
X_te_df  = pd.DataFrame(X_raw.values[idx_test], columns=feature_names)
y_val_status = y_status[idx_val]
y_te_status  = y_status[idx_test]

X_val_eng = fe.transform(X_val_df)
X_te_eng  = fe.transform(X_te_df)
print(f"  Val set: {len(X_val_eng):,}  |  Test set: {len(X_te_eng):,}")

# ── 4. WARM-START CLASSIFIER UPDATE ──────────────────────────────────────────
print(f"\n  Warm-starting LGBMClassifier (+{N_NEW_TREES} trees)...")

new_clf = LGBMClassifier(
    n_estimators      = N_NEW_TREES,
    learning_rate     = 0.03,       # lower LR for fine-tuning pass
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
new_clf.fit(X_val_eng, y_val_status, init_model=clf_raw)

new_cal_clf = CalibratedClassifierCV(new_clf, cv="prefit", method="isotonic")
new_cal_clf.fit(X_val_eng, y_val_status)

# ── 5. SAFETY GATE ────────────────────────────────────────────────────────────
print("\n  Evaluating new model on held-out test set...")
new_preds    = (new_cal_clf.predict_proba(X_te_eng)[:, 1] >= 0.5).astype(int)
new_accuracy = accuracy_score(y_te_status, new_preds)
delta        = new_accuracy - old_accuracy

print(f"  Old: {old_accuracy:.4f}  →  New: {new_accuracy:.4f}  (Δ {delta:+.4f})")

if delta < -ACCURACY_FLOOR:
    print(f"\n  WARNING: accuracy dropped {abs(delta):.4f} > {ACCURACY_FLOOR} threshold.")
    print("  Rolling back — existing model unchanged.")
    sys.exit(1)

# ── 6. SAVE VERSIONED + CURRENT MODELS ───────────────────────────────────────
version = metadata.get("model_version", 1) + 1
print(f"\n  Saving model version {version}...")

# Prune old versioned backups
versioned = sorted(MODEL_DIR.glob("lgbm_clf_v*.pkl"))
while len(versioned) >= MAX_VERSIONS:
    versioned.pop(0).unlink()
    versioned = sorted(MODEL_DIR.glob("lgbm_clf_v*.pkl"))

joblib.dump(new_clf,     MODEL_DIR / f"lgbm_clf_v{version}.pkl")
joblib.dump(new_clf,     MODEL_DIR / "lgbm_clf.pkl")
joblib.dump(new_cal_clf, MODEL_DIR / "lgbm_clf_calibrated.pkl")
print(f"  Saved: lgbm_clf_v{version}.pkl, lgbm_clf.pkl, lgbm_clf_calibrated.pkl")

# ── 7. UPDATE METADATA ────────────────────────────────────────────────────────
metadata["model_version"]       = version
metadata["trained_at"]          = datetime.now(timezone.utc).isoformat()
metadata["overall_accuracy"]    = new_accuracy
metadata["incremental_updates"] = metadata.get("incremental_updates", 0) + 1
metadata["feedback_rows_used"]  = metadata.get("feedback_rows_used", 0) + n_new
joblib.dump(metadata, MODEL_DIR / "metadata.pkl")

# ── 8. ARCHIVE FEEDBACK AND RESET ─────────────────────────────────────────────
write_header = not ARCHIVE_CSV.exists()
feedback_df.to_csv(ARCHIVE_CSV, mode="a", header=write_header, index=False)
with open(FEEDBACK_CSV, "w", encoding="utf-8") as f:
    f.write("flight_id,timestamp,actual_delayed,actual_delay_min\n")

print(f"\n  Archived {n_new} feedback rows → data/feedback_archive.csv")
print(f"  data/feedback.csv reset to header only")
print(f"\n{SEP}")
print(f"  Incremental update complete — v{version}  accuracy: {new_accuracy:.4f}")
print(SEP)
