"""
Flight Delay Prediction — Classification Training Pipeline
==========================================================
Pipeline: Pre-flight features (203 cols)
          -> StandardScaler
          -> SelectKBest ANOVA (k=30)
          -> DBSCAN (eps=0.4, min_samples=10) -> 818 clusters
          -> Per-cluster RandomForestClassifier (507 models)
          -> KNN cluster-assigner for inference routing
          -> Global HistGradientBoosting fallback
          -> Soft blend: 80% cluster + 20% global

Test accuracy: ~73.4% (80/20 stratified split)

Why no post-flight features?
  TAXI_OUT, ACTUAL_ELAPSED_TIME, CARRIER_DELAY are only known AFTER a
  flight lands. Using them = data leakage. We use only features known
  at booking/check-in time.

After running this, also run: python train_regressor.py
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "final_preprocessed_data.csv"
MODEL_DIR  = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── feature split ─────────────────────────────────────────────────────────────
# These columns are measured AFTER the flight operates — never use for inference
POST_FLIGHT = [
    "Unnamed: 0",
    "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
TARGET = "FLIGHT_STATUS"

# ── config ────────────────────────────────────────────────────────────────────
TOP_K_FEATURES  = 30    # ANOVA keeps the 30 most discriminative features
DBSCAN_EPS      = 0.4   # neighbourhood radius for DBSCAN
DBSCAN_MIN_SAMP = 10    # minimum points to form a cluster
MIN_CLUSTER_SIZE = 20   # skip clusters too small to train on meaningfully
RF_ESTIMATORS    = 300  # more trees → lower variance
RANDOM_STATE     = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(data):,} rows × {data.shape[1]} columns")
print(f"  Class balance: {data[TARGET].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — keep only pre-flight features
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2 — Selecting pre-flight features (removing data leakage)")

drop_cols = [c for c in POST_FLIGHT if c in data.columns] + [TARGET]
X_raw = data.drop(columns=drop_cols)
y     = data[TARGET]

# Drop columns that are constant across the whole dataset (carry zero information)
constant_cols = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
if constant_cols:
    print(f"  Dropping {len(constant_cols)} constant column(s): {constant_cols}")
    X_raw = X_raw.drop(columns=constant_cols)

print(f"  Features after cleanup: {X_raw.shape[1]}")

# Save the feature names so the API knows what columns to expect
feature_names = list(X_raw.columns)
joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3 — Train/test split (80/20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. STANDARDISE
#    Neural networks need this; Random Forests don't strictly, but DBSCAN does.
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4 — Standardising features (zero mean, unit variance)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(X_test)         # apply same scale to test

joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
print("  Saved scaler -> model/scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE SELECTION — ANOVA F-test
#    SelectKBest scores each feature independently against the target.
#    Higher F-score -> more discriminative. We keep top K.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSTEP 5 — ANOVA feature selection (keeping top {TOP_K_FEATURES})")

selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel  = selector.transform(X_test_scaled)

selected_mask  = selector.get_support()
selected_feats = [f for f, m in zip(feature_names, selected_mask) if m]
print(f"  Selected features: {selected_feats}")

joblib.dump(selector, MODEL_DIR / "selector.pkl")
print("  Saved selector -> model/selector.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLUSTERING — DBSCAN
#    DBSCAN groups similar flights together. Each cluster gets its own model,
#    which captures the unique delay patterns of that flight type.
#
#    Problem: DBSCAN has no .predict() for new points.
#    Solution: after clustering, train a KNN to map new points -> cluster id.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSTEP 6 — DBSCAN clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMP})")

dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMP)
train_clusters = dbscan.fit_predict(X_train_sel)

unique_clusters = set(train_clusters)
n_clusters      = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
n_noise         = (train_clusters == -1).sum()
print(f"  Clusters found : {n_clusters}")
print(f"  Noise points   : {n_noise:,} ({100*n_noise/len(train_clusters):.1f}%)")

# Train a KNN cluster-assigner so new points can be routed to a cluster
# Points predicted as -1 (noise) will use a global fallback model
knn_assigner = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_assigner.fit(X_train_sel, train_clusters)
joblib.dump(knn_assigner, MODEL_DIR / "knn_assigner.pkl")
print("  Saved KNN cluster-assigner -> model/knn_assigner.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PER-CLUSTER RANDOM FOREST TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSTEP 7 — Training per-cluster RandomForest models")

cluster_models = {}
cluster_stats  = {}

for cluster_id in sorted(unique_clusters):
    mask      = train_clusters == cluster_id
    X_c       = X_train_sel[mask]
    y_c       = y_train.values[mask]
    n_samples = len(X_c)

    if n_samples < MIN_CLUSTER_SIZE:
        continue  # too small, skip — falls back to global model at inference
    if len(np.unique(y_c)) < 2:
        continue  # only one class — nothing to learn

    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=20,              # limit depth to reduce over-fitting in small clusters
        min_samples_leaf=2,        # smoother probability estimates
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",   # handles any residual imbalance within cluster
    )
    rf.fit(X_c, y_c)
    cluster_models[cluster_id] = rf

    # Quick train accuracy for logging
    train_acc = accuracy_score(y_c, rf.predict(X_c))
    cluster_stats[cluster_id] = {"n": n_samples, "train_acc": train_acc}

print(f"  Trained {len(cluster_models)} cluster models "
      f"(skipped small/single-class clusters)")

joblib.dump(cluster_models, MODEL_DIR / "cluster_models.pkl")
print("  Saved cluster models -> model/cluster_models.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 8. GLOBAL FALLBACK MODEL
#    Used when a new point is assigned to cluster -1 (noise) or a cluster
#    that was too small to train on.
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 8 — Training global fallback HistGradientBoosting (for noise/unknown clusters)")

# HistGradientBoostingClassifier is sklearn's native gradient-boosted trees.
# It handles large datasets efficiently and consistently outperforms plain RF.
global_rf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
global_rf.fit(X_train_sel, y_train)
joblib.dump(global_rf, MODEL_DIR / "global_model.pkl")
print("  Saved global model -> model/global_model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION ON HELD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 9 — Evaluating on test set")

all_preds = []
all_probs = []

# Batch-assign clusters for the entire test set (much faster than per-row)
test_cluster_ids = knn_assigner.predict(X_test_sel)
global_probs     = global_rf.predict_proba(X_test_sel)[:, 1]   # global P(delayed)

for i, x in enumerate(X_test_sel):
    cluster_id = int(test_cluster_ids[i])
    if cluster_id in cluster_models:
        cluster_prob = float(cluster_models[cluster_id].predict_proba([x])[0][1])
        # Soft blend: 80% cluster model + 20% global model
        prob = 0.80 * cluster_prob + 0.20 * float(global_probs[i])
    else:
        prob = float(global_probs[i])

    pred = int(prob >= 0.5)
    all_preds.append(pred)
    all_probs.append(prob)

overall_acc = accuracy_score(y_test, all_preds)
print(f"\n  Overall Test Accuracy : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, all_preds, target_names=["On-Time", "Delayed"]))

# Save metadata (feature list, selected features, accuracy) for the API
metadata = {
    "overall_accuracy"  : overall_acc,
    "n_clusters"        : n_clusters,
    "n_cluster_models"  : len(cluster_models),
    "selected_features" : selected_feats,
    "all_features"      : feature_names,
    "top_k"             : TOP_K_FEATURES,
    "dbscan_eps"        : DBSCAN_EPS,
    "dbscan_min_samples": DBSCAN_MIN_SAMP,
}
joblib.dump(metadata, MODEL_DIR / "metadata.pkl")
print("\n  Saved metadata -> model/metadata.pkl")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE — all artifacts saved to model/")
print("=" * 60)
