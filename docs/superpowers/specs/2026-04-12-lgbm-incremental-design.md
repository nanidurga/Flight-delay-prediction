# Sprint 7 — LightGBM End-to-End + Incremental Learning
**Date:** 2026-04-12  
**Student:** 21MA23002  
**Status:** Approved, ready for implementation

---

## Goal

Replace the DBSCAN + KNN + 507-cluster RF classification pipeline and HistGBR regression pipeline with a single LightGBM-based architecture that:
1. Achieves higher classification accuracy (target 78–82%) and lower regression MAE (target <28 min)
2. Supports incremental/online learning — the model warm-starts on new labeled flight outcomes as they accumulate
3. Is extendable for cloud deployment (Render + GitHub Actions nightly retrain as documented in `CLOUD_DEPLOYMENT_GUIDE.html`)

---

## What Is Being Replaced

| Old component | Why replaced |
|---|---|
| DBSCAN + KNN cluster assigner | No `.predict()` for new points; fragments signal into 507 small models |
| SelectKBest ANOVA (k=30) | Throws away 173 of 203 features; LightGBM handles all features natively |
| 507 per-cluster RandomForestClassifiers | Each trained on ~135 rows avg; high variance, low capacity |
| Global HistGradientBoostingClassifier | Replaced by single LGBMClassifier on full feature set |
| StandardScaler | Not needed for tree-based models |
| HistGradientBoostingRegressor suite | Replaced by LGBMRegressor suite with same quantile + per-type structure |
| `hist_feature_lookup.pkl` (7 features) | Superseded by `feature_engineering.pkl` (15 features) |

**Artifacts removed:** `scaler.pkl`, `selector.pkl`, `knn_assigner.pkl`, `cluster_models.pkl`, `global_model.pkl`, `hist_feature_lookup.pkl`, `delay_type_props.pkl`

---

## Section 1 — Feature Engineering

All features computed **leak-free from the training fold only**. Lookup tables saved to `model/feature_engineering.pkl` and used identically at inference time via the shared `api/services/feature_engineering.py` module.

### Existing features kept (203 pre-flight columns)
All columns except `POST_FLIGHT` list and `FLIGHT_STATUS` target. No ANOVA filtering — LightGBM uses all 203 natively.

### New engineered features (~15 additional)

**Route-pair features** (origin_city × destination_city):
| Feature | Description |
|---|---|
| `route_hist_mean_delay` | Avg total delay on this exact origin→dest route (training fold, delayed flights only) |
| `route_hist_delay_rate` | P(delayed) on this route |
| `route_hist_n_flights` | Flight volume on this route (busy route proxy) |

**Cross-group interaction features:**
| Feature | Description |
|---|---|
| `carrier_origin_hist_mean_delay` | Avg delay for this carrier departing from this origin city |
| `carrier_month_hist_mean_delay` | Avg delay for this carrier in this month |
| `depslot_origin_hist_mean_delay` | Avg delay at (origin city, departure slot) pair |

**Expanded single-group features** (replacing the 7 from v2):
| Feature | Description |
|---|---|
| `carrier_hist_mean_delay` | Avg delay for this carrier |
| `carrier_hist_delay_rate` | P(delayed) for this carrier |
| `origin_hist_mean_delay` | Avg delay from this origin city |
| `origin_hist_delay_rate` | P(delayed) from this origin city |
| `dest_hist_mean_delay` | Avg delay to this destination city |
| `depslot_hist_mean_delay` | Avg delay at this departure slot |
| `month_hist_mean_delay` | Avg delay in this month |

**Weather severity scores** (2 features):
| Feature | Description |
|---|---|
| `origin_weather_severity` | Ordinal 0–8: Sunny=0 … Heavy rain/thunder=8 (collapses 16 binary cols) |
| `dest_weather_severity` | Same for destination |

**Total feature count:** 203 + 15 = **~218 features**

### Shared feature engineering module
`api/services/feature_engineering.py` — `FeatureEngineer` class with:
- `fit(X_train, y_total_delay, y_binary_status)` — computes all lookup tables from training fold (`y_total_delay` = sum of delay minutes, `y_binary_status` = 0/1 FLIGHT_STATUS label)
- `transform(X)` — applies lookup tables to any dataset (train, test, or single inference row)
- Serialized to `model/feature_engineering.pkl`

Fallback for unseen groups at inference: global historical mean from training fold.

---

## Section 2 — Model Architecture

### Classification

**Model:** `LGBMClassifier` + `CalibratedClassifierCV(method='isotonic', cv='prefit')`

```python
LGBMClassifier(
    n_estimators          = 1000,
    learning_rate         = 0.05,
    max_depth             = 8,
    num_leaves            = 63,
    min_child_samples     = 20,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.1,
    reg_lambda            = 0.1,
    class_weight          = 'balanced',
    early_stopping_rounds = 50,
    random_state          = 42,
    n_jobs                = -1,
)
```

Calibration uses `cv='prefit'`: the LGBMClassifier is first trained on the training fold, then `CalibratedClassifierCV(cv='prefit').fit(X_val, y_val)` calibrates it on a separate 20% validation slice. This avoids refitting LightGBM 5 times (which would be very slow on 86k rows). The calibrated wrapper is what gets saved and used for inference.

**Inference output:** `probability`, `delayed` (threshold 0.5), `confidence` (high/medium/low by distance from 0.5)

### Regression

Same structural approach as v2 — four model files:

| Model | Purpose |
|---|---|
| `LGBMRegressor` (main) | Point estimate, `objective='regression'`, target=`log1p(total_delay)` |
| `LGBMRegressor` (p10) | 10th percentile, `objective='quantile'`, `alpha=0.10` |
| `LGBMRegressor` (p90) | 90th percentile, `objective='quantile'`, `alpha=0.90` |
| 4× `LGBMRegressor` | Per-type (carrier/weather/nas/late_aircraft), rescaled to sum to point estimate |

All regressors trained on **delayed flights only** (same as v2). Targets log1p-transformed; predictions back-transformed with expm1.

```python
# Shared base params for all regressors
LGBMRegressor(
    n_estimators          = 1000,
    learning_rate         = 0.04,
    max_depth             = 8,
    num_leaves            = 63,
    min_child_samples     = 20,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.1,
    reg_lambda            = 0.1,
    early_stopping_rounds = 50,
    random_state          = 42,
    n_jobs                = -1,
)
# Per-type regressors: n_estimators=500, max_depth=6
```

---

## Section 3 — Training Pipeline & File Structure

### Files changed / added

| File | Status | Notes |
|---|---|---|
| `train_lgbm.py` | **New** (replaces `train.py` + `train_regressor.py`) | Full from-scratch training |
| `train_incremental.py` | **New** | Warm-start update on feedback rows |
| `api/services/feature_engineering.py` | **New** | Shared FeatureEngineer class |
| `api/services/predictor.py` | **Updated** | Loads LightGBM artifacts |
| `api/main.py` | **Updated** | Adds `POST /feedback` endpoint |
| `predict.py` | **Updated** | Uses FeatureEngineer, loads LGBM models |
| `data/feedback.csv` | **New** | Accumulates actual flight outcomes |
| `.github/workflows/retrain.yml` | **New** | Nightly GitHub Actions job |

### `train_lgbm.py` flow

```
1. Load CSV (86,478 rows)
2. Drop post-flight columns → 203 pre-flight features
3. 80/20 stratified split (seed=42)
4. FeatureEngineer.fit(X_train, y_train) → compute all lookup tables
5. FeatureEngineer.transform(X_train/X_test) → ~218 features
6. Train LGBMClassifier (early stopping on 10% val split)
7. Wrap with CalibratedClassifierCV → save lgbm_clf_calibrated.pkl
8. Filter to delayed rows → train LGBMRegressor main + p10/p90
9. Train 4 per-type LGBMRegressors
10. Evaluate all models on held-out test set
11. Save all artifacts to model/ + metadata.pkl
```

### `train_incremental.py` flow

```
1. Load data/feedback.csv
2. If rows < RETRAIN_THRESHOLD (500): exit early
3. FeatureEngineer.transform(feedback_rows) using saved feature_engineering.pkl
4. lgbm_clf.fit(X_new, y_new, init_model='model/lgbm_clf.pkl')
5. lgbm_reg.fit(X_new_delayed, y_log, init_model='model/lgbm_reg.pkl')
6. Evaluate on held-out test set → compare to metadata accuracy
7. If accuracy drops > 2%: abort + log warning, keep old model
8. Save versioned copy: model/lgbm_clf_v{N}.pkl (keep last 3 versions)
9. Overwrite model/lgbm_clf.pkl (and reg equivalent)
10. Archive processed feedback rows (move to feedback_archive.csv)
```

### New model artifact layout

```
model/
├── lgbm_clf.pkl                # LGBMClassifier (raw)
├── lgbm_clf_calibrated.pkl     # CalibratedClassifierCV (used for inference)
├── lgbm_reg.pkl                # Main LGBMRegressor
├── lgbm_reg_p10.pkl            # 10th percentile
├── lgbm_reg_p90.pkl            # 90th percentile
├── lgbm_type_regressors.pkl    # Dict: {carrier, weather, nas, late_aircraft}
├── feature_engineering.pkl     # Fitted FeatureEngineer state
├── feature_names.pkl           # 203 pre-flight column names (unchanged)
└── metadata.pkl                # Metrics + version + trained_at timestamp
```

---

## Section 4 — Incremental Learning & API Changes

### Feedback endpoint

```
POST /feedback
Body: {
  "flight_id": "UA123_2026-04-12",
  "actual_delayed": true,
  "actual_delay_min": 87
}
Response: {"status": "recorded", "feedback_count": 1243}
```

**Validation rules:**
- `actual_delay_min` ∈ [0, 800]
- If `actual_delay_min > 15` then `actual_delayed` must be `true`
- Duplicate `flight_id` silently ignored (idempotent)

### GitHub Actions workflow (nightly)

```yaml
# .github/workflows/retrain.yml
on:
  schedule:
    - cron: '0 2 * * *'   # 2 AM UTC nightly
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - checkout (with lfs: true)
      - setup Python 3.11
      - pip install scikit-learn lightgbm joblib numpy pandas
      - run: python train_incremental.py
      - commit updated model files + feedback_archive
      - trigger Render redeploy via secrets.RENDER_DEPLOY_HOOK_URL
```

When feedback rows < 500: `train_incremental.py` exits cleanly, Actions job succeeds with no-op. Full retrain (`train_lgbm.py`) is run manually or on a monthly schedule.

### `/model/info` response after sprint

```json
{
  "model_version": 3,
  "trained_at": "2026-04-12T02:14:33Z",
  "accuracy": 0.801,
  "roc_auc": 0.876,
  "regression_mae": 26.4,
  "regression_r2": 0.714,
  "pi_80_coverage": 79.3,
  "n_features": 218,
  "model_type": "LGBMClassifier + LGBMRegressor",
  "incremental_updates": 0,
  "feedback_rows_used": 0
}
```

---

## Evaluation Targets

| Metric | Current (v2) | Target (Sprint 7) |
|---|---|---|
| Classification accuracy | 73.43% | **78–82%** |
| ROC-AUC | not tracked | **>0.85** |
| Regression MAE | 35.4 min | **<28 min** |
| Regression Median AE | 18.1 min | **<15 min** |
| R² | 0.633 | **>0.70** |
| 80% PI coverage | 78% | **78–82%** |

---

## Error Handling

- **Model load failure:** `predictor.py` raises `RuntimeError` at startup; `/predict` returns HTTP 503
- **Incremental accuracy regression:** if new accuracy < old − 2%, discard new model, keep previous pkl, log warning
- **Unknown group at inference:** fall back to global historical mean (same as v2)
- **Feedback validation failure:** return HTTP 422 with descriptive error message

---

## Out of Scope for This Sprint

- Frontend changes (gauge, breakdown bar already handle the same response shape)
- Hyperparameter tuning via Optuna (fixed params above are well-motivated starting points; can be added in a future sprint)
- AviationStack paid integration (BTS monthly download is the default data source)
- Security delay type regressor (0.1% of delays; predicted as near-zero, same as v2)
