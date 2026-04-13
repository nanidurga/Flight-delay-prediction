# MTP — Flight Delay Prediction
**Student:** 21MA23002 | **Type:** Master's Thesis Project

Goal: Predict US flight delays using ML and serve predictions via a real-time website.

---

## Live Deployment (Sprint 8 — Deployed 2026-04-13)

| Service | URL | Platform |
|---------|-----|----------|
| Backend API | https://mtp-flight-api.onrender.com | Render (free tier, Python) |
| Frontend | https://mtp-flight-delay-exaj8moop-durgas-projects-9ea2fbeb.vercel.app | Vercel |
| GitHub Repo | https://github.com/nanidurga/Flight-delay-prediction | master branch |

**Render service ID:** `srv-d7eh888sfn5c73d07o1g`
**Vercel project ID:** `prj_nNItwgRGGIDD4udXJVkB7nnLRyAW`

The frontend `VITE_API_URL` env var is set to the Render backend URL.
Render auto-deploys on every push to `master`. Vercel auto-deploys via the GitHub integration.

To redeploy manually, trigger via `deploy_agent` MCP tools or the Render/Vercel dashboards.

---

## Folder Structure

```
MTP/                                  <- PROJECT ROOT (always work from here)
│
├── data/
│   ├── final_preprocessed_data.csv  # 86,478 rows · 219 cols · balanced 50/50
│   ├── feedback.csv                  # Accumulates POST /feedback outcomes (header-only at start)
│   └── feedback_archive.csv          # Archived feedback rows after each incremental update
│
├── model/                            # Trained artifacts — never edit manually
│   ├── lgbm_clf.pkl                  # LGBMClassifier (raw, used for warm-start)
│   ├── lgbm_clf_calibrated.pkl       # CalibratedClassifierCV (used for inference)
│   ├── lgbm_reg.pkl                  # LGBMRegressor — point estimate (log1p target)
│   ├── lgbm_reg_p10.pkl              # 10th-percentile quantile regressor
│   ├── lgbm_reg_p90.pkl              # 90th-percentile quantile regressor
│   ├── lgbm_type_regressors.pkl      # Dict: {carrier, weather, nas, late_aircraft}
│   ├── feature_engineering.pkl       # Fitted FeatureEngineer instance (15 engineered features)
│   ├── feature_names.pkl             # Ordered list of 203 pre-flight feature names
│   └── metadata.pkl                  # Metrics + version + trained_at + incremental counters
│
│   # Legacy artifacts (superseded by Sprint 7, kept on disk, not loaded):
│   ├── scaler.pkl, selector.pkl, knn_assigner.pkl, cluster_models.pkl
│   ├── global_model.pkl, hist_feature_lookup.pkl, delay_type_props.pkl
│   ├── delay_regressor.pkl, delay_regressor_p10.pkl, delay_regressor_p90.pkl
│   └── delay_type_regressors.pkl
│
├── api/
│   ├── main.py                       # FastAPI app — 8 endpoints (incl. POST /feedback)
│   ├── requirements.txt              # fastapi uvicorn httpx python-dotenv pydantic lightgbm
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── predictor.py              # Singleton loader — loads FlightPredictor once at startup
│       ├── feature_engineering.py    # Shared FeatureEngineer class (fit/transform, 15 features)
│       └── flights.py                # OpenSky live flights + Open-Meteo weather + feature builder
│
├── frontend/                         # React 18 + Vite + Tailwind (dark theme)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js                # Proxy /api -> http://localhost:8000
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.jsx                  # Entry point
│       ├── App.jsx                   # Router — 4 routes
│       ├── api.js                    # Axios client — all API calls
│       ├── index.css                 # Tailwind base + Leaflet dark fix
│       ├── components/
│       │   └── Navbar.jsx            # Sticky nav with mobile menu
│       └── pages/
│           ├── Home.jsx              # Predict page — form + result panel
│           ├── Dashboard.jsx         # Charts — monthly/airline/weather/time-of-day
│           ├── LiveMap.jsx           # Leaflet map — live OpenSky flights + predictions
│           └── About.jsx             # Pipeline explanation + model stats
│
├── train_lgbm.py                     # Sprint 7 full training (replaces train.py + train_regressor.py)
├── train_incremental.py              # Warm-start LightGBM update on feedback rows
├── predict.py                        # FlightPredictor class v3 (LightGBM, used by the API)
├── train.py                          # SUPERSEDED — old DBSCAN+RF classification training
├── train_regressor.py                # SUPERSEDED — old HistGBR regression training
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared pytest fixtures (sample_X, sample_y_*)
│   ├── test_feature_engineering.py   # 10 tests for FeatureEngineer
│   ├── test_predict.py               # 10 tests for FlightPredictor
│   └── test_api_feedback.py          # 6 tests for POST /feedback + /model/info
│
├── .github/
│   └── workflows/
│       └── retrain.yml               # Nightly incremental model update via GitHub Actions
│
├── notebooks/
│   └── newer_model.ipynb             # Original exploratory notebook (reference only)
├── MTP_codes/DurgaMTP/               # Original source dir (.venv lives here)
└── CLAUDE.md                         # This file
```

---

## ML Pipeline (Sprint 7 — Current)

### Classification (binary: delayed / on-time)

```
Raw CSV (219 cols)
  -> Drop post-flight features (data leakage) -> 203 pre-flight features remain
  -> 70/10/20 stratified split (seed=42): train / calibration-val / test
  -> FeatureEngineer.fit(train) -> 218 features (203 + 15 engineered)
  -> LGBMClassifier (n_estimators=1000, early stopping on val, class_weight='balanced')
  -> CalibratedClassifierCV(cv='prefit', method='isotonic').fit(val)
  -> Inference: lgbm_clf_calibrated.pkl -> P(delayed), confidence
```

**Test Accuracy: 91.34%** | **ROC-AUC: 0.973** (17,296 held-out samples, 70/10/20 split, seed=42)

### Regression (delay minutes — delayed flights only)

```
Delayed flights only
  -> Same 218 features (FeatureEngineer.transform)
  -> LGBMRegressor main (objective='regression', log1p target) -> point estimate
  -> LGBMRegressor p10 (objective='quantile', alpha=0.10) -> lower bound
  -> LGBMRegressor p90 (objective='quantile', alpha=0.90) -> upper bound
  -> 4x LGBMRegressor per-type (carrier, weather, nas, late_aircraft)
     -> rescaled so breakdown sums exactly to point estimate
```

**MAE: 30.8 min** | **Median AE: 14.7 min** | **R²: 0.704** | **80% PI coverage: 77%**

### Engineered Features (15 additional, leak-free from training fold)

Computed by `api/services/feature_engineering.py` → `FeatureEngineer` class:

| Feature | Description |
|---|---|
| `carrier_hist_mean_delay` | Avg delay for this carrier |
| `carrier_hist_delay_rate` | P(delayed) for this carrier |
| `origin_hist_mean_delay` | Avg delay from this origin city |
| `origin_hist_delay_rate` | P(delayed) from this origin city |
| `dest_hist_mean_delay` | Avg delay to this destination city |
| `depslot_hist_mean_delay` | Avg delay at this departure slot |
| `month_hist_mean_delay` | Avg delay in this month |
| `route_hist_mean_delay` | Avg delay on this origin→dest route |
| `route_hist_delay_rate` | P(delayed) on this route |
| `route_hist_n_flights` | Flight volume on this route |
| `carrier_origin_hist_mean_delay` | Avg delay for (carrier, origin) pair |
| `carrier_month_hist_mean_delay` | Avg delay for (carrier, month) pair |
| `depslot_origin_hist_mean_delay` | Avg delay for (depslot, origin) pair |
| `origin_weather_severity` | Ordinal 0–8 (Sunny=0, Heavy rain/thunder=8) |
| `dest_weather_severity` | Same for destination |

Fallback for unseen groups at inference: global historical mean from training fold.

### Key Design Decisions

- **No data leakage**: Post-flight columns (TAXI_OUT, CARRIER_DELAY, etc.) excluded from both train and inference. FeatureEngineer fitted on training fold only.
- **LightGBM over DBSCAN+RF**: Single model on 218 features replaces 507 per-cluster models trained on ~135 rows avg. Accuracy jumped from 73.43% to 91.34%.
- **Calibration**: `CalibratedClassifierCV(cv='prefit')` calibrates already-trained LGBM on a held-out val set — avoids refitting LightGBM 5× (would be very slow on 86k rows).
- **Incremental learning**: `train_incremental.py` warm-starts with `init_model=clf_raw` — adds N_NEW_TREES on top of existing model when ≥500 feedback rows accumulate.
- **Safety gate**: Incremental update rolls back if accuracy drops >2% on the held-out test set.
- **Log1p transform**: Delay distribution skewness 4.66. Log1p normalizes it; all regressors predict in log-space, back-transformed with expm1.
- **Regression on delayed only**: Training on all flights collapses predictions to the mean.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/model/info` | Version, accuracy, ROC-AUC, MAE, incremental update count |
| POST | `/predict` | Predict delay — returns binary + minutes + breakdown |
| POST | `/feedback` | Record actual outcome for incremental retraining |
| GET | `/flights/live?limit=N` | Live aircraft from OpenSky + predictions |
| GET | `/flights/weather?iata=JFK` | Current airport weather from Open-Meteo |
| GET | `/meta/options` | Valid carriers/cities for the form dropdowns |
| GET | `/stats/overview` | Pre-computed stats for dashboard charts |

### POST /predict — response shape
```json
{
  "delayed": true,
  "probability": 0.893,
  "probability_pct": "89.3%",
  "confidence": "high",
  "expected_delay_min": 105,
  "delay_range": "68–141 min",
  "delay_category": "significant",
  "delay_breakdown": {
    "carrier": 37,
    "weather": 8,
    "nas": 17,
    "late_aircraft": 41
  },
  "verdict": "High delay risk — expect around 105 min delay.",
  "cluster": -1,
  "model_used": "lgbm"
}
```

`cluster` is always `-1` and `model_used` is always `"lgbm"` in Sprint 7.
`delay_category` values: `on-time` / `minor` (<30 min) / `moderate` (30–60) / `significant` (60–120) / `severe` (>120)

### POST /feedback — request / response
```json
// Request body
{
  "flight_id": "UA123_2026-04-12",
  "actual_delayed": true,
  "actual_delay_min": 87
}
// Response
{"status": "recorded", "feedback_count": 1243}
```
Validation: `actual_delay_min` ∈ [0, 800]; if `actual_delayed=True` then `actual_delay_min > 0`; if `actual_delayed=False` then `actual_delay_min ≤ 15`. Duplicate `flight_id` silently ignored (idempotent).

---

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Predict | `/` | Form + result panel: gauge, delay minutes, stacked breakdown bar, tips |
| Dashboard | `/dashboard` | Recharts: delay by month, time-of-day, airline, weather, model stats |
| Live Map | `/live` | Leaflet dark map — live flights colored by delay risk, sidebar list |
| About | `/about` | Pipeline diagram, data leakage explanation, real-time data sources |

Frontend uses:
- **React 18 + Vite** — fast dev server with `/api` proxy to port 8000
- **Tailwind CSS** — dark theme (slate-900/950 base)
- **Recharts** — Dashboard charts
- **React-Leaflet + Leaflet** — Live Map
- **Lucide-React** — icons
- **Axios** — API client via `src/api.js`

**Frontend unchanged in Sprint 7** — response shape is identical to v2, no frontend changes needed.

---

## Running Locally

```bash
# 1. Start the API (Terminal 1, from MTP root)
uvicorn api.main:app --port 8000

# 2. Start the frontend (Terminal 2)
cd frontend
npm run dev
# -> http://localhost:5173

# 3. Full retrain from scratch (takes ~5-10 min, produces all model artifacts)
python train_lgbm.py

# 4. Incremental update on feedback rows (no-op if < 500 rows in data/feedback.csv)
python train_incremental.py

# 5. Run all tests (26 tests)
pytest tests/ -v
```

---

## Sprint History

| Sprint | Status | What was built |
|--------|--------|---------------|
| 1 — ML Model | Done | Data leakage audit, StandardScaler, ANOVA, DBSCAN, per-cluster RF, KNN assigner, global fallback, evaluation |
| 2 — FastAPI | Done | 7 endpoints, Pydantic schemas, CORS, startup model loading |
| 3 — Real-time | Done | OpenSky live flights, Open-Meteo weather, feature builder, airport coord lookup |
| 4 — Frontend | Done | 4 pages, dark UI, gauge, breakdown bar, live map, dashboard charts |
| 5 — Quantification | Done | Regression model for delay minutes, delay_category, delay_breakdown by type |
| 6 — Regression v2 | Done | HistGBR pipeline: 210 features, 7 historical features, quantile p10/p90, per-type regressors |
| 7 — LightGBM + Incremental | Done | Full LightGBM pipeline, 218 features, 91.34% accuracy, POST /feedback, nightly GitHub Actions |
| 8 — Cloud Deployment | Done | deploy_agent MCP server, Render (FastAPI backend) + Vercel (React frontend), auto-deploy on push |

---

## Sprint 7 — LightGBM + Incremental Learning

### What replaced what

| Old component | Replacement |
|---|---|
| DBSCAN + KNN cluster assigner | Removed — LGBMClassifier needs no clustering |
| SelectKBest ANOVA (k=30) | Removed — LightGBM uses all 218 features natively |
| 507 per-cluster RandomForestClassifiers | Single `LGBMClassifier` |
| Global HistGradientBoostingClassifier | Removed |
| StandardScaler | Removed — not needed for tree-based models |
| HistGradientBoostingRegressor suite | `LGBMRegressor` suite with same quantile + per-type structure |
| `hist_feature_lookup.pkl` (7 features) | `feature_engineering.pkl` via `FeatureEngineer` (15 features) |

### Incremental learning flow

```
POST /feedback (actual outcomes)
  -> data/feedback.csv (accumulates rows)
  -> GitHub Actions nightly cron (2 AM UTC)
  -> train_incremental.py
     -> if rows < 500: exit 0 (no-op)
     -> LGBMClassifier.fit(X_val, y_val, init_model='model/lgbm_clf.pkl')
     -> CalibratedClassifierCV(cv='prefit').fit(val)
     -> evaluate on test set
     -> if accuracy drops > 2%: sys.exit(1), keep old model
     -> save lgbm_clf_v{N}.pkl (keep last 3), overwrite lgbm_clf.pkl
     -> update metadata.pkl
     -> archive feedback rows -> data/feedback_archive.csv
     -> reset feedback.csv to header-only
  -> git commit model artifacts
  -> curl RENDER_DEPLOY_HOOK_URL (redeploy API)
```

### Model artifact layout

```
model/
├── lgbm_clf.pkl                # LGBMClassifier (raw, for init_model warm-start)
├── lgbm_clf_calibrated.pkl     # CalibratedClassifierCV (used for inference)
├── lgbm_reg.pkl                # Main LGBMRegressor (point estimate)
├── lgbm_reg_p10.pkl            # 10th percentile
├── lgbm_reg_p90.pkl            # 90th percentile
├── lgbm_type_regressors.pkl    # Dict: {carrier, weather, nas, late_aircraft}
├── feature_engineering.pkl     # Fitted FeatureEngineer (15 engineered features)
├── feature_names.pkl           # 203 pre-flight column names
└── metadata.pkl                # Keys: model_version, trained_at, overall_accuracy,
                                #   roc_auc, regression_mae, regression_median_ae,
                                #   regression_r2, pi_80_coverage, n_features,
                                #   incremental_updates, feedback_rows_used
```

### Test results (Sprint 7, seed=42, 70/10/20 split)

| Metric | Sprint 6 (v2) | Sprint 7 | Target |
|--------|--------------|----------|--------|
| Classification accuracy | 73.43% | **91.34%** | 78–82% |
| ROC-AUC | — | **0.973** | >0.85 |
| Regression MAE | 35.4 min | **30.8 min** | <28 min |
| Regression Median AE | 18.1 min | **14.7 min** | <15 min |
| R² | 0.633 | **0.704** | >0.70 |
| 80% PI coverage | 78% | **77%** | 78–82% |
