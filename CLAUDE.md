# MTP — Flight Delay Prediction (FlightSense)
**Student:** 21MA23002 | **Type:** Master's Thesis Project

Goal: Predict US domestic flight delays using LightGBM and serve real-time predictions via a deployed website.

---

## Live Deployment (Sprint 8 — Deployed 2026-04-13, post-fixes 2026-04-14)

| Service | URL | Platform |
|---------|-----|----------|
| Backend API | https://mtp-flight-api.onrender.com | Render (free tier, Python 3.11) |
| Frontend | Vercel — URL changes each push (see Vercel dashboard for current) | Vercel |
| GitHub Repo | https://github.com/nanidurga/Flight-delay-prediction | master branch |

**Render service ID:** `srv-d7eh888sfn5c73d07o1g`
**Vercel project ID:** `prj_nNItwgRGGIDD4udXJVkB7nnLRyAW`  
**Vercel project name:** `mtp-flight-delay` (owned by `durgas-projects-9ea2fbeb`)

The frontend `VITE_API_URL` env var (set on Vercel) points to the Render backend URL.  
`api.js` uses `import.meta.env.VITE_API_URL ?? '/api'` — falls back to Vite proxy in local dev.  
Render auto-deploys on every push to `master`. Vercel auto-deploys via the GitHub integration.

> **Note on Vercel URLs:** Vercel generates a new unique hash URL for each deployment  
> (e.g. `mtp-flight-delay-abc123-durgas-projects-9ea2fbeb.vercel.app`). The CORS in  
> `api/main.py` uses `allow_origin_regex` to cover all of them automatically.

To redeploy manually, trigger via `deploy_agent` MCP tools or the Render/Vercel dashboards.  
To view Render runtime logs, open: https://dashboard.render.com/web/srv-d7eh888sfn5c73d07o1g/logs

---

## Folder Structure

**Key files added/changed since Sprint 8 (includes post-deploy fixes 2026-04-14):**
- `frontend/vercel.json` — SPA routing rewrites (fixes direct navigation on Vercel)
- `frontend/src/api.js` — Uses `VITE_API_URL ?? '/api'`; was hardcoded `/api` (broke production)
- `api/main.py` — CORS now uses `allow_origin_regex` (Vercel hash URLs change each push); fixed file handle leaks in `/feedback`; version 2.0.0
- `.github/workflows/retrain.yml` — Added `permissions: contents: write`; git push now conditional on actual changes (was failing nightly)
- `collect_opensky_data.py` — Daily OpenSky data collector (runs before retrain in CI)
- `deploy_agent/render.py` — `get_logs` uses `/events` endpoint (Render REST API has no log stream)
- `deploy_agent/server.py` — Updated `render_get_logs` tool docstring
- `deploy_agent/tests/test_render.py` — Updated test mock for new events-based get_logs
- `api/services/flights.py` — KNOWN_DEST_CITIES expanded to match KNOWN_ORIGIN_CITIES
- `frontend/src/pages/Home.jsx` — Haversine fallback for all city pairs; distance always auto-fills
- `frontend/src/pages/LiveMap.jsx` — Callsign search + risk filter (All / At-Risk / On-Track)
- `frontend/src/pages/Dashboard.jsx` — ROC-AUC and Regression MAE stat cards (replaced DBSCAN metrics)

---

```
MTP/                                  <- PROJECT ROOT (always work from here)
│
├── data/
│   ├── final_preprocessed_data.csv  # 86,478 rows · 219 cols · balanced 50/50
│   ├── feedback.csv                  # Accumulates POST /feedback outcomes (header-only at start)
│   ├── feedback_archive.csv          # Archived feedback rows after each incremental update
│   └── opensky_collected.csv         # Daily OpenSky domestic flight records (CI-collected)
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
│   ├── vercel.json                   # SPA routing rewrites — REQUIRED for Vercel page navigation
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
├── collect_opensky_data.py           # Daily OpenSky domestic flight data collector (runs in CI)
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

# 5. Run all tests (49 tests: 26 core + 23 deploy_agent)
pytest tests/ deploy_agent/tests/ -v

# 6. Start the deploy_agent MCP server (for Claude Code tool access)
python deploy_agent/server.py
# Registered automatically via ~/.claude/settings.json — restart Claude Code to pick up
```

---

## Known Issues / Gotchas

- **OpenSky rate limit:** Free API allows ~1 req/10 s for anonymous access. `collect_opensky_data.py` sleeps 10 s between airports. In CI the step is `continue-on-error: true` so rate-limits don't fail the retrain.
- **OpenSky doesn't provide scheduled times:** Exact delay labels can't be computed from OpenSky alone. Ground-truth labels come from `POST /feedback` and BTS On-Time Performance monthly data.
- **Render free tier cold starts:** API takes ~30 s to wake up after inactivity. Frontend shows loading state. Users in India will see ~250 ms extra latency to Render's Oregon server.
- **`vercel.json` must stay in `frontend/`** — without it, refreshing any page other than `/` returns Vercel 404.
- **Vercel URL changes on every push:** Each Vercel deployment gets a new hash URL. CORS in `api/main.py` uses `allow_origin_regex=r"https://mtp-flight-delay[^.]*\.vercel\.app"` to handle this automatically. Do NOT revert to a hardcoded URL.
- **deploy_agent MCP tools:** Registered in `~/.claude/settings.json`. Requires Claude Code restart to appear. Tools: `render_get_logs`, `render_get_status`, `vercel_get_status`, `deploy_full_stack`. Runtime logs not available via REST API — use Render dashboard.
- **CI crash if feedback hits 500 rows:** `train_incremental.py` tries to read `data/final_preprocessed_data.csv` (gitignored, 41 MB) for the safety gate. If you ever accumulate 500 feedback rows, the nightly retrain will crash with `FileNotFoundError`. Fix: save `X_test.pkl`/`y_test.pkl` from `train_lgbm.py` to `model/` and load them in `train_incremental.py` instead.
- **Python versions:** Local dev runs Python 3.13 (system). Render and GitHub Actions use Python 3.11 (configured). All packages are compatible with both — no action needed.
- **Deprecation warnings in tests:** `@app.on_event("startup")` and `Field(..., example=...)` produce warnings but work correctly. Leave for thesis; fix before any production upgrade.
- **Google Flights API:** No free official API. Options: SerpAPI (~100 free/month), Amadeus (~2000 free/month). Not implemented — see IMPLEMENTATION_TODO.md.

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
| Post-8 — Bug Fixes | Done | CORS regex (Vercel hash URLs), api.js VITE_API_URL, workflow permissions, file handle leaks, render.py events endpoint, 49 tests passing |

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
