# FlightSense — US Domestic Flight Delay Predictor

> Master's Thesis Project · 21MA23002  
> Predict US domestic flight delays in real time using LightGBM (91.34% accuracy, ROC-AUC 0.973)

---

## Live Demo

| Service | URL |
|---------|-----|
| Frontend | https://mtp-flight-delay.vercel.app |
| Backend API | https://mtp-flight-api.onrender.com |
| API Docs | https://mtp-flight-api.onrender.com/docs |

> **Note:** Render free tier has ~30 s cold-start delay after inactivity. The first prediction may be slow.

---

## What It Does

FlightSense predicts whether a US domestic flight will be delayed before it departs. Given a flight's route, carrier, departure time, date, and current weather, the system returns:

- **Delay probability** with a confidence rating
- **Expected delay minutes** with a 10th–90th percentile range
- **Delay breakdown** by cause: carrier, weather, air traffic (NAS), late aircraft
- **Live flight map** — real aircraft from OpenSky Network, each predicted in real time

---

## Screenshots

| Predict Page | Dashboard | Live Map |
|---|---|---|
| Form → result panel with probability gauge, delay breakdown bar | Monthly/airline/weather/time-of-day charts, model stats | Leaflet dark map + 3D globe with live flights coloured by risk |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React 18 + Vite                   │
│   Predict · Dashboard · Live Map · About            │
│   (Vercel CDN, auto-deploys on git push)            │
└─────────────────────┬───────────────────────────────┘
                      │ axios  VITE_API_URL
┌─────────────────────▼───────────────────────────────┐
│           FastAPI (Python 3.11, Render)             │
│  /predict  /model/info  /flights/live               │
│  /flights/weather  /meta/options  /stats/overview   │
│  /feedback                                          │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
  ┌────────▼───────┐   ┌──────────▼──────────┐
  │  LightGBM      │   │  OpenSky Network     │
  │  Classifier    │   │  (live aircraft)     │
  │  + Regressors  │   │                      │
  │  218 features  │   │  Open-Meteo          │
  └────────────────┘   │  (live weather)      │
                       └─────────────────────┘

GitHub Actions (nightly 2 AM UTC):
  collect_opensky_data.py → train_incremental.py → redeploy Render
```

---

## ML Pipeline

### Classification — binary (delayed / on-time)

| Step | Detail |
|------|--------|
| Training data | 86,478 rows, 219 raw columns, balanced 50/50 |
| Feature engineering | 15 historical features (carrier/route/depslot means) fitted on training fold only |
| Total features | 218 (203 pre-flight + 15 engineered) |
| Model | `LGBMClassifier` with early stopping, then `CalibratedClassifierCV(method='isotonic')` |
| Split | 70 / 10 / 20 (train / calibration-val / test), seed=42 |
| **Test accuracy** | **91.34%** |
| **ROC-AUC** | **0.973** |

### Regression — delay minutes (delayed flights only)

| Model | Purpose |
|-------|---------|
| `lgbm_reg.pkl` | Point estimate (log1p target, back-transformed with expm1) |
| `lgbm_reg_p10.pkl` | 10th-percentile quantile regressor |
| `lgbm_reg_p90.pkl` | 90th-percentile quantile regressor |
| `lgbm_type_regressors.pkl` | 4× per-cause regressors (carrier / weather / NAS / late aircraft) |

| Metric | Value |
|--------|-------|
| MAE | 30.8 min |
| Median AE | 14.7 min |
| R² | 0.704 |
| 80% PI coverage | 77% |

### Incremental Learning

Feedback submitted via `POST /feedback` accumulates in `data/feedback.csv`. A GitHub Actions cron job runs nightly at 2 AM UTC:

1. Collects yesterday's OpenSky domestic flight records → `data/opensky_collected.csv`
2. If feedback rows ≥ 500: warm-starts `LGBMClassifier` with `init_model=lgbm_clf.pkl`, adds new trees
3. Re-calibrates with `CalibratedClassifierCV(cv='prefit')`
4. Evaluates on held-out test set — rolls back if accuracy drops > 2%
5. Commits updated model artifacts, triggers Render redeploy

---

## API Reference

**Base URL:** `https://mtp-flight-api.onrender.com`  
Interactive docs: `/docs` (Swagger UI)

### `POST /predict`

```json
// Request
{
  "origin_city":      "new york",
  "dest_city":        "chicago",
  "carrier":          "Southwest Airlines",
  "distance":         790,
  "crs_elapsed_time": 130,
  "dep_hour":         8,
  "arr_hour":         10,
  "month":            7,
  "day":              15,
  "is_weekend":       false,
  "origin_iata":      "JFK",
  "dest_iata":        "ORD"
}

// Response
{
  "delayed":            true,
  "probability":        0.893,
  "probability_pct":    "89.3%",
  "confidence":         "high",
  "expected_delay_min": 105,
  "delay_range":        "68–141 min",
  "delay_category":     "significant",
  "delay_breakdown":    { "carrier": 37, "weather": 8, "nas": 17, "late_aircraft": 41 },
  "verdict":            "High delay risk — expect around 105 min delay.",
  "cluster":            -1,
  "model_used":         "lgbm"
}
```

`delay_category`: `on-time` · `minor` (<30 min) · `moderate` (30–60) · `significant` (60–120) · `severe` (>120)

### `POST /feedback`

```json
// Request
{ "flight_id": "UA123_2026-04-12", "actual_delayed": true, "actual_delay_min": 87 }
// Response
{ "status": "recorded", "feedback_count": 1243 }
```

Idempotent — duplicate `flight_id` values are silently ignored.

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/model/info` | Version, accuracy, ROC-AUC, MAE, update count |
| GET | `/flights/live?limit=N` | Live aircraft from OpenSky + delay predictions |
| GET | `/flights/weather?iata=JFK` | Current weather at an airport |
| GET | `/meta/options` | Valid carriers and cities for form dropdowns |
| GET | `/stats/overview` | Pre-computed stats for dashboard charts |

---

## Frontend Pages

| Page | Route | What's there |
|------|-------|-------------|
| Predict | `/` | Origin/dest/carrier form, Haversine distance auto-fill, probability gauge, delay breakdown bar, travel tips |
| Dashboard | `/dashboard` | Recharts: delay by month, time-of-day, airline, weather; ROC-AUC and MAE stat cards |
| Live Map | `/live` | Leaflet dark map + 3D globe; live OpenSky flights coloured by delay risk; callsign search; risk filter (All/At-Risk/On-Track) |
| About | `/about` | Pipeline diagram, data leakage explanation, real-time data sources |

---

## Running Locally

```bash
# Prerequisites: Python 3.11+, Node 18+

# 1. Clone the repo
git clone https://github.com/nanidurga/Flight-delay-prediction
cd Flight-delay-prediction

# 2. Start the API (Terminal 1)
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000
# → http://localhost:8000/docs

# 3. Start the frontend (Terminal 2)
cd frontend
npm install
npm run dev
# → http://localhost:5173

# 4. Full retrain from scratch (~5-10 min, requires final_preprocessed_data.csv)
pip install scikit-learn lightgbm joblib numpy pandas
python train_lgbm.py

# 5. Incremental update on feedback rows (no-op if < 500 rows)
python train_incremental.py

# 6. Collect yesterday's OpenSky flight data
python collect_opensky_data.py

# 7. Run all tests (49 tests)
pytest tests/ deploy_agent/tests/ -v
```

---

## Project Structure

```
MTP/
├── api/
│   ├── main.py                    FastAPI app — 8 endpoints, CORS regex for Vercel
│   ├── requirements.txt
│   └── services/
│       ├── predictor.py           Singleton model loader
│       ├── feature_engineering.py FeatureEngineer class (15 features)
│       └── flights.py             OpenSky live flights + Open-Meteo weather
├── frontend/
│   ├── src/
│   │   ├── App.jsx                Router — 4 routes
│   │   ├── api.js                 Axios client (uses VITE_API_URL in prod)
│   │   ├── components/Navbar.jsx
│   │   └── pages/
│   │       ├── Home.jsx           Predict form + result panel + Haversine distance
│   │       ├── Dashboard.jsx      Charts + model stat cards
│   │       ├── LiveMap.jsx        Leaflet map + 3D globe + callsign/risk filters
│   │       └── About.jsx
│   └── vercel.json                SPA routing rewrites (required for Vercel)
├── deploy_agent/                  MCP server for Render/Vercel automation
│   ├── server.py                  FastMCP server — 4 tools
│   ├── render.py                  Render API client
│   ├── vercel.py                  Vercel API client
│   ├── orchestrator.py            Deployment orchestration logic
│   ├── config.py                  Token loading from .env
│   ├── .env                       Local secrets (gitignored)
│   └── tests/                     23 pytest tests
├── model/
│   ├── lgbm_clf.pkl               LGBMClassifier (raw, for warm-start)
│   ├── lgbm_clf_calibrated.pkl    CalibratedClassifierCV (inference)
│   ├── lgbm_reg.pkl               LGBMRegressor — point estimate
│   ├── lgbm_reg_p10/p90.pkl       Quantile regressors
│   ├── lgbm_type_regressors.pkl   Per-cause regressors
│   ├── feature_engineering.pkl    Fitted FeatureEngineer
│   ├── feature_names.pkl          203 pre-flight column names
│   └── metadata.pkl               Metrics + version + update counters
├── data/
│   ├── final_preprocessed_data.csv   86,478 rows · 219 cols (gitignored, 41 MB)
│   ├── feedback.csv                  POST /feedback accumulator (header-only at start)
│   ├── feedback_archive.csv          Archived after each incremental update
│   └── opensky_collected.csv         Daily OpenSky collector output
├── tests/                         26 pytest tests (core API, features, predictions)
├── .github/workflows/retrain.yml  Nightly retraining cron (permissions: write)
├── train_lgbm.py                  Full retrain script
├── train_incremental.py           Warm-start update (requires 500 feedback rows)
├── collect_opensky_data.py        Daily OpenSky data collector
└── predict.py                     FlightPredictor class
```

---

## Deployment

| Platform | Service | Trigger |
|----------|---------|---------|
| Render (free) | FastAPI backend (Python 3.11) | Auto-deploy on `git push master` |
| Vercel | React frontend | Auto-deploy via GitHub integration |
| GitHub Actions | Nightly retraining | Cron `0 2 * * *` (2 AM UTC) |

**Vercel env vars (set in Vercel dashboard):**
- `VITE_API_URL` = `https://mtp-flight-api.onrender.com`

**GitHub Actions secrets (set in repo settings):**
- `RENDER_DEPLOY_HOOK_URL` — triggers Render redeploy after incremental model update

**deploy_agent MCP server** (registered in `~/.claude/settings.json`):
- Tools: `render_get_logs`, `render_get_status`, `vercel_get_status`, `deploy_full_stack`
- Requires restart of Claude Code to activate after settings change

---

## Sprint History

| Sprint | What was built |
|--------|---------------|
| 1 — ML Model | DBSCAN + per-cluster RF, KNN cluster assigner |
| 2 — FastAPI | 7 endpoints, Pydantic schemas, CORS, startup model loading |
| 3 — Real-time | OpenSky live flights, Open-Meteo weather, feature builder |
| 4 — Frontend | 4 pages, dark UI, gauge, breakdown bar, live map, dashboard |
| 5 — Quantification | Regression for delay minutes, delay_category, delay_breakdown |
| 6 — Regression v2 | HistGBR pipeline: 210 features, quantile p10/p90, per-type regressors |
| 7 — LightGBM + Incremental | Full LGBM pipeline, 218 features, 91.34% accuracy, POST /feedback, nightly CI |
| 8 — Cloud Deployment | Render + Vercel deployment, deploy_agent MCP server, auto-deploy on push |
| Post-8 — Bug Fixes | CORS regex, api.js VITE_API_URL, workflow permissions, file handle leaks, 49 tests |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | LightGBM 4, scikit-learn (calibration), NumPy, pandas |
| API | FastAPI, Pydantic v2, httpx, uvicorn |
| Frontend | React 18, Vite, Tailwind CSS 3, Recharts, React-Leaflet, react-globe.gl |
| Data sources | OpenSky Network (live flights), Open-Meteo (weather), BTS On-Time Performance |
| Deployment | Render (backend), Vercel (frontend), GitHub Actions (CI/CD + nightly retrain) |
| Tooling | deploy_agent MCP server (FastMCP), pytest (49 tests) |

---

## License

Academic project — not for commercial use.  
Data sources: [BTS](https://www.transtats.bts.gov), [OpenSky Network](https://opensky-network.org), [Open-Meteo](https://open-meteo.com).
