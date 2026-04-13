"""
main.py — FastAPI backend for the FlightSense Flight Delay Predictor.

Endpoints:
  GET  /                    health check
  GET  /model/info          model metadata (accuracy, ROC-AUC, MAE, update count)
  POST /predict             predict delay for a manually entered flight
  POST /feedback            record actual outcome for incremental retraining
  GET  /flights/live        fetch live flights from OpenSky + predict each
  GET  /flights/weather     current weather at an airport (IATA code)
  GET  /meta/options        valid carrier/city lists for form dropdowns
  GET  /stats/overview      summary stats from training data for dashboard charts

Run with:  uvicorn api.main:app --reload   (from MTP root)
"""

import asyncio
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import csv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# ── imports from our modules ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.services.predictor import get_predictor
from api.services.flights import (
    fetch_live_flights,
    get_airport_weather,
    build_feature_dict,
    get_season,
    KNOWN_CARRIERS,
    KNOWN_ORIGIN_CITIES,
    KNOWN_DEST_CITIES,
)

FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "feedback.csv"
_feedback_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Flight Delay Predictor API",
    description="Predicts US domestic flight delays using a LightGBM ML pipeline (91.34% accuracy, ROC-AUC 0.973).",
    version="2.0.0",
)

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mtp-flight-delay-exaj8moop-durgas-projects-9ea2fbeb.vercel.app",
        "http://localhost:5173",
        "http://localhost:4173",   # vite preview
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
@app.on_event("startup")
def load_model():
    get_predictor()   # warms the singleton


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class FlightInput(BaseModel):
    """All fields a user enters to get a prediction."""
    origin_city      : str = Field(..., example="new york")
    dest_city        : str = Field(..., example="chicago")
    carrier          : str = Field(..., example="Southwest Airlines")
    distance         : float = Field(..., example=790, gt=0)
    crs_elapsed_time : float = Field(..., example=130, gt=0,
                                     description="Scheduled flight duration in minutes")
    dep_hour         : int  = Field(..., example=8, ge=0, le=23,
                                    description="Scheduled departure hour (0-23)")
    arr_hour         : int  = Field(..., example=10, ge=0, le=23,
                                    description="Scheduled arrival hour (0-23)")
    month            : int  = Field(..., example=7, ge=1, le=12)
    day              : int  = Field(..., example=15, ge=1, le=31)
    is_weekend       : bool = Field(False)
    # Optional: user can supply weather, otherwise we fetch it live
    origin_humidity           : Optional[float] = None
    dest_humidity             : Optional[float] = None
    origin_temp_celsius       : Optional[float] = None
    dest_temp_celsius         : Optional[float] = None
    origin_condition_text     : Optional[str]   = None
    dest_condition_text       : Optional[str]   = None
    origin_iata               : Optional[str]   = None   # used to fetch live weather
    dest_iata                 : Optional[str]   = None


class DelayBreakdown(BaseModel):
    carrier      : int
    weather      : int
    nas          : int
    late_aircraft: int

class PredictionResponse(BaseModel):
    delayed             : bool
    probability         : float
    probability_pct     : str
    cluster             : int
    model_used          : str
    confidence          : str
    verdict             : str
    expected_delay_min  : int
    delay_range         : str
    delay_category      : str
    delay_breakdown     : DelayBreakdown


class FeedbackInput(BaseModel):
    """Actual outcome for a previously predicted flight."""
    flight_id       : str
    actual_delayed  : bool
    actual_delay_min: int = Field(..., ge=0, le=800,
                                  description="Actual delay in minutes (0–800)")

    @model_validator(mode="after")
    def check_delay_consistency(self) -> "FeedbackInput":
        if self.actual_delayed and self.actual_delay_min == 0:
            raise ValueError(
                "actual_delay_min must be > 0 when actual_delayed is True")
        if not self.actual_delayed and self.actual_delay_min > 15:
            raise ValueError(
                "actual_delay_min > 15 but actual_delayed is False — "
                "a flight with >15 min delay should have actual_delayed=True")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "Flight Delay Predictor API"}


@app.get("/model/info", tags=["model"])
def model_info():
    """Return training metadata: accuracy, ROC-AUC, regression metrics, version."""
    predictor = get_predictor()
    info = predictor.info
    return {
        "model_version"        : info.get("model_version", 1),
        "trained_at"           : info.get("trained_at", "unknown"),
        "model_type"           : info.get("model_type", "LGBMClassifier + LGBMRegressor"),
        "overall_accuracy_pct" : f"{info['overall_accuracy']*100:.2f}%",
        "roc_auc"              : round(info.get("roc_auc", 0), 4),
        "regression_mae"       : round(info.get("regression_mae", 0), 1),
        "regression_r2"        : round(info.get("regression_r2", 0), 3),
        "pi_80_coverage"       : round(info.get("pi_80_coverage", 0), 3),
        "n_features"           : info.get("n_features", 0),
        "incremental_updates"  : info.get("incremental_updates", 0),
        "feedback_rows_used"   : info.get("feedback_rows_used", 0),
        # Legacy fields — kept so any external clients don't break
        "n_clusters"           : info.get("n_clusters", 0),
        "n_cluster_models"     : info.get("n_cluster_models", 0),
        "selected_features"    : info.get("selected_features", []),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_flight(flight: FlightInput):
    """
    Predict delay probability for a single flight.
    If origin/dest IATA codes are provided and weather isn't, we fetch live weather.
    """
    predictor = get_predictor()

    # Build weather dicts — use user-supplied values or fetch from Open-Meteo
    if flight.origin_iata and flight.origin_humidity is None:
        origin_weather = await get_airport_weather(flight.origin_iata)
    else:
        origin_weather = {
            "temperature_celsius": flight.origin_temp_celsius or 20,
            "humidity"           : flight.origin_humidity or 60,
            "condition_text"     : flight.origin_condition_text or "Partly cloudy",
        }

    if flight.dest_iata and flight.dest_humidity is None:
        dest_weather = await get_airport_weather(flight.dest_iata)
    else:
        dest_weather = {
            "temperature_celsius": flight.dest_temp_celsius or 20,
            "humidity"           : flight.dest_humidity or 60,
            "condition_text"     : flight.dest_condition_text or "Partly cloudy",
        }

    season = get_season(flight.month)

    features = build_feature_dict(
        origin_city      = flight.origin_city,
        dest_city        = flight.dest_city,
        carrier          = flight.carrier,
        distance         = flight.distance,
        crs_elapsed_time = flight.crs_elapsed_time,
        origin_weather   = origin_weather,
        dest_weather     = dest_weather,
        dep_hour         = flight.dep_hour,
        arr_hour         = flight.arr_hour,
        month            = flight.month,
        day              = flight.day,
        is_weekend       = flight.is_weekend,
        season           = season,
    )

    result = predictor.predict(features)
    prob   = result["probability"]
    return PredictionResponse(
        delayed              = result["delayed"],
        probability          = prob,
        probability_pct      = result["probability_pct"],
        cluster              = result["cluster"],
        model_used           = result["model_used"],
        confidence           = result["confidence"],
        verdict              = (
            f"High delay risk — expect around {result['expected_delay_min']} min delay."
            if prob > 0.6 else
            f"Moderate risk — could be delayed by {result['expected_delay_min']} min."
            if prob > 0.4 else
            "Likely on-time flight."
        ),
        expected_delay_min   = result["expected_delay_min"],
        delay_range          = result["delay_range"],
        delay_category       = result["delay_category"],
        delay_breakdown      = DelayBreakdown(**result["delay_breakdown"]),
    )


@app.get("/flights/live", tags=["live"])
async def live_flights(limit: int = Query(30, le=100)):
    """
    Fetch live flights from OpenSky Network and predict delay for each.
    Returns list of flights with position, weather, and delay prediction.
    """
    predictor = get_predictor()
    flights   = await fetch_live_flights(limit=limit)

    if not flights:
        return {"flights": [], "note": "OpenSky returned no data (may be rate-limited)"}

    enriched = []
    for f in flights:
        w = f.get("weather", {})
        # Build minimal features for live flights (limited info available)
        features = {
            "origin_temperature_celsius": w.get("temperature_celsius", 20),
            "origin_humidity"           : w.get("humidity", 60),
            "dest_temperature_celsius"  : w.get("temperature_celsius", 20),
            "dest_humidity"             : w.get("humidity", 60),
            "CRS_ELAPSED_TIME"          : 120,   # unknown, use average
            "DISTANCE"                  : 500,
        }
        cond = w.get("condition_text", "Partly cloudy")
        features[f"origin_condition_text_{cond}"] = 1
        features[f"dest_condition_text_{cond}"]   = 1

        # Add current month/season defaults
        now = datetime.utcnow()
        for m in range(2, 13):
            features[f"MONTH_{m}"] = int(now.month == m)
        season = get_season(now.month)
        for s in ["Autumn", "Spring", "Summer", "Winter"]:
            features[f"Season_{s}"] = int(season == s)

        pred = predictor.predict(features)

        enriched.append({
            "callsign"   : f["callsign"],
            "icao24"     : f["icao24"],
            "lat"        : f["lat"],
            "lon"        : f["lon"],
            "altitude_m" : f["altitude"],
            "velocity_ms": f["velocity"],
            "heading"    : f["heading"],
            "weather"    : w,
            "prediction" : pred,
        })

    return {
        "count"  : len(enriched),
        "flights": enriched,
    }


@app.get("/flights/weather", tags=["live"])
async def airport_weather(iata: str = Query(..., examples=["JFK"])):
    """Get current weather at an airport by IATA code."""
    weather = await get_airport_weather(iata.upper())
    return {"iata": iata.upper(), "weather": weather}


@app.get("/meta/options", tags=["meta"])
def options():
    """Return the list of valid carrier, city, and destination options for the form."""
    return {
        "carriers"      : KNOWN_CARRIERS,
        "origin_cities" : KNOWN_ORIGIN_CITIES,
        "dest_cities"   : KNOWN_DEST_CITIES,
    }


@app.get("/stats/overview", tags=["stats"])
def stats_overview():
    """
    Pre-computed stats from the training dataset for dashboard charts.
    These are hardcoded summaries (since we can't run pandas in production).
    """
    return {
        "total_flights"     : 86478,
        "delayed_pct"       : 50.0,     # balanced dataset
        "on_time_pct"       : 50.0,
        "top_delay_carriers": [
            {"carrier": "Southwest Airlines", "delay_rate_pct": 52},
            {"carrier": "American Airlines",  "delay_rate_pct": 51},
            {"carrier": "Spirit Airlines",    "delay_rate_pct": 55},
            {"carrier": "Delta Airlines",     "delay_rate_pct": 48},
            {"carrier": "JetBlue Airways",    "delay_rate_pct": 53},
        ],
        "delay_by_month": [
            {"month": "Jan", "delay_rate_pct": 55},
            {"month": "Feb", "delay_rate_pct": 57},
            {"month": "Mar", "delay_rate_pct": 49},
            {"month": "Apr", "delay_rate_pct": 47},
            {"month": "May", "delay_rate_pct": 46},
            {"month": "Jun", "delay_rate_pct": 52},
            {"month": "Jul", "delay_rate_pct": 54},
            {"month": "Aug", "delay_rate_pct": 53},
            {"month": "Sep", "delay_rate_pct": 45},
            {"month": "Oct", "delay_rate_pct": 44},
            {"month": "Nov", "delay_rate_pct": 48},
            {"month": "Dec", "delay_rate_pct": 56},
        ],
        "delay_by_time_of_day": [
            {"slot": "Early Morning (0-6)",  "delay_rate_pct": 42},
            {"slot": "Morning (6-12)",       "delay_rate_pct": 44},
            {"slot": "Afternoon (12-18)",    "delay_rate_pct": 54},
            {"slot": "Evening (18-24)",      "delay_rate_pct": 60},
        ],
        "top_delay_conditions": [
            {"condition": "Thundery outbreaks", "delay_rate_pct": 68},
            {"condition": "Heavy rain",          "delay_rate_pct": 63},
            {"condition": "Fog",                 "delay_rate_pct": 61},
            {"condition": "Moderate rain",       "delay_rate_pct": 55},
            {"condition": "Overcast",            "delay_rate_pct": 49},
            {"condition": "Sunny",               "delay_rate_pct": 41},
        ],
    }


@app.post("/feedback", tags=["feedback"])
def record_feedback(fb: FeedbackInput):
    """
    Record the actual delay outcome for a previously predicted flight.
    Appends one row to data/feedback.csv for nightly incremental retraining.
    Idempotent: duplicate flight_id values are silently ignored.
    """
    try:
        with _feedback_lock:
            # Read existing flight IDs to detect duplicates
            if FEEDBACK_PATH.exists():
                with open(FEEDBACK_PATH, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)   # skip header
                    existing_ids = {row[0] for row in reader if row}
                if fb.flight_id in existing_ids:
                    count = max(0, sum(1 for _ in open(FEEDBACK_PATH, encoding="utf-8")) - 1)
                    return {"status": "duplicate_ignored", "feedback_count": count}

            # Append new row
            write_header = (not FEEDBACK_PATH.exists() or
                            FEEDBACK_PATH.stat().st_size == 0)
            with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["flight_id", "timestamp",
                                     "actual_delayed", "actual_delay_min"])
                writer.writerow([
                    fb.flight_id,
                    datetime.utcnow().isoformat(),
                    int(fb.actual_delayed),
                    fb.actual_delay_min,
                ])

            count = max(0, sum(1 for _ in open(FEEDBACK_PATH, encoding="utf-8")) - 1)
            return {"status": "recorded", "feedback_count": count}
    except OSError as exc:
        raise HTTPException(status_code=500,
                            detail=f"Could not write feedback: {exc}") from exc
