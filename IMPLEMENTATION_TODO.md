# Implementation TODO — MTP Flight Delay Predictor

**Created:** 2026-04-13  
**Last updated:** 2026-04-14

---

## ✅ COMPLETED

### 1. Vercel SPA Routing Fix
`frontend/vercel.json` added — all direct navigation and page refreshes now work on Vercel.

### 2. Distance Auto-Update + Haversine Fallback
`frontend/src/pages/Home.jsx` — Haversine formula covers all city pairs that aren't in the lookup table. Distance always auto-fills when origin/dest changes; shows `(est.)` badge for computed values.

### 3. LiveMap Filters
`frontend/src/pages/LiveMap.jsx` — callsign search + risk filter (All / At-Risk / On-Track) + city-pair filter.

### 4. Daily OpenSky Data Collection
`collect_opensky_data.py` created. `retrain.yml` runs it before `train_incremental.py` each night with `continue-on-error: true`.

---

## PENDING

### 5. Google Flights / Amadeus Flight Search Integration

**Status:** Not implemented — no free official Google Flights API.

Options ranked by feasibility:

| Option | Free Quota | Notes |
|--------|-----------|-------|
| **Amadeus** (recommended) | 2,000 calls/month | Sign up at https://developers.amadeus.com · env vars: `AMADEUS_API_KEY`, `AMADEUS_API_SECRET` · `pip install amadeus` |
| **SerpAPI** | 100 searches/month | https://serpapi.com · env var: `SERPAPI_KEY` |
| **aviationstack** | 100 calls/month | Real-time only |

**Files to change:**
- `api/services/flights.py` — add `search_flights()` function
- `api/main.py` — add `GET /flights/search?origin=JFK&dest=ORD&date=2026-05-01`
- `api/requirements.txt` — add `amadeus` or `serpapi`
- `frontend/src/pages/Home.jsx` — add flight picker that calls `/flights/search` to pre-fill the form

**Rough Amadeus integration for `api/services/flights.py`:**
```python
import os
from amadeus import Client as AmadeusClient, ResponseError

_amadeus = AmadeusClient(
    client_id=os.getenv("AMADEUS_API_KEY", ""),
    client_secret=os.getenv("AMADEUS_API_SECRET", ""),
)

async def search_amadeus_flights(origin_iata: str, dest_iata: str, date: str) -> list[dict]:
    """date format: YYYY-MM-DD"""
    try:
        resp = _amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin_iata,
            destinationLocationCode=dest_iata,
            departureDate=date,
            adults=1,
            max=10,
        )
        flights = []
        for offer in resp.data:
            for itinerary in offer.get("itineraries", []):
                for seg in itinerary.get("segments", []):
                    flights.append({
                        "carrier":        seg["carrierCode"],
                        "flight_number":  seg["number"],
                        "departure_time": seg["departure"]["at"],
                        "arrival_time":   seg["arrival"]["at"],
                        "duration_min":   _iso_duration_to_min(itinerary["duration"]),
                        "price_usd":      offer["price"]["total"],
                    })
        return flights
    except ResponseError:
        return []

def _iso_duration_to_min(iso: str) -> int:
    """Parse PT2H30M -> 150"""
    import re
    h = int(re.search(r"(\d+)H", iso).group(1)) if "H" in iso else 0
    m = int(re.search(r"(\d+)M", iso).group(1)) if "M" in iso else 0
    return h * 60 + m
```

---

### 6. Known Latent Bug — CI Crash if Feedback Reaches 500 Rows

`train_incremental.py` (line 92) reads `data/final_preprocessed_data.csv` to rebuild
val/test sets. This file is gitignored (41 MB) and not available in the GitHub Actions runner.

**Impact:** If feedback ever accumulates ≥ 500 rows (triggers real retrain), the nightly
CI job crashes with `FileNotFoundError`.

**Fix (one-time, run locally):**

Add to the end of `train_lgbm.py` after fitting:
```python
# Save val/test splits so train_incremental.py can run in CI without the full CSV
import joblib, numpy as np
joblib.dump((X_val_eng, y_val_status), MODEL_DIR / "val_split.pkl",  compress=3)
joblib.dump((X_te_eng,  y_te_status),  MODEL_DIR / "test_split.pkl", compress=3)
print("Saved val_split.pkl and test_split.pkl")
```

Then update `train_incremental.py` to load these instead of re-splitting:
```python
# Replace lines 80–111 with:
print("\n  Loading cached val/test splits...")
X_val_eng, y_val_status = joblib.load(MODEL_DIR / "val_split.pkl")
X_te_eng,  y_te_status  = joblib.load(MODEL_DIR / "test_split.pkl")
print(f"  Val set: {len(X_val_eng):,}  |  Test set: {len(X_te_eng):,}")
```

Then run `python train_lgbm.py` once locally to produce `val_split.pkl` and `test_split.pkl`,
commit them to git, and the CI will work correctly.

---

## Priority Order

| # | Task | Effort | Risk if skipped |
|---|------|--------|----------------|
| 6 | Save val/test splits for CI | 30 min | CI crashes if 500 feedback rows accumulate |
| 5 | Amadeus flight search | 2 hr | Feature gap, not a bug |
