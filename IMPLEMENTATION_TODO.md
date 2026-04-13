# Implementation TODO — MTP Flight Delay Predictor

**Created:** 2026-04-13  
**Status:** Pending implementation

---

## 1. CRITICAL — Fix Vercel SPA Routing (frontend navigation broken)

**Problem:** React Router uses client-side routing. On Vercel, refreshing any page except `/`
or navigating directly to `/dashboard`, `/live`, `/about` returns a 404 because Vercel tries
to serve a static file at that path instead of serving `index.html`.

**Fix:** Add `vercel.json` at `frontend/vercel.json` (Vercel reads it from the project root
being deployed, which is the `frontend/` folder since Vercel is configured to deploy from there).

```json
// frontend/vercel.json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

If the Vercel project root is set to the repo root (not `frontend/`), put it at
`vercel.json` in the repo root instead.

**Also check:** In the Vercel dashboard → Project Settings → Build & Output Settings, confirm:
- Framework Preset: `Vite`
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

---

## 2. Fix Distance Not Auto-Updating on Origin/Destination Change

**Problem (`frontend/src/pages/Home.jsx`):** The `DISTANCES` lookup table only covers ~30
city pairs. When the user picks an origin and destination not in the table, `getApproxDistance`
returns `null` and the distance field shows nothing or stale data. The form also doesn't
auto-fill distance when origin/dest change — it requires the user to type it manually.

**Fix:**
- Wire up `useEffect` to auto-compute distance whenever `origin` or `dest` state changes.
- Expand `DISTANCES` with the missing pairs (or switch to Haversine formula using airport
  coordinates for all city pairs as a fallback).
- Show "~NNN mi (estimated)" when using Haversine, "NNN mi" when from lookup table.

**Airport coordinates map to add** (add to `Home.jsx` alongside `DISTANCES`):
```js
const AIRPORT_COORDS = {
  'new york':       { lat: 40.6413, lon: -73.7781 },  // JFK
  'los angeles':    { lat: 33.9425, lon: -118.4081 },
  'chicago':        { lat: 41.9742, lon: -87.9073 },
  'miami':          { lat: 25.7959, lon: -80.2870 },
  'dallas':         { lat: 32.8998, lon: -97.0403 },
  'houston':        { lat: 29.9902, lon: -95.3368 },
  'atlanta':        { lat: 33.6407, lon: -84.4277 },
  'seattle':        { lat: 47.4502, lon: -122.3088 },
  'denver':         { lat: 39.8561, lon: -104.6737 },
  'boston':         { lat: 42.3656, lon: -71.0096 },
  'san francisco':  { lat: 37.6213, lon: -122.3790 },
  'washington':     { lat: 38.9531, lon: -77.4565 },
  'las vegas':      { lat: 36.0840, lon: -115.1537 },
  'phoenix':        { lat: 33.4343, lon: -112.0116 },
  'orlando':        { lat: 28.4312, lon: -81.3081 },
  'philadelphia':   { lat: 39.8744, lon: -75.2424 },
  'minneapolis':    { lat: 44.8848, lon: -93.2223 },
  'san diego':      { lat: 32.7338, lon: -117.1933 },
  'tampa':          { lat: 27.9755, lon: -82.5332 },
  'portland':       { lat: 45.5898, lon: -122.5951 },
  'charlotte':      { lat: 35.2144, lon: -80.9473 },
  'nashville':      { lat: 36.1263, lon: -86.6774 },
  'salt lake city': { lat: 40.7899, lon: -111.9791 },
  'kansas city':    { lat: 39.2976, lon: -94.7139 },
  'memphis':        { lat: 35.0421, lon: -89.9792 },
  'new orleans':    { lat: 29.9934, lon: -90.2580 },
  'baltimore':      { lat: 39.1754, lon: -76.6682 },
  'san jose':       { lat: 37.3626, lon: -121.9290 },
  'oakland':        { lat: 37.7213, lon: -122.2208 },
  'austin':         { lat: 30.1975, lon: -97.6664 },
  'san antonio':    { lat: 29.5337, lon: -98.4698 },
  'jacksonville':   { lat: 30.4941, lon: -81.6879 },
  'indianapolis':   { lat: 39.7173, lon: -86.2944 },
  'columbus':       { lat: 39.9980, lon: -82.8919 },
  'detroit':        { lat: 42.2162, lon: -83.3554 },
  'richmond':       { lat: 37.5052, lon: -77.3197 },
  'norfolk':        { lat: 36.8973, lon: -76.0179 },
  'raleigh-durham': { lat: 35.8801, lon: -78.7880 },
  'pittsburgh':     { lat: 40.4915, lon: -80.2329 },
  'cleveland':      { lat: 41.4117, lon: -81.8498 },
  'buffalo':        { lat: 42.9405, lon: -78.7322 },
  'sacramento':     { lat: 38.6954, lon: -121.5908 },
  'honolulu':       { lat: 21.3245, lon: -157.9251 },
  'san juan':       { lat: 18.4394, lon: -66.0018 },
}

function haversineDistanceMiles(c1, c2) {
  const R = 3958.8  // Earth radius in miles
  const dLat = (c2.lat - c1.lat) * Math.PI / 180
  const dLon = (c2.lon - c1.lon) * Math.PI / 180
  const a = Math.sin(dLat/2)**2 +
            Math.cos(c1.lat * Math.PI/180) * Math.cos(c2.lat * Math.PI/180) *
            Math.sin(dLon/2)**2
  return Math.round(R * 2 * Math.asin(Math.sqrt(a)))
}

function getApproxDistance(origin, dest) {
  const known = DISTANCES[`${origin}|${dest}`] || DISTANCES[`${dest}|${origin}`]
  if (known) return { miles: known, estimated: false }
  const c1 = AIRPORT_COORDS[origin], c2 = AIRPORT_COORDS[dest]
  if (c1 && c2) return { miles: haversineDistanceMiles(c1, c2), estimated: true }
  return null
}
```

**Then in the form state** (wherever `origin`/`dest` are in state):
```js
useEffect(() => {
  if (form.origin && form.dest) {
    const result = getApproxDistance(form.origin, form.dest)
    if (result) setForm(f => ({ ...f, distance: result.miles }))
  }
}, [form.origin, form.dest])
```

---

## 3. Fix Live Map — Origin/Destination Point Selection

**Problem (`frontend/src/pages/LiveMap.jsx`):** The LiveMap shows real-time OpenSky flights
but there is no origin/destination input for users to filter or select flights between two
airports. The "start point and end point" the user wants likely refers to being able to click
on map markers to set origin/destination for a prediction.

**Fix — add click-to-predict flow:**
1. Add state: `originFlight`, `destAirport` (or just show a sidebar form)
2. When a map marker (flight) is clicked in the Leaflet map, populate a sidebar panel with
   that flight's data
3. Add a "Predict this flight" button that calls `predictFlight()` and shows the result
   in the sidebar
4. Alternatively: add two airport selector dropdowns at the top of LiveMap so users can
   filter flights by origin/dest city and see only those routes highlighted

**Suggested simpler fix (filter by city):**
```jsx
// Add at top of LiveMap:
const [filterOrigin, setFilterOrigin] = useState('')
const [filterDest,   setFilterDest]   = useState('')

// Filter flights:
const visibleFlights = flights.filter(f => {
  if (filterOrigin && !f.origin?.toLowerCase().includes(filterOrigin)) return false
  if (filterDest   && !f.destination?.toLowerCase().includes(filterDest)) return false
  return true
})
```

---

## 4. NEW FEATURE — Daily OpenSky Data Collection for Recursive Retraining

**Problem:** The current GitHub Actions workflow (`retrain.yml`) only does incremental
warm-start retraining when ≥500 POST /feedback rows accumulate. It does NOT pull new
labeled flight data from OpenSky automatically. The training data stays frozen at the
initial 86,478 rows from `data/final_preprocessed_data.csv`.

**Goal:** Every day, pull previous day's actual flight records, label them (delayed/on-time),
and append to the training dataset so the model trains on fresh real-world data recursively.

**Note on OpenSky API:** OpenSky's free `/flights/departure` and `/flights/arrival` endpoints
return historical flight records (icao24, firstSeen, lastSeen, callsign, estDepartureAirport,
estArrivalAirport). These can be used to reconstruct approximate delay labels.

### New file: `collect_opensky_data.py`

```python
"""
Daily OpenSky data collector.
Pulls yesterday's US domestic flight records from OpenSky Network's REST API,
computes approximate delay labels, and appends to data/opensky_collected.csv.

Runs as a GitHub Actions step before train_incremental.py.
"""
import requests, pandas as pd, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# OpenSky free API — no auth needed for public data, rate-limited
OPENSKY_BASE = "https://opensky-network.org/api"

# Major US airport ICAO codes (mapped from city names used in the model)
US_AIRPORTS = {
    "KATL": "atlanta", "KORD": "chicago", "KLAX": "los angeles",
    "KDFW": "dallas-fort worth", "KDEN": "denver", "KJFK": "new york",
    "KSFO": "san francisco", "KSEA": "seattle", "KLAS": "las vegas",
    "KMCO": "orlando", "KEWR": "newark", "KPHX": "phoenix",
    "KIAH": "houston", "KMIA": "miami", "KBOS": "boston",
    "KATL": "atlanta", "KMSP": "minneapolis", "KDTW": "detroit",
    "KPHL": "philadelphia", "KLGA": "new york", "KBWI": "baltimore",
    "KDAL": "dallas", "KHOU": "houston", "KSLC": "salt lake city",
    "KSAN": "san diego", "KTPA": "tampa", "KPDX": "portland",
    "KSTL": "st. louis", "KBNA": "nashville", "KCLT": "charlotte",
    "KAUS": "austin", "KSAT": "san antonio", "KMEM": "memphis",
    "KMSY": "new orleans", "KRDU": "raleigh-durham", "KPIT": "pittsburgh",
    "KCLE": "cleveland", "KRIC": "richmond", "KORF": "norfolk",
    "KCMH": "columbus", "KIND": "indianapolis", "KMKE": "milwaukee",
    "KOAK": "oakland", "KSJC": "san jose", "KSMF": "sacramento",
    "KRSW": "fort myers", "KFLL": "fort lauderdale", "KPBI": "west palm beach",
    "KHNL": "honolulu", "TJSJ": "san juan",
}
ICAO_SET = set(US_AIRPORTS.keys())

def fetch_yesterday_flights():
    """Fetch yesterday's departure records from OpenSky for major US airports."""
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    begin = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                         0, 0, 0, tzinfo=timezone.utc).timestamp())
    end   = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                         23, 59, 59, tzinfo=timezone.utc).timestamp())
    rows = []
    for icao, city in US_AIRPORTS.items():
        try:
            r = requests.get(
                f"{OPENSKY_BASE}/flights/departure",
                params={"airport": icao, "begin": begin, "end": end},
                timeout=30
            )
            if r.status_code != 200:
                continue
            for f in r.json():
                dest_icao = f.get("estArrivalAirport", "")
                if dest_icao not in ICAO_SET:
                    continue  # keep domestic only
                # Delay proxy: difference between firstSeen/lastSeen vs scheduled
                # OpenSky doesn't give scheduled times — use duration as proxy
                duration_min = ((f.get("lastSeen", 0) - f.get("firstSeen", 0)) / 60
                                if f.get("lastSeen") and f.get("firstSeen") else None)
                rows.append({
                    "ORIGIN_CITY_NAME": city,
                    "DEST_CITY_NAME":   US_AIRPORTS.get(dest_icao, ""),
                    "DEP_TIME_BLK":     _hour_block(f.get("firstSeen")),
                    "MONTH":            yesterday.month,
                    "DAY_OF_WEEK":      yesterday.weekday() + 1,
                    "DISTANCE":         None,  # filled by Haversine post-process
                    "callsign":         f.get("callsign", "").strip(),
                    "icao24":           f.get("icao24", ""),
                    "duration_min":     duration_min,
                    "_source":          "opensky",
                    "_date":            yesterday.strftime("%Y-%m-%d"),
                })
        except Exception as e:
            print(f"  [warn] {icao}: {e}")
    return rows

def _hour_block(ts):
    if not ts: return "0600-0659"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    h = dt.hour
    return f"{h:02d}00-{h:02d}59"

if __name__ == "__main__":
    print("Collecting yesterday's OpenSky flight data...")
    rows = fetch_yesterday_flights()
    if not rows:
        print("No rows collected — OpenSky may be rate-limited. Exiting.")
        raise SystemExit(0)
    df = pd.DataFrame(rows)
    out = Path("data/opensky_collected.csv")
    if out.exists():
        existing = pd.read_csv(out)
        df = pd.concat([existing, df], ignore_index=True)
        # Deduplicate by (icao24, _date)
        df = df.drop_duplicates(subset=["icao24", "_date"])
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} total rows to {out}")
```

### Updated `retrain.yml` — add data collection step

```yaml
# Add BEFORE the "Run incremental model update" step:
- name: Collect yesterday's OpenSky flight data
  run: python collect_opensky_data.py

# Also install requests in the deps step:
pip install "scikit-learn>=1.3" "lightgbm>=4.0.0" joblib numpy pandas requests
```

### Limitations of OpenSky for labeling
- OpenSky doesn't provide scheduled departure times, so exact delay minutes can't be
  computed from OpenSky alone. It gives `firstSeen` (actual wheels-up) and `lastSeen`
  (landing) timestamps.
- For true labeled data (with delay minutes), the **BTS On-Time Performance** dataset
  is the gold standard — it's updated monthly at:
  `https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ`
- Recommend: download BTS monthly data in the GitHub Action (or a separate monthly cron)
  and append to training data for ground-truth labels.

---

## 5. NEW FEATURE — Google Flights / Flight Search Integration

**Important:** Google Flights does **not** have a free official API.

Options ranked by feasibility:

### Option A — SerpAPI Google Flights (recommended, ~100 free searches/month)
- Sign up at https://serpapi.com (free tier: 100 searches/month)
- API key in GitHub Secrets as `SERPAPI_KEY` and Render env var
- Add to `api/services/flights.py`:

```python
import httpx

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

async def search_google_flights(origin_iata: str, dest_iata: str, date: str):
    """Search Google Flights via SerpAPI for real flight schedules."""
    if not SERPAPI_KEY:
        return []
    params = {
        "engine":          "google_flights",
        "departure_id":    origin_iata,
        "arrival_id":      dest_iata,
        "outbound_date":   date,          # YYYY-MM-DD
        "currency":        "USD",
        "hl":              "en",
        "api_key":         SERPAPI_KEY,
        "type":            "2",           # 2 = one way
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://serpapi.com/search", params=params)
        r.raise_for_status()
        data = r.json()
    flights = []
    for itinerary in data.get("best_flights", []) + data.get("other_flights", []):
        for leg in itinerary.get("flights", []):
            flights.append({
                "airline":         leg.get("airline"),
                "flight_number":   leg.get("flight_number"),
                "departure_time":  leg.get("departure_airport", {}).get("time"),
                "arrival_time":    leg.get("arrival_airport", {}).get("time"),
                "duration_min":    leg.get("duration"),
                "price":           itinerary.get("price"),
            })
    return flights
```

Add new endpoint to `api/main.py`:
```python
@app.get("/flights/search")
async def search_flights(origin: str, dest: str, date: str):
    """Search real flight schedules via Google Flights (SerpAPI)."""
    flights = await search_google_flights(origin, dest, date)
    return {"flights": flights, "count": len(flights)}
```

### Option B — Amadeus Flight Offers API (free tier: 2000 calls/month)
- Sign up at https://developers.amadeus.com (free self-service tier)
- Env vars: `AMADEUS_API_KEY`, `AMADEUS_API_SECRET`
- SDK: `pip install amadeus`
- Returns real schedules with carrier, departure time, price

### Option C — aviationstack (free tier: 100 calls/month)
- https://aviationstack.com — free plan limited to real-time flights only
- Gives actual departure/arrival times for live flights

**Recommendation:** Use **Amadeus** for flight search (better free quota) and **SerpAPI**
as a fallback for price comparison. For the thesis, mock the data if API quotas are hit.

---

## 6. Domestic Flights — Expand Coverage

The model already trains on US domestic routes. The OpenSky collector in step 4 filters
to US airport pairs. The frontend `ALL_CITIES` list already covers ~60 US cities.

**Missing pieces for full domestic coverage:**
1. Add BTS carrier codes for regional carriers that do domestic-only (Allegiant, Sun Country,
   Breeze Airways, Avelo Airlines)
2. The `DISTANCES` lookup in `Home.jsx` covers only ~30 pairs — step 2 (Haversine fallback)
   covers all remaining pairs
3. The `meta/options` endpoint in the API should return the full carrier + city lists from
   the training data — verify `api/main.py`'s `/meta/options` endpoint returns all carriers

---

## Priority Order

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | Add `vercel.json` SPA routing fix | 5 min | **Critical** — navigation completely broken |
| 2 | Fix distance auto-update + Haversine fallback | 30 min | High — core UX bug |
| 3 | Fix LiveMap start/end point selection | 1 hr | Medium |
| 4 | OpenSky daily data collection workflow | 2 hr | High — recursive training |
| 5 | Google Flights / Amadeus API integration | 2 hr | Medium |
| 6 | Domestic coverage expansion | 30 min | Low — already mostly covered |

---

## Files to Create/Modify

| File | Change |
|------|--------|
| `frontend/vercel.json` | **CREATE** — SPA routing rewrites |
| `frontend/src/pages/Home.jsx` | Edit — Haversine fallback, auto-update distance via useEffect |
| `frontend/src/pages/LiveMap.jsx` | Edit — origin/dest filter or click-to-predict |
| `collect_opensky_data.py` | **CREATE** — daily data collector |
| `.github/workflows/retrain.yml` | Edit — add data collection step |
| `api/main.py` | Edit — add `/flights/search` endpoint |
| `api/services/flights.py` | Edit — add `search_google_flights()` |
| `api/requirements.txt` | Edit — add `amadeus` or `serpapi` |
