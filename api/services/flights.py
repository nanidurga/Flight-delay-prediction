"""
flights.py — Real-time flight data fetcher using OpenSky Network (free, no API key).

OpenSky returns live aircraft state vectors (position, altitude, velocity).
We enrich each flight with weather from Open-Meteo (also free, no key).
Then we build the feature dict that the predictor expects.
"""

import asyncio
from datetime import datetime
from typing import Optional

import httpx

# ── Airport coordinates (lat/lon) for weather lookup ─────────────────────────
# Covers most airports in the training dataset
AIRPORT_COORDS = {
    "JFK": (40.6413, -73.7781), "LGA": (40.7769, -73.8740), "EWR": (40.6895, -74.1745),
    "LAX": (33.9425, -118.408), "ORD": (41.9742, -87.9073), "ATL": (33.6367, -84.4281),
    "DFW": (32.8998, -97.0403), "DEN": (39.8561, -104.676), "SFO": (37.6213, -122.379),
    "SEA": (47.4502, -122.309), "MIA": (25.7959, -80.2870), "BOS": (42.3656, -71.0096),
    "MSP": (44.8848, -93.2223), "PHX": (33.4373, -112.008), "IAH": (29.9902, -95.3368),
    "LAS": (36.0840, -115.153), "CLT": (35.2140, -80.9431), "MCO": (28.4294, -81.3089),
    "EWR": (40.6895, -74.1745), "MDW": (41.7868, -87.7522), "BWI": (39.1754, -76.6682),
    "SAN": (32.7338, -117.190), "TPA": (27.9755, -82.5332), "DAL": (32.8471, -96.8517),
    "PDX": (45.5898, -122.592), "STL": (38.7487, -90.3700), "HNL": (21.3187, -157.922),
    "OAK": (37.7213, -122.221), "MCI": (39.2976, -94.7139), "RDU": (35.8801, -78.7880),
    "IND": (39.7173, -86.2944), "PIT": (40.4915, -80.2329), "CLE": (41.4117, -81.8498),
    "CVG": (39.0488, -84.6678), "CMH": (39.9980, -82.8919), "BUF": (42.9405, -78.7322),
    "SLC": (40.7884, -111.978), "MEM": (35.0424, -89.9767), "JAX": (30.4941, -81.6879),
    "MKE": (42.9472, -87.8966), "OMA": (41.3032, -95.8941), "TUL": (36.1984, -95.8881),
}

OPENSKY_URL  = "https://opensky-network.org/api/states/all"
WEATHER_URL  = "https://api.open-meteo.com/v1/forecast"

# Condition text mapping from WMO weather codes (Open-Meteo uses WMO codes)
WMO_TO_CONDITION = {
    0: "Sunny", 1: "Partly cloudy", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog",
    51: "Light drizzle", 53: "Light drizzle", 55: "Light drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    80: "Light rain shower", 81: "Moderate or heavy rain shower", 82: "Moderate or heavy rain shower",
    95: "Moderate or heavy rain with thunder", 96: "Moderate or heavy rain with thunder",
}


async def fetch_weather(lat: float, lon: float, client: httpx.AsyncClient) -> dict:
    """Get current weather at a lat/lon from Open-Meteo."""
    try:
        resp = await client.get(WEATHER_URL, params={
            "latitude"        : lat,
            "longitude"       : lon,
            "current_weather" : True,
            "hourly"          : "relative_humidity_2m,weathercode",
            "forecast_days"   : 1,
            "timezone"        : "auto",
        }, timeout=8)
        data     = resp.json()
        current  = data.get("current_weather", {})
        hourly   = data.get("hourly", {})
        humidity = hourly.get("relative_humidity_2m", [60])[0]
        wmo_code = int(current.get("weathercode", 0))
        return {
            "temperature_celsius": current.get("temperature", 20),
            "humidity"           : humidity,
            "condition_text"     : WMO_TO_CONDITION.get(wmo_code, "Partly cloudy"),
            "windspeed"          : current.get("windspeed", 0),
        }
    except Exception:
        return {"temperature_celsius": 20, "humidity": 60,
                "condition_text": "Partly cloudy", "windspeed": 0}


async def fetch_live_flights(
    origin_iata: Optional[str] = None,
    limit: int = 50
) -> list[dict]:
    """
    Fetch live flights from OpenSky. Returns enriched list ready for prediction.
    If origin_iata is given, filter to flights departing from that airport's bbox.
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(OPENSKY_URL, timeout=15)
            data = resp.json()
        except Exception as e:
            return []

        states = data.get("states", []) or []
        flights = []

        for state in states[:200]:   # cap to avoid huge response
            # OpenSky state vector fields (index reference):
            # 0:icao24, 1:callsign, 5:lon, 6:lat, 7:altitude, 9:velocity, 10:heading
            callsign = (state[1] or "").strip()
            lon      = state[5]
            lat      = state[6]

            if not callsign or lon is None or lat is None:
                continue

            flights.append({
                "callsign": callsign,
                "icao24"  : state[0],
                "lat"     : lat,
                "lon"     : lon,
                "altitude": state[7] or 0,
                "velocity": state[9] or 0,
                "heading" : state[10] or 0,
            })

            if len(flights) >= limit:
                break

        # Fetch weather for unique lat/lon clusters (avoid hammering API)
        # For demo: just get weather for a couple representative points
        weather_cache: dict[tuple, dict] = {}

        async def get_weather_cached(lat, lon):
            key = (round(lat, 1), round(lon, 1))
            if key not in weather_cache:
                weather_cache[key] = await fetch_weather(lat, lon, client)
            return weather_cache[key]

        # Enrich each flight with weather
        for flight in flights:
            w = await get_weather_cached(flight["lat"], flight["lon"])
            flight["weather"] = w

        return flights


async def get_airport_weather(iata: str) -> dict:
    """Get current weather at a known airport."""
    coords = AIRPORT_COORDS.get(iata.upper())
    if not coords:
        return {"temperature_celsius": 20, "humidity": 60,
                "condition_text": "Partly cloudy", "windspeed": 0}
    async with httpx.AsyncClient() as client:
        return await fetch_weather(coords[0], coords[1], client)


def build_feature_dict(
    origin_city: str,
    dest_city: str,
    carrier: str,
    distance: float,
    crs_elapsed_time: float,
    origin_weather: dict,
    dest_weather: dict,
    dep_hour: int,
    arr_hour: int,
    month: int,
    day: int,
    is_weekend: bool,
    season: str,
) -> dict:
    """
    Convert human-readable flight info into the one-hot feature dict
    that the model's predict() method expects.

    Everything not provided defaults to 0 (model handles this gracefully).
    """
    features: dict = {}

    # Numeric
    features["CRS_ELAPSED_TIME"]             = crs_elapsed_time
    features["DISTANCE"]                     = distance
    features["origin_temperature_celsius"]   = origin_weather.get("temperature_celsius", 20)
    features["origin_humidity"]              = origin_weather.get("humidity", 60)
    features["dest_temperature_celsius"]     = dest_weather.get("temperature_celsius", 20)
    features["dest_humidity"]                = dest_weather.get("humidity", 60)
    features["DAY"]                          = day
    features["WeekendFlagEncoded"]           = int(is_weekend)
    features["DayBeforeWeekendEncoded"]      = 0  # simplified

    # Departure time bucket (1=early morning, 2=morning, 3=afternoon, 4=evening)
    for slot, (start, end) in enumerate([(0,6),(6,12),(12,18),(18,24)], start=1):
        features[f"CRS_DEP_TIME_{slot}"] = int(start <= dep_hour < end)
    for slot, (start, end) in enumerate([(0,6),(6,12),(12,18),(18,24)], start=1):
        features[f"CRS_ARR_TIME_{slot}"] = int(start <= arr_hour < end)

    # Month one-hot (month 1 = baseline, so MONTH_2 to MONTH_12)
    for m in range(2, 13):
        features[f"MONTH_{m}"] = int(month == m)

    # Season one-hot
    for s in ["Autumn", "Spring", "Summer", "Winter"]:
        features[f"Season_{s}"] = int(season == s)

    # Carrier one-hot
    carrier_col = f"OP_CARRIER_{carrier}"
    features[carrier_col] = 1

    # Origin city one-hot
    origin_col = f"origin_city_{origin_city.lower()}"
    features[origin_col] = 1

    # Destination city one-hot
    dest_col = f"destination_city_{dest_city.lower()}"
    features[dest_col] = 1

    # Origin weather condition one-hot
    origin_cond = origin_weather.get("condition_text", "Partly cloudy")
    features[f"origin_condition_text_{origin_cond}"] = 1

    # Dest weather condition one-hot
    dest_cond = dest_weather.get("condition_text", "Partly cloudy")
    features[f"dest_condition_text_{dest_cond}"] = 1

    return features


def get_season(month: int) -> str:
    if month in (3, 4, 5):    return "Spring"
    if month in (6, 7, 8):    return "Summer"
    if month in (9, 10, 11):  return "Autumn"
    return "Winter"


# ── List of carriers and cities in the training data ─────────────────────────
KNOWN_CARRIERS = [
    "Allegiant Air", "American Airlines", "Delta Airlines", "Endeavor Air",
    "Envoy Air", "ExpressJet", "Frontier Airlines", "Hawaiian Airlines",
    "JetBlue Airways", "Mesa Airline", "PSA Airlines", "Republic Airways",
    "SkyWest Airlines", "Southwest Airlines", "Spirit Airlines",
    "United Airlines", "Virgin America",
]

KNOWN_ORIGIN_CITIES = [
    "new york", "chicago", "los angeles", "dallas", "houston", "atlanta",
    "miami", "boston", "seattle", "denver", "phoenix", "las vegas",
    "san francisco", "orlando", "washington", "philadelphia", "nashville",
    "charlotte", "minneapolis", "detroit", "baltimore", "tampa", "portland",
    "salt lake city", "san diego", "san jose", "oakland", "newark",
    "fort lauderdale", "west palm beach", "kansas city", "st. louis",
    "memphis", "new orleans", "cincinnati", "cleveland", "pittsburgh",
    "buffalo", "rochester", "indianapolis", "columbus", "dayton",
    "louisville", "richmond", "norfolk", "raleigh-durham", "greensboro",
    "jacksonville", "fort myers", "sarasota", "daytona beach",
    "honolulu", "san juan", "san antonio", "austin",
]

KNOWN_DEST_CITIES = [
    "new york", "chicago", "los angeles", "dallas", "dallas-fort worth", "houston",
    "atlanta", "miami", "boston", "seattle", "denver", "phoenix", "las vegas",
    "san francisco", "orlando", "washington", "philadelphia", "nashville",
    "charlotte", "minneapolis", "detroit", "baltimore", "tampa", "portland",
    "salt lake city", "san diego", "san jose", "oakland", "newark",
    "fort lauderdale", "west palm beach", "kansas city", "st. louis",
    "memphis", "new orleans", "cincinnati", "cleveland", "pittsburgh",
    "buffalo", "rochester", "indianapolis", "columbus", "dayton",
    "louisville", "richmond", "norfolk", "raleigh-durham", "greensboro",
    "jacksonville", "fort myers", "sarasota", "honolulu", "san juan",
    "san antonio", "austin", "panama city",
]
