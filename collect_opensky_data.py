"""
collect_opensky_data.py — Daily OpenSky flight data collector.

Pulls yesterday's departure records from OpenSky Network's free REST API for major
US domestic airports and appends them to data/opensky_collected.csv.

Run by GitHub Actions daily before train_incremental.py so the model trains on fresh
real-world data. OpenSky's /flights/departure endpoint requires no API key for public
data but is rate-limited (~1 req/10 s per IP for anonymous access).

Note on delay labels:
  OpenSky doesn't provide scheduled departure times, so exact delay minutes are unknown.
  We record raw flight metadata only. Ground-truth labels come from POST /feedback
  (users reporting actual outcomes) and from the monthly BTS On-Time Performance dataset.
"""

import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

OPENSKY_BASE = "https://opensky-network.org/api"

# Major US domestic airport ICAO → city name (model's city vocabulary)
US_AIRPORTS = {
    "KATL": "atlanta",          "KORD": "chicago",          "KLAX": "los angeles",
    "KDFW": "dallas-fort worth","KDEN": "denver",            "KJFK": "new york",
    "KSFO": "san francisco",    "KSEA": "seattle",           "KLAS": "las vegas",
    "KMCO": "orlando",          "KEWR": "newark",            "KPHX": "phoenix",
    "KIAH": "houston",          "KMIA": "miami",             "KBOS": "boston",
    "KMSP": "minneapolis",      "KDTW": "detroit",           "KPHL": "philadelphia",
    "KLGA": "new york",         "KBWI": "baltimore",         "KDAL": "dallas",
    "KHOU": "houston",          "KSLC": "salt lake city",    "KSAN": "san diego",
    "KTPA": "tampa",            "KPDX": "portland",          "KSTL": "st. louis",
    "KBNA": "nashville",        "KCLT": "charlotte",         "KAUS": "austin",
    "KSAT": "san antonio",      "KMEM": "memphis",           "KMSY": "new orleans",
    "KRDU": "raleigh-durham",   "KPIT": "pittsburgh",        "KCLE": "cleveland",
    "KRIC": "richmond",         "KCMH": "columbus",          "KIND": "indianapolis",
    "KMKE": "milwaukee",        "KOAK": "oakland",           "KSJC": "san jose",
    "KSMF": "sacramento",       "KRSW": "fort myers",        "KFLL": "fort lauderdale",
    "KPBI": "west palm beach",  "KHNL": "honolulu",          "TJSJ": "san juan",
    "KORF": "norfolk",           "KDAB": "daytona beach",
}
ICAO_SET = set(US_AIRPORTS.keys())

OUT_PATH = Path(__file__).parent / "data" / "opensky_collected.csv"
FIELDNAMES = [
    "date", "icao24", "callsign",
    "origin_icao", "origin_city",
    "dest_icao", "dest_city",
    "dep_hour", "duration_min",
]


def fetch_departures(airport_icao: str, begin: int, end: int) -> list[dict]:
    """Fetch departure records for one airport from OpenSky."""
    url = f"{OPENSKY_BASE}/flights/departure"
    try:
        r = requests.get(url, params={"airport": airport_icao, "begin": begin, "end": end}, timeout=30)
        if r.status_code == 200:
            return r.json() or []
        if r.status_code == 404:
            return []   # no flights for this airport/window — normal
        print(f"  [warn] {airport_icao}: HTTP {r.status_code}", flush=True)
        return []
    except requests.RequestException as exc:
        print(f"  [warn] {airport_icao}: {exc}", flush=True)
        return []


def dep_hour(first_seen_ts: int | None) -> int | None:
    if not first_seen_ts:
        return None
    return datetime.fromtimestamp(first_seen_ts, tz=timezone.utc).hour


def main() -> None:
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    date_str  = yesterday.strftime("%Y-%m-%d")
    begin     = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                             0, 0, 0, tzinfo=timezone.utc).timestamp())
    end       = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                             23, 59, 59, tzinfo=timezone.utc).timestamp())

    print(f"Collecting OpenSky departures for {date_str}…", flush=True)

    # Load existing records to avoid duplicates
    existing_keys: set[tuple[str, str]] = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row.get("date", ""), row.get("icao24", "")))

    new_rows: list[dict] = []
    failed = 0

    for i, (icao, city) in enumerate(US_AIRPORTS.items()):
        flights = fetch_departures(icao, begin, end)
        if not flights:
            failed += 1
        for f in flights:
            dest_icao = f.get("estArrivalAirport", "")
            if dest_icao not in ICAO_SET:
                continue   # keep domestic-only pairs

            icao24    = f.get("icao24", "").strip()
            key       = (date_str, icao24)
            if key in existing_keys:
                continue   # deduplicate

            first_seen = f.get("firstSeen")
            last_seen  = f.get("lastSeen")
            dur = (
                round((last_seen - first_seen) / 60)
                if first_seen and last_seen else None
            )

            new_rows.append({
                "date"        : date_str,
                "icao24"      : icao24,
                "callsign"    : (f.get("callsign") or "").strip(),
                "origin_icao" : icao,
                "origin_city" : city,
                "dest_icao"   : dest_icao,
                "dest_city"   : US_AIRPORTS.get(dest_icao, ""),
                "dep_hour"    : dep_hour(first_seen),
                "duration_min": dur,
            })
            existing_keys.add(key)

        # Respect OpenSky rate limit (~1 req / 10 s for anonymous)
        if i < len(US_AIRPORTS) - 1:
            time.sleep(10)

    total_airports = len(US_AIRPORTS)
    print(f"  Collection summary: {total_airports - failed}/{total_airports} airports responded, "
          f"{failed} failed/empty (rate-limit or no flights).", flush=True)

    if not new_rows:
        print("No new rows collected (rate-limited or no flights). Continuing.", flush=True)
        sys.exit(0)

    write_header = not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"  Appended {len(new_rows)} new flight records → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
