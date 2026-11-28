"""
airport.py
Real visibility measured at the airports (METAR reports).

OpenWeatherMap reports 10 km most of the time, even in heavy smog.
Airports measure visibility with instruments every 30 minutes.
So we use the airports for:
    1. the "visibility now" input of the model (run_forecast.py)
    2. the real visibility used to check predictions (verify.py)
    3. the current conditions shown in the app (current_conditions.py)

Data: Iowa Environmental Mesonet (free, no key), which keeps all METAR reports.
All 4 airports are downloaded in ONE request, and the answer is kept in memory,
because the free service blocks many quick requests (error 429).
"""

import csv
import io
import time
from datetime import datetime, timedelta

import requests

AIRPORTS = {
    "Lahore": "OPLA",       # Allama Iqbal International
    "Islamabad": "OPIS",    # Islamabad International
    "Faisalabad": "OPFA",   # Faisalabad International
    "Multan": "OPMT",       # Multan International
}
CODE_TO_CITY = {code: city for city, code in AIRPORTS.items()}

CAP_KM = 10.0            # airports report 10 km and more as 9999
MILES_TO_KM = 1.609344
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_cache = {}   # (start date, end date) -> {city: [(time, km), ...]}


def download_all(start, end):
    """Reports of all 4 airports between two dates (Pakistan time),
    in one request. Returns {city: [(time, visibility_km), ...]}, oldest first."""
    key = (start.date(), end.date())
    if key in _cache:
        return _cache[key]

    params = [("station", code) for code in AIRPORTS.values()]
    params += [
        ("data", "vsby"),
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("tz", "Asia/Karachi"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),
    ]

    # Try up to 4 times. If the service says "too many requests", wait and retry.
    response = None
    for attempt in range(4):
        response = requests.get(URL, params=params, timeout=90)
        if response.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"  Airport service busy, waiting {wait} seconds...")
            time.sleep(wait)
            continue
        break
    response.raise_for_status()

    result = {city: [] for city in AIRPORTS}
    for line in csv.DictReader(io.StringIO(response.text)):
        city = CODE_TO_CITY.get((line.get("station") or "").strip())
        if city is None:
            continue
        try:
            when = datetime.strptime(line["valid"], "%Y-%m-%d %H:%M")
            km = float(line["vsby"]) * MILES_TO_KM
        except (KeyError, ValueError, TypeError):
            continue
        result[city].append((when, round(min(km, CAP_KM), 2)))
    for city in result:
        result[city].sort()

    _cache[key] = result
    return result


def download(city, start, end):
    """Reports of one airport (uses the shared download above)."""
    return download_all(start, end)[city]


def nearest(reports, target, max_minutes=40):
    """Visibility of the report closest to target, if within max_minutes."""
    best, best_gap = None, None
    for when, km in reports:
        gap = abs((when - target).total_seconds()) / 60
        if gap <= max_minutes and (best_gap is None or gap < best_gap):
            best, best_gap = km, gap
    return best


def latest(city, max_age_hours=3):
    """Most recent airport visibility for a city, or None if there is no
    report from the last few hours."""
    now = datetime.now()
    reports = download(city, now - timedelta(days=1), now + timedelta(days=1))
    recent = [(w, km) for w, km in reports
              if timedelta(0) <= now - w <= timedelta(hours=max_age_hours)]
    return recent[-1][1] if recent else None
