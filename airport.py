"""
airport.py
Real measurements from the airports (METAR reports).

OpenWeatherMap reports 10 km visibility most of the time, even in heavy smog,
and its forecasts are often far too dry for smoggy Punjab nights.
Airports measure visibility, temperature and dew point with instruments
every 30 minutes. So we use the airports for:
    1. the "visibility now" input of the model (run_forecast.py)
    2. correcting the forecast temperature and dew point (run_forecast.py)
    3. the real visibility used to check predictions (verify.py)
    4. the current conditions shown in the app (current_conditions.py)

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

_cache = {}   # (start date, end date) -> {"vis": {...}, "met": {...}}


def _f_to_c(f):
    return (f - 32) * 5 / 9


def _download(start, end):
    """One request for all 4 airports. Returns
    {"vis": {city: [(time, km)]}, "met": {city: [(time, temp_c, dew_c)]}}"""
    key = (start.date(), end.date())
    if key in _cache:
        return _cache[key]

    params = [("station", code) for code in AIRPORTS.values()]
    params += [
        ("data", "vsby"), ("data", "tmpf"), ("data", "dwpf"),
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

    vis = {city: [] for city in AIRPORTS}
    met = {city: [] for city in AIRPORTS}
    for line in csv.DictReader(io.StringIO(response.text)):
        city = CODE_TO_CITY.get((line.get("station") or "").strip())
        if city is None:
            continue
        try:
            when = datetime.strptime(line["valid"], "%Y-%m-%d %H:%M")
        except (KeyError, ValueError, TypeError):
            continue
        try:
            km = float(line["vsby"]) * MILES_TO_KM
            vis[city].append((when, round(min(km, CAP_KM), 2)))
        except (KeyError, ValueError, TypeError):
            pass
        try:
            met[city].append((when, round(_f_to_c(float(line["tmpf"])), 1),
                              round(_f_to_c(float(line["dwpf"])), 1)))
        except (KeyError, ValueError, TypeError):
            pass
    for city in AIRPORTS:
        vis[city].sort()
        met[city].sort()

    _cache[key] = {"vis": vis, "met": met}
    return _cache[key]


def download_all(start, end):
    """Visibility reports of all 4 airports: {city: [(time, km), ...]}."""
    return _download(start, end)["vis"]


def download(city, start, end):
    """Visibility reports of one airport."""
    return download_all(start, end)[city]


def nearest(reports, target, max_minutes=40):
    """Visibility of the report closest to target, if within max_minutes."""
    best, best_gap = None, None
    for when, km in reports:
        gap = abs((when - target).total_seconds()) / 60
        if gap <= max_minutes and (best_gap is None or gap < best_gap):
            best, best_gap = km, gap
    return best


def _recent(items, max_age_hours):
    now = datetime.now()
    return [x for x in items if timedelta(0) <= now - x[0] <= timedelta(hours=max_age_hours)]


def latest(city, max_age_hours=3):
    """Most recent airport visibility (km) for a city, or None."""
    now = datetime.now()
    recent = _recent(download(city, now - timedelta(days=1), now + timedelta(days=1)), max_age_hours)
    return recent[-1][1] if recent else None


def latest_met(city, max_age_hours=3):
    """Most recent airport temperature and dew point (deg C) for a city,
    as (time, temp_c, dew_c), or None."""
    now = datetime.now()
    met = _download(now - timedelta(days=1), now + timedelta(days=1))["met"][city]
    recent = _recent(met, max_age_hours)
    return recent[-1] if recent else None
