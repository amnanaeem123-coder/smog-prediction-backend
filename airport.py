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
"""

import csv
import io
from datetime import datetime, timedelta

import requests

AIRPORTS = {
    "Lahore": "OPLA",       # Allama Iqbal International
    "Islamabad": "OPIS",    # Islamabad International
    "Faisalabad": "OPFA",   # Faisalabad International
    "Multan": "OPMT",       # Multan International
}

CAP_KM = 10.0            # airports report 10 km and more as 9999
MILES_TO_KM = 1.609344
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def download(city, start, end):
    """All airport reports for a city between two dates (Pakistan time).
    Returns a list of (time, visibility_km), oldest first."""
    params = [
        ("station", AIRPORTS[city]), ("data", "vsby"),
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end.year), ("month2", end.month), ("day2", end.day),
        ("tz", "Asia/Karachi"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),
    ]
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    reports = []
    for line in csv.DictReader(io.StringIO(r.text)):
        try:
            when = datetime.strptime(line["valid"], "%Y-%m-%d %H:%M")
            km = float(line["vsby"]) * MILES_TO_KM
        except (KeyError, ValueError, TypeError):
            continue
        reports.append((when, round(min(km, CAP_KM), 2)))
    reports.sort()
    return reports


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
