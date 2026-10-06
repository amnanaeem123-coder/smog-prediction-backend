"""
Quick forecast script for the Smog Risk Prediction System.

What it does (takes about 1 minute):
    1. Loads the ALREADY TRAINED model (no retraining).
    2. Gets weather + pollution forecasts from OpenWeatherMap.
    3. Predicts visibility for 4 cities about 5 hours ahead.
    4. Saves latest_risk.json for the app.

How to run (Anaconda Prompt):
    cd "C:\\Users\\Admin\\fyp project"
    python run_forecast.py

API key:
    Put your OpenWeatherMap key in a file called api_key.txt
    in the same folder as this script. Only the key, nothing else.
    Never upload api_key.txt to GitHub.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\FYP_Dataset\xgb_model_with_pollution_final.pkl"

# Frontend app folder (the live server reads this file)
SAVE_PATH = r"C:\Users\Admin\Desktop\smog-risk-prediction-system\assets\data\latest_risk.json"
# Backup copy
BACKUP_PATH = r"C:\Users\Admin\OneDrive\Desktop\FYP_Dataset\latest_risk.json"

FORECAST_HOURS = 5

city_coords = {
    'Lahore':     [31.5204, 74.3587],
    'Islamabad':  [33.6844, 73.0479],
    'Faisalabad': [31.4504, 73.1350],
    'Multan':     [30.1575, 71.5249]
}

# ------------------------------------------------------------
# LOAD API KEY AND MODEL
# ------------------------------------------------------------
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
if not os.path.exists(KEY_FILE):
    raise SystemExit(f"api_key.txt not found. Create it here: {KEY_FILE}")
with open(KEY_FILE, "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()

model = joblib.load(MODEL_PATH)


def safe(err):
    """Hide the API key in error messages so it never shows on screen."""
    return str(err).replace(API_KEY, "***KEY***")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def get_current_visibility(lat, lon):
    """Get current visibility, used as visibility_prev_reading."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['visibility'] / 1000  # meters -> km
    except Exception as e:
        print(f"  WARNING: current weather API failed ({safe(e)}). Using fallback value of 10 km (SAFE default).")
        return 10.0


def closest_entry_to(entries, target_time, time_key='dt'):
    """Return (entry, time) of the entry closest to target_time. Times converted to Pakistan local time."""
    best, best_time, best_diff = None, None, None
    for entry in entries:
        entry_time = datetime.fromtimestamp(entry[time_key])
        diff = abs((entry_time - target_time).total_seconds())
        if best_diff is None or diff < best_diff:
            best, best_time, best_diff = entry, entry_time, diff
    return best, best_time


def get_hourly_forecast_entry(lat, lon, target_time):
    """Paid One Call API 4.0, 1-hour step forecast."""
    try:
        url = (f"https://api.openweathermap.org/data/4.0/onecall/timeline/1h"
               f"?lat={lat}&lon={lon}&units=metric&appid={API_KEY}")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  WARNING: hourly One Call API failed ({safe(e)}).")
        return None, None

    entry, entry_time = closest_entry_to(data.get('data', []), target_time)
    if entry is None:
        return None, None

    weather = {
        'temp': entry['temp'],
        'humidity': entry['humidity'],
        'dew': entry['dew_point'],
        'windspeed_kmh': entry['wind_speed'] * 3.6,
        'windgust_kmh': entry.get('wind_gust', 0) * 3.6,
        'winddir': entry.get('wind_deg', 0),
        'pressure': entry['pressure'],
        'cloudcover': entry['clouds'],
        'precip': entry.get('rain', {}).get('1h', 0),
        'snow': entry.get('snow', {}).get('1h', 0),
    }
    return weather, entry_time


def get_3hour_forecast_entry(lat, lon, target_time):
    """Backup: free 3-hour forecast, used only if the hourly API fails."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR: 3-hour forecast API also failed ({safe(e)}).")
        return None, None

    entry, entry_time = closest_entry_to(data['list'], target_time)
    if entry is None:
        return None, None

    weather = {
        'temp': entry['main']['temp'],
        'humidity': entry['main']['humidity'],
        'dew': entry['main']['dew_point'],
        'windspeed_kmh': entry['wind']['speed'] * 3.6,
        'windgust_kmh': entry['wind'].get('gust', 0) * 3.6,
        'winddir': entry['wind'].get('deg', 0),
        'pressure': entry['main'].get('sea_level', entry['main']['pressure']),
        'cloudcover': entry['clouds']['all'],
        'precip': entry.get('rain', {}).get('3h', 0),
        'snow': entry.get('snow', {}).get('3h', 0),
    }
    return weather, entry_time


def get_pollution_forecast_entry(lat, lon, target_time):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR: pollution forecast API failed ({safe(e)}).")
        return None

    entry, _ = closest_entry_to(data['list'], target_time)
    return entry


def build_feature_row(city, lat, lon):
    now = datetime.now()
    wanted_time = now + pd.Timedelta(hours=FORECAST_HOURS)
    print(f"  Now: {now:%H:%M} | Target: {wanted_time:%H:%M}")

    prev_visibility = get_current_visibility(lat, lon)

    weather, target_time = get_hourly_forecast_entry(lat, lon, wanted_time)
    source = "One Call 4.0 (hourly)"
    if weather is None:
        weather, target_time = get_3hour_forecast_entry(lat, lon, wanted_time)
        source = "Free forecast (3-hour)"
    if weather is None:
        return None, None

    print(f"  Weather source: {source} | Forecast slot used: {target_time:%Y-%m-%d %H:%M}")

    pollution_entry = get_pollution_forecast_entry(lat, lon, target_time)
    if pollution_entry is None:
        return None, None

    temp, dew, winddir = weather['temp'], weather['dew'], weather['winddir']

    row = {
        'temp': temp,
        'humidity': weather['humidity'],
        'dew': dew,
        'windspeed': weather['windspeed_kmh'],
        'windgust': weather['windgust_kmh'],
        'winddir': winddir,
        'sealevelpressure': weather['pressure'],
        'cloudcover': weather['cloudcover'],
        'precip': weather['precip'],
        'snow': weather['snow'],
        'snowdepth': 0,
        'month': target_time.month,
        'hour': target_time.hour,
        'is_smog_prone_hour': 1 if 4 <= target_time.hour <= 9 else 0,
        'dew_point_depression': temp - dew,
        'city_Faisalabad': 1 if city == 'Faisalabad' else 0,
        'city_Islamabad': 1 if city == 'Islamabad' else 0,
        'city_Lahore': 1 if city == 'Lahore' else 0,
        'city_Multan': 1 if city == 'Multan' else 0,
        'visibility_prev_reading': prev_visibility,
        'winddir_sin': np.sin(np.radians(winddir)),
        'winddir_cos': np.cos(np.radians(winddir)),
        'aqi': pollution_entry['main']['aqi'],
        'co': pollution_entry['components']['co'],
        'no': pollution_entry['components']['no'],
        'no2': pollution_entry['components']['no2'],
        'o3': pollution_entry['components']['o3'],
        'so2': pollution_entry['components']['so2'],
        'pm2_5': pollution_entry['components']['pm2_5'],
        'pm10': pollution_entry['components']['pm10'],
        'nh3': pollution_entry['components']['nh3'],
    }
    return row, target_time


def classify_risk(v):
    if v < 0.5: return "CRITICAL"
    elif v < 2.0: return "HIGH"
    elif v < 5.0: return "MODERATE"
    elif v < 10.0: return "LOW"
    else: return "SAFE"


recommendation_map = {
    "CRITICAL": "Motorway Closure Required",
    "HIGH": "Speed Limit 40 km/h — Caution Required",
    "MODERATE": "Speed Limit 60 km/h — Caution Required",
    "LOW": "Reduced Speed Advised — Monitor Conditions",
    "SAFE": "Normal Operations"
}

feature_order = ['temp', 'humidity', 'dew', 'windspeed', 'windgust', 'winddir',
                 'sealevelpressure', 'cloudcover', 'precip', 'snow', 'snowdepth',
                 'month', 'hour', 'is_smog_prone_hour', 'dew_point_depression',
                 'city_Faisalabad', 'city_Islamabad', 'city_Lahore', 'city_Multan',
                 'visibility_prev_reading', 'winddir_sin', 'winddir_cos',
                 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']


# ------------------------------------------------------------
# MAKE PREDICTIONS
# ------------------------------------------------------------
results = []

for city, (lat, lon) in city_coords.items():
    print(f"Fetching forecast for {city}...")
    row, target_time = build_feature_row(city, lat, lon)

    if row is None:
        print(f"  SKIPPING {city}: forecast data unavailable.")
        results.append({'city': city, 'risk_level': 'UNAVAILABLE'})
        continue

    X_row = pd.DataFrame([row])[feature_order]
    predicted_visibility = float(model.predict(X_row)[0])
    risk = classify_risk(predicted_visibility)

    results.append({
        'city': city,
        'datetime': str(target_time),
        'visibility_km': round(predicted_visibility, 2),
        'risk_level': risk,
        'recommendation': recommendation_map[risk]
    })
    print(f"  {city}: {predicted_visibility:.2f} km -> {risk}")


# ------------------------------------------------------------
# SAVE latest_risk.json (only if ALL cities worked)
# ------------------------------------------------------------
if any(r['risk_level'] == 'UNAVAILABLE' for r in results):
    print("\nWARNING: some cities have no forecast. JSON NOT updated, old file kept.")
else:
    for path in (SAVE_PATH, BACKUP_PATH):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nlatest_risk.json updated!")
    print(json.dumps(results, indent=2, ensure_ascii=False))
