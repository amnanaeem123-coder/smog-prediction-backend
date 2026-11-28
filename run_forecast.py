"""
Quick forecast script for the Smog Risk Prediction System.

What it does (takes about 1 minute):
    1. Loads the ALREADY TRAINED models (no retraining).
       The new model (lag5) makes the prediction shown in the app.
       The old model also predicts the same hours in the background,
       so verify.py can compare both on exactly the same hours.
    2. Gets weather + pollution forecasts from OpenWeatherMap.
    3. Predicts visibility for 4 cities about 5 hours ahead.
    4. Saves latest_risk.json for the app.
    5. Adds every prediction to prediction_log.csv (history for testing).
    6. Runs verify.py to check old predictions against what really happened.

How to run (Anaconda Prompt):
    cd "C:\\Users\\Admin\\fyp project"
    python run_forecast.py

API key:
    Put your OpenWeatherMap key in a file called api_key.txt
    in the same folder as this script. Only the key, nothing else.
    Never upload api_key.txt to GitHub.
"""

import csv
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests

import airport

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Main model (shown in the app). Trained with visibility 5 hours before.
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "xgb_model_lag5.pkl")
MODEL_NAME = "lag5"
# Old model. Runs in the background only, for comparison.
OLD_MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "xgb_model_with_pollution_final.pkl")

# Frontend app folder (the live server reads this file)
# On GitHub (cloud) the files are saved inside the repository instead
IN_CLOUD = os.environ.get("GITHUB_ACTIONS") == "true"

SAVE_PATH = r"C:\Users\Admin\Desktop\smog-risk-prediction-system\assets\data\latest_risk.json"
# Backup copy
BACKUP_PATH = r"C:\Users\Admin\OneDrive\Desktop\FYP_Dataset\latest_risk.json"
if IN_CLOUD:
    SAVE_PATH = os.path.join(SCRIPT_DIR, "latest_risk.json")
    BACKUP_PATH = None

FORECAST_HOURS = 5

# History of every prediction (kept next to this script)
LOG_PATH = os.path.join(SCRIPT_DIR, "prediction_log.csv")
LOG_COLUMNS = ["run_time", "city", "target_time", "predicted_km", "predicted_risk",
               "weather_source", "actual_km", "actual_risk", "level_match", "checked_at",
               "model", "old_predicted_km", "old_predicted_risk",
               "input_visibility_km", "input_source", "actual_source"]

city_coords = {
    'Lahore':     [31.5204, 74.3587],
    'Islamabad':  [33.6844, 73.0479],
    'Faisalabad': [31.4504, 73.1350],
    'Multan':     [30.1575, 71.5249]
}

# ------------------------------------------------------------
# LOAD API KEY AND MODEL
# ------------------------------------------------------------
KEY_FILE = os.path.join(SCRIPT_DIR, "api_key.txt")
# In the cloud the key comes from a hidden GitHub Secret, not from a file
API_KEY = os.environ.get("OWM_API_KEY", "").strip()
if not API_KEY:
    if not os.path.exists(KEY_FILE):
        raise SystemExit(f"api_key.txt not found. Create it here: {KEY_FILE}")
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        API_KEY = f.read().strip()

model = joblib.load(MODEL_PATH)
old_model = joblib.load(OLD_MODEL_PATH) if os.path.exists(OLD_MODEL_PATH) else None


def safe(err):
    """Hide the API key in error messages so it never shows on screen."""
    return str(err).replace(API_KEY, "***KEY***")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def get_current_visibility(city, lat, lon):
    """Visibility now, used as visibility_prev_reading.
    First choice: the airport measurement (real instrument).
    Backup: OpenWeatherMap (often stuck at 10 km)."""
    try:
        km = airport.latest(city)
        if km is not None:
            print(f"  Visibility now (airport {airport.AIRPORTS[city]}): {km} km")
            return km, "airport"
        print("  No recent airport report. Using OpenWeatherMap instead.")
    except Exception as e:
        print(f"  Airport data failed ({e}). Using OpenWeatherMap instead.")
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['visibility'] / 1000, "openweather"  # meters -> km
    except Exception as e:
        print(f"  WARNING: current weather API failed ({safe(e)}). Using fallback value of 10 km (SAFE default).")
        return 10.0, "default"


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

    prev_visibility, prev_source = get_current_visibility(city, lat, lon)

    weather, target_time = get_hourly_forecast_entry(lat, lon, wanted_time)
    source = "One Call 4.0 (hourly)"
    if weather is None:
        weather, target_time = get_3hour_forecast_entry(lat, lon, wanted_time)
        source = "Free forecast (3-hour)"
    if weather is None:
        return None, None, None

    print(f"  Weather source: {source} | Forecast slot used: {target_time:%Y-%m-%d %H:%M}")

    pollution_entry = get_pollution_forecast_entry(lat, lon, target_time)
    if pollution_entry is None:
        return None, None, None

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
        '_input_source': prev_source,   # not a model feature, only for the log
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
    return row, target_time, source


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
log_rows = []
run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for city, (lat, lon) in city_coords.items():
    print(f"Fetching forecast for {city}...")
    row, target_time, source = build_feature_row(city, lat, lon)

    if row is None:
        print(f"  SKIPPING {city}: forecast data unavailable.")
        results.append({'city': city, 'risk_level': 'UNAVAILABLE'})
        continue

    X_row = pd.DataFrame([row])[feature_order]
    predicted_visibility = float(model.predict(X_row)[0])
    risk = classify_risk(predicted_visibility)
    old_visibility = float(old_model.predict(X_row)[0]) if old_model is not None else None

    results.append({
        'city': city,
        'datetime': str(target_time),
        'visibility_km': round(predicted_visibility, 2),
        'risk_level': risk,
        'recommendation': recommendation_map[risk]
    })
    print(f"  {city}: {predicted_visibility:.2f} km -> {risk}")

    log_rows.append({
        "run_time": run_time,
        "city": city,
        "target_time": target_time.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_km": round(predicted_visibility, 2),
        "predicted_risk": risk,
        "weather_source": source,
        "model": MODEL_NAME,
        "input_visibility_km": row['visibility_prev_reading'],
        "input_source": row.get('_input_source', ''),
        "old_predicted_km": round(old_visibility, 2) if old_visibility is not None else "",
        "old_predicted_risk": classify_risk(old_visibility) if old_visibility is not None else "",
    })


# ------------------------------------------------------------
# SAVE latest_risk.json (only if ALL cities worked)
# ------------------------------------------------------------
if any(r['risk_level'] == 'UNAVAILABLE' for r in results):
    print("\nWARNING: some cities have no forecast. JSON NOT updated, old file kept.")
else:
    for path in (SAVE_PATH, BACKUP_PATH):
        if not path:
            continue
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nlatest_risk.json updated!")
    print(json.dumps(results, indent=2, ensure_ascii=False))


# ------------------------------------------------------------
# ADD THIS RUN TO prediction_log.csv
# ------------------------------------------------------------
def upgrade_log_columns():
    """If the log file has the old set of columns, rewrite it with the new
    columns. Old rows keep all their values. New columns are left empty."""
    if not os.path.exists(LOG_PATH):
        return
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if header == LOG_COLUMNS:
        return
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in LOG_COLUMNS})


if log_rows:
    upgrade_log_columns()
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nAdded {len(log_rows)} predictions to prediction_log.csv")


# ------------------------------------------------------------
# CHECK OLD PREDICTIONS (verify.py in the same folder)
# ------------------------------------------------------------
try:
    import verify
    verify.run(with_chart=False)
except Exception as e:
    print(f"\nVerification skipped: {safe(e)}")
