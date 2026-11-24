"""
current_conditions.py
Gets the CURRENT weather and air pollution for the 4 cities from OpenWeather
and saves them in current_conditions.json.

GitHub Actions runs this every hour, together with run_forecast.py.
The app reads current_conditions.json directly from GitHub,
so no server is needed.

The format is the same as the /current address of live_server.py:
    {"source": ..., "updated_at": ..., "data": [ one entry per city ]}
"""

import json
import os
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "current_conditions.json")
KEY_FILE = os.path.join(HERE, "api_key.txt")

CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479),
    "Faisalabad": (31.4504, 73.1350),
    "Multan": (30.1575, 71.5249),
}

RECOMMENDATION = {
    "CRITICAL": "Motorway Closure Required",
    "HIGH": "Speed Limit 40 km/h - Caution Required",
    "MODERATE": "Speed Limit 60 km/h - Caution Required",
    "LOW": "Reduced Speed Advised - Monitor Conditions",
    "SAFE": "Normal Operations",
}

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}


def classify_risk(v):
    if v < 0.5: return "CRITICAL"
    elif v < 2.0: return "HIGH"
    elif v < 5.0: return "MODERATE"
    elif v < 10.0: return "LOW"
    else: return "SAFE"


def load_key():
    key = os.environ.get("OWM_API_KEY", "").strip()
    if key:
        return key
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    api_key = load_key()
    results = []

    for city, (lat, lon) in CITIES.items():
        try:
            weather = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
                timeout=20,
            )
            weather.raise_for_status()
            weather = weather.json()

            pollution = requests.get(
                "https://api.openweathermap.org/data/2.5/air_pollution",
                params={"lat": lat, "lon": lon, "appid": api_key},
                timeout=20,
            )
            pollution.raise_for_status()
            pollution = pollution.json()["list"][0]

            visibility_km = round(weather["visibility"] / 1000, 2)
            risk = classify_risk(visibility_km)
            aqi = pollution["main"]["aqi"]
            comp = pollution["components"]

            results.append({
                "city": city,
                "datetime": datetime.now().astimezone().isoformat(),
                "visibility_km": visibility_km,
                "risk_level": risk,
                "recommendation": RECOMMENDATION[risk],
                "aqi": aqi,
                "aqi_label": AQI_LABELS.get(aqi, "Unknown"),
                "pm2_5": comp.get("pm2_5"),
                "pm10": comp.get("pm10"),
                "no2": comp.get("no2"),
                "o3": comp.get("o3"),
                "so2": comp.get("so2"),
                "co": comp.get("co"),
                "temperature_c": weather.get("main", {}).get("temp"),
                "humidity": weather.get("main", {}).get("humidity"),
            })
            print(f"{city}: visibility {visibility_km} km, AQI {aqi}")
        except Exception as e:
            print(f"{city}: could not get current data ({str(e).replace(api_key, '***KEY***')})")

    if len(results) < len(CITIES):
        print("Some cities failed. Old current_conditions.json kept.")
        return

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "source": "OpenWeather Current Weather and Air Pollution (updated every hour)",
            "updated_at": datetime.now().astimezone().isoformat(),
            "data": results,
        }, f, indent=2, ensure_ascii=False)
    print("current_conditions.json updated!")


if __name__ == "__main__":
    main()
